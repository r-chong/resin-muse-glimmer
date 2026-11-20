"""Exact-shape CUDA graphs for stateless Muse decode tensor regions.

Cache addressing, attention, and sampling remain on the ordinary runner path.
No context length or page-table value is part of a captured region. All graphs
run serially on the current stream and share the existing Resin capture pool.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from resin.backend.cuda.graphs import CudaGraphExecutor, CudaGraphMode, CudaGraphStep


@dataclass(frozen=True)
class _Region:
    name: str
    operation: Callable
    inputs: tuple[tuple[tuple[int, ...], torch.dtype], ...]
    outputs: tuple[tuple[tuple[int, ...], torch.dtype], ...]


class MuseDecodeGraphs:
    """Capture exact batches, without padding or request-dependent state.

    Projection outputs are consumed by eager attention before another projection
    graph runs. Feed-forward output is consumed before another feed-forward graph
    overwrites its shared output buffer. Two concurrent forwards are unsupported,
    as with the reference runner's existing cache arena.
    """

    def __init__(self, model, *, batches: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        weight = model.lm_head.weight
        if weight.device.type != "cuda":
            raise ValueError("Muse decode graphs require a CUDA model")
        if (
            not batches
            or any(n <= 0 for n in batches)
            or len(set(batches)) != len(batches)
        ):
            raise ValueError("decode graph batches must be distinct positive integers")
        self.device, self.dtype = weight.device, weight.dtype
        self.batches = tuple(sorted(batches))
        self.regions: list[_Region] = []
        self.buffers = {}
        self.fallbacks = Counter()
        text = model.model.language_model
        hidden = ((1, text.config.hidden_size), self.dtype)
        position = ((1,), torch.int64)
        for layer in text.layers:
            attention = layer.self_attn
            index = len(self.regions)
            self.regions.append(
                _Region(
                    "qkv_rope",
                    attention._project,
                    (hidden, position),
                    (
                        ((attention.num_heads, 1, attention.head_dim), self.dtype),
                        ((attention.num_kv_heads, 1, attention.head_dim), self.dtype),
                        ((attention.num_kv_heads, 1, attention.head_dim), self.dtype),
                    ),
                )
            )
            attention._project = lambda h, p, i=index: self.invoke(i, h, p)
            index = len(self.regions)
            self.regions.append(
                _Region("feed_forward", layer._feed_forward, (hidden,), (hidden,))
            )
            layer._feed_forward = lambda h, i=index: self.invoke(i, h)
        self.executor = CudaGraphExecutor(
            self.device,
            {
                "decode_regions": CudaGraphMode(
                    tuple(
                        (n, i + 1)
                        for i in range(len(self.regions))
                        for n in self.batches
                    ),
                    self._build,
                    lambda shape: self._buffers(shape[0], shape[1] - 1)[0],
                )
            },
            synchronize_failure=lambda: torch.cuda.synchronize(self.device),
        )

    def _buffers(self, batch: int, index: int):
        region = self.regions[index]
        # Regions of a matching layout execute serially, so all layers share
        # fixed inputs/outputs. Captured temporary allocations share Resin's pool.
        key = batch, region.inputs, region.outputs
        if key not in self.buffers:

            def allocate(layout):
                shape, dtype = layout
                return torch.zeros((batch, *shape), dtype=dtype, device=self.device)

            inputs = {
                str(i): allocate(layout) for i, layout in enumerate(region.inputs)
            }
            outputs = tuple(allocate(layout) for layout in region.outputs)
            self.buffers[key] = inputs, outputs
        return self.buffers[key]

    def _build(self, shape: tuple[int, int]) -> CudaGraphStep:
        batch, one_based_index = shape
        index = one_based_index - 1
        region = self.regions[index]
        inputs, outputs = self._buffers(batch, index)

        def launch():
            result = region.operation(
                *(inputs[str(i)] for i in range(len(region.inputs)))
            )
            values = result if isinstance(result, tuple) else (result,)
            for output, value in zip(outputs, values, strict=True):
                output.copy_(value)

        return CudaGraphStep(inputs, outputs, launch)

    def invoke(self, index: int, *values: Tensor):
        region = self.regions[index]
        batch = values[0].shape[0]
        if batch not in self.batches or any(
            tuple(value.shape) != (batch, *layout[0])
            or value.dtype != layout[1]
            or value.device != self.device
            for value, layout in zip(values, region.inputs, strict=True)
        ):
            self.fallbacks["shape_or_layout"] += 1
            return region.operation(*values)
        output = self.executor.mode("decode_regions").run(
            (batch, index + 1), {str(i): value for i, value in enumerate(values)}
        )
        return output if len(region.outputs) > 1 else output[0]

    def capture(self):
        # These operations do not read or write the request cache, so capture
        # cannot initialize, advance, or corrupt a request's KV state.
        with torch.inference_mode():
            return self.executor.capture_all()

    @property
    def counts(self):
        counts = self.executor.runtime_counts
        row = counts["decode_regions"]
        row["eager_fallback_count"] += sum(self.fallbacks.values())
        for reason, count in self.fallbacks.items():
            row["fallback_counts"][reason] = (
                row["fallback_counts"].get(reason, 0) + count
            )
        return counts

    @property
    def stats(self):
        stats = self.executor.stats
        stats["decode_regions"].update(
            capture_scope="stateless_decode_tensor_regions",
            captured_operations=(
                "qkv_projection_qk_norm_rope",
                "feed_forward_norms_residual",
            ),
            **self.counts["decode_regions"],
        )
        return stats

    def close(self):
        self.executor.close()
