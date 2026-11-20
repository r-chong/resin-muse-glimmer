"""Native decode attention over the existing Engine-owned Muse KV arena.

Only one-token queries use this adapter. The eager SDPA prefill/reference path
is unchanged, and projections, QK norms, RoPE and output gating stay in Torch.
"""

from __future__ import annotations

import torch

from resin_models_torch.muse_glimmer.paged import BLOCK_TOKENS, WINDOW_GROUP


def decode_page_lengths(rows, group: int) -> tuple[list[int], list[int]]:
    """Return live page counts and last-page lengths, excluding reserved tails.

    A window group's first page can represent a nonzero absolute position.
    FlashInfer sees only the retained, chronological suffix; for a one-token
    query its relative last-token/window mask is therefore the same mask.
    """
    counts, last_lengths = [], []
    for row in rows:
        cache = row.groups[group]
        length = row.end - cache.first_position
        if length <= 0:
            raise ValueError("decode attention needs at least one live cached token")
        count = (length + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        if count > cache.pages.numel():
            raise ValueError("decode page table does not cover the live KV suffix")
        counts.append(count)
        last_lengths.append((length - 1) % BLOCK_TOKENS + 1)
    return counts, last_lengths


class FlashinferPagedDecode:
    """Two reusable FA2 plans, shared by all layers in each retention group.

    Float workspace is shared because the two groups/layers execute serially
    on the caller's current stream. No CUDA graph or overlap is introduced.
    Strided K/V views are passed directly; no dense history or repeated heads
    are materialized.
    """

    def __init__(self, geometry, num_query_heads: int, *, device, dtype):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        self.geometry = geometry
        self.num_query_heads = num_query_heads
        self.device = device
        self.dtype = dtype
        self.workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=device)
        self.wrappers = tuple(
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, "NHD", backend="fa2", use_tensor_cores=True
            )
            for _ in range(2)
        )
        self.plan_calls = 0
        self.layer_calls = 0

    def plan(self, rows) -> None:
        g = self.geometry
        for group, wrapper in enumerate(self.wrappers):
            counts, last_lengths = decode_page_lengths(rows, group)
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            # Metadata is prepared once per group/forward, not in each layer.
            indices = torch.cat(
                [row.groups[group].pages[:count] for row, count in zip(rows, counts)]
            ).to(torch.int32)
            wrapper.plan(
                torch.tensor(offsets, dtype=torch.int32, device=self.device),
                indices,
                torch.tensor(last_lengths, dtype=torch.int32, device=self.device),
                self.num_query_heads,
                g.kv_heads,
                g.head_dim,
                BLOCK_TOKENS,
                pos_encoding_mode="NONE",
                window_left=g.read_window - 1 if group == WINDOW_GROUP else -1,
                sm_scale=g.head_dim**-0.5,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
            )
            self.plan_calls += 1

    def run(self, group: int, queries, key_pages, value_pages):
        self.layer_calls += 1
        return self.wrappers[group].run(queries, (key_pages, value_pages))
