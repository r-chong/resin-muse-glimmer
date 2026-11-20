"""Readable Muse Glimmer loading and Engine execution on PyTorch.

This runner prioritizes a transparent, reference-correct forward pass over
speed: request state lives in Python-owned tensors (not Engine pages), attention
uses ``scaled_dot_product_attention`` with explicit masks, and nothing is
graph-captured. It is the correctness baseline for an optimized CUDA runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from resin import RunnerConfig
from resin.backend.loading import (
    load_safetensor_weights,
    read_config,
    resolve_model,
)
from resin.backend.plans import BatchView
from resin.backend.safetensors import indexed_tensor_files
from resin.backend.sampling import (
    PromptScores,
    TorchSampler,
    format_output,
    prompt_scores,
)
from resin.backend.topology import ParallelismCapabilities
from resin.protocols import Runner
from resin_models_torch.muse_glimmer.config import MuseGlimmerConfig
from resin_models_torch.muse_glimmer.model import (
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerState,
)

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput


DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
VISION_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
)
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=list(CHECKPOINT_FILES)
        )
    )


def resolve_model_path(model: str | Path, revision: str | None) -> Path:
    return resolve_model(model, revision, _download_model)


def read_muse_config(model_path: Path) -> MuseGlimmerConfig:
    generation_path = model_path / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text()) if generation_path.exists() else None
    )
    return MuseGlimmerConfig.from_dict(read_config(model_path), generation)


def load_model(
    model_path: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    include_vision: bool = True,
) -> MuseGlimmerForConditionalGeneration:
    config = read_muse_config(model_path)
    target = torch.device(device)
    with torch.device("meta"):
        model = MuseGlimmerForConditionalGeneration(
            config, include_vision=include_vision
        )
    keys = [
        key
        for key in indexed_tensor_files(model_path)
        if include_vision or not key.startswith(VISION_PREFIXES)
    ]
    load_safetensor_weights(
        model_path, model, device=target, dtype=dtype, checkpoint_keys=keys
    )
    if missing := {
        name for name, tensor in model.state_dict().items() if tensor.is_meta
    }:
        raise ValueError(f"checkpoint did not populate: {sorted(missing)[:8]}")
    model.materialize_buffers(target)
    return model.eval()


@dataclass
class _Request:
    state: MuseGlimmerState
    media: Any = None
    image_rows: dict[int, int] | None = None
    image_embeds: torch.Tensor | None = None


class MuseGlimmerTorchRunner(Runner):
    """Muse Glimmer runner consuming Rust-built integer-token batches.

    Uniform one-token decode steps run as one batched forward. Prefill chunks
    and mixed steps run one request at a time.
    """

    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        include_vision: bool = True,
        prefill_chunk_size: int | None = None,
        max_batch_tokens: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_path = resolve_model_path(model, revision)
        self._model = load_model(
            self.model_path,
            device=self.device,
            dtype=dtype,
            include_vision=include_vision,
        )
        self.config = self._model.config
        self.include_vision = include_vision
        self.vocab_size = self.config.text.vocab_size
        self.eos_token_ids = self.config.text.eos_token_ids
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            prefill_chunk_size=prefill_chunk_size,
            max_batch_tokens=max_batch_tokens,
        )
        self._sampler = TorchSampler()
        self._requests: dict[int, _Request] = {}
        self.forward_calls = 0
        self.vision_calls = 0

    # -- request state -----------------------------------------------------

    def _request(self, sequence_id: int, media: Any) -> _Request:
        request = self._requests.get(sequence_id)
        if request is None:
            request = _Request(self._model.make_state(), media)
            self._requests[sequence_id] = request
        elif (
            media is not None
            and request.media is not None
            and request.media is not media
        ):
            raise ValueError("a live Muse Glimmer request changed its media object")
        if media is not None and request.image_rows is None:
            self._prepare_media(request, media)
        return request

    def _prepare_media(self, request: _Request, media: Any) -> None:
        if not self.include_vision:
            raise ValueError("image input requires the Muse vision tower")
        if int(media.image_token_id) != self.config.image_token_id:
            raise ValueError("media image token does not match the checkpoint")
        positions = tuple(int(p) for p in media.image_positions)
        request.media = media
        request.image_rows = {position: row for row, position in enumerate(positions)}
        if not positions:
            return
        pixels = torch.from_numpy(np.asarray(media.pixel_values)).to(self.device)
        grids = torch.from_numpy(np.asarray(media.image_grid_thw))
        with torch.inference_mode():
            embeds = self._model.image_features(pixels, grids)
        self.vision_calls += 1
        if embeds.shape[0] != len(positions):
            raise ValueError("vision features do not match image placeholder count")
        request.image_embeds = embeds

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._requests.pop(int(request_id), None)

    @property
    def live_cache_count(self) -> int:
        return len(self._requests)

    # -- execution ---------------------------------------------------------

    def _validate(self, batch: BatchView, plan: BatchPlan) -> list[_Request]:
        media = plan.media if plan.media else (None,) * batch.num_sequences
        requests = []
        for index, sequence_id in enumerate(batch.sequence_ids):
            request = self._request(int(sequence_id), media[index])
            token_slice = batch.token_slice(index)
            positions = batch.positions[token_slice]
            offset = request.state.length
            if int(batch.context_lengths[index]) != offset or (
                len(positions) and int(positions[0]) != offset
            ):
                raise ValueError(
                    f"non-contiguous positions for request {int(sequence_id)}: "
                    f"state has {offset} tokens, plan starts at {positions[:1].tolist()}"
                )
            if np.any(batch.token_ids[token_slice] >= self.vocab_size):
                raise ValueError(
                    f"token id is out of range for request {int(sequence_id)}"
                )
            requests.append(request)
        return requests

    def _hidden_for(
        self, request: _Request, tokens: torch.Tensor, positions: np.ndarray
    ):
        if request.image_rows:
            rows = [request.image_rows.get(int(p), -1) for p in positions]
            selected = [row for row in rows if row >= 0]
            if selected:
                assert request.image_embeds is not None
                mask = torch.tensor([row >= 0 for row in rows], device=self.device)[
                    None
                ]
                index = torch.tensor(selected, device=self.device)
                return self._model.embed(
                    tokens, request.image_embeds.index_select(0, index), mask
                )
        return self._model.embed(tokens)

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.forward_calls += 1
        batch = BatchView.from_plan(plan)
        requests = self._validate(batch, plan)
        logits_rows: list[torch.Tensor] = []
        prompt_rows: list[PromptScores | None] = []
        with torch.inference_mode():
            uniform_decode = (
                batch.num_sequences > 1
                and bool(np.all(batch.query_lengths == 1))
                and not plan.needs_prompt_logprobs
                and not any(
                    request.image_rows and int(batch.positions[i]) in request.image_rows
                    for i, request in enumerate(requests)
                )
            )
            if uniform_decode:
                tokens = torch.as_tensor(
                    batch.token_ids.astype(np.int64)[:, None], device=self.device
                )
                positions = torch.as_tensor(
                    batch.positions.astype(np.int64)[:, None], device=self.device
                )
                hidden = self._model.forward_batch(
                    tokens, [request.state for request in requests], positions
                )
                logits_rows.append(self._model.logits(hidden[:, -1]))
            else:
                for index, request in enumerate(requests):
                    token_slice = batch.token_slice(index)
                    row_positions = batch.positions[token_slice]
                    tokens = torch.as_tensor(
                        batch.token_ids[token_slice].astype(np.int64)[None],
                        device=self.device,
                    )
                    positions = torch.as_tensor(
                        row_positions.astype(np.int64)[None], device=self.device
                    )
                    hidden = self._model(
                        tokens,
                        request.state,
                        positions,
                        hidden=self._hidden_for(request, tokens, row_positions),
                    )
                    if plan.needs_prompt_logprobs:
                        prompt_rows.append(
                            prompt_scores(
                                plan,
                                hidden[0],
                                self._model.logits,
                                token_start=token_slice.start,
                            )
                        )
                    logits_rows.append(self._model.logits(hidden[:, -1]))
        logits = torch.cat(logits_rows).float()
        scores = PromptScores.concat(cast(list[PromptScores], prompt_rows))
        return format_output(
            logits, self._sampler(logits, plan), plan, prompt_logprobs=scores
        )
