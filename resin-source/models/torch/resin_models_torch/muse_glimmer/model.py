# Portions are adapted from Hugging Face Transformers (Apache-2.0),
# src/transformers/models/muse_glimmer/modeling_muse_glimmer.py at v5.19.0.

"""Readable Muse Glimmer text decoder with request-owned attention state.

Module names mirror the checkpoint so safetensors stream directly into the
tree. Numerics follow the Transformers reference: norms in FP32 cast back to
the activation dtype, RoPE tables cast to the activation dtype, and softcapped
logits computed in the activation dtype.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin_models_torch.muse_glimmer.config import (
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
)
from resin_models_torch.muse_glimmer.vision import MuseGlimmerVisionModel


class AttentionState:
    """Post-norm, post-RoPE keys and values for one layer of one request.

    Sliding layers retain only the ``window - 1`` most recent positions, which
    is everything a later query can attend to. ``start`` is the absolute
    position of the first retained key.
    """

    def __init__(self, window: int | None) -> None:
        self.window = window
        self.keys: Tensor | None = None
        self.values: Tensor | None = None
        self.start = 0

    @property
    def end(self) -> int:
        return self.start + (0 if self.keys is None else self.keys.shape[-2])

    def extend(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor, int]:
        """Return every retained key including the new ones, then trim."""

        if self.keys is None:
            all_keys, all_values = keys, values
        else:
            assert self.values is not None
            all_keys = torch.cat((self.keys, keys), dim=-2)
            all_values = torch.cat((self.values, values), dim=-2)
        start = self.start
        keep = all_keys.shape[-2]
        if self.window is not None:
            keep = min(keep, self.window - 1)
        drop = all_keys.shape[-2] - keep
        self.keys = all_keys[..., drop:, :]
        self.values = all_values[..., drop:, :]
        self.start = start + drop
        return all_keys, all_values, start


class MuseGlimmerState:
    """One request's attention state for every decoder layer."""

    def __init__(self, config: MuseGlimmerTextConfig) -> None:
        self.layers = [
            AttentionState(
                config.sliding_window if kind == "sliding_attention" else None
            )
            for kind in config.layer_types
        ]
        self.length = 0


def weightless_rms_norm(hidden: Tensor, eps: float) -> Tensor:
    normalized = hidden.float()
    normalized = normalized * torch.pow(
        normalized.pow(2).mean(-1, keepdim=True) + eps, -0.5
    )
    return normalized.type_as(hidden)


class ScaledRMSNorm(nn.Module):
    """``x * w`` RMSNorm (the decoder's final norm)."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = hidden.float()
        normalized = normalized * torch.pow(
            normalized.pow(2).mean(-1, keepdim=True) + self.eps, -0.5
        )
        return (normalized * self.weight.float()).type_as(hidden)


class CenteredRMSNorm(nn.Module):
    """``x * (1 + w)`` RMSNorm, computed in FP32 before the output cast."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).type_as(hidden)


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer: int) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        self.qk_scale = config.qk_scale_factor
        self.window = (
            config.sliding_window
            if config.layer_types[layer] == "sliding_attention"
            else None
        )
        self.rope_theta = config.layer_rope_theta[layer]
        heads = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(width, heads, bias=False)
        self.k_proj = nn.Linear(width, kv, bias=False)
        self.v_proj = nn.Linear(width, kv, bias=False)
        self.gate_proj = nn.Linear(width, heads, bias=False)
        self.o_proj = nn.Linear(heads, width, bias=False)
        self.register_buffer("inv_freq", self._frequencies("cpu"), persistent=False)

    def _frequencies(self, device) -> Tensor:
        if not self.rope_theta:
            return torch.empty(0, device=device)
        exponents = torch.arange(
            0, self.head_dim, 2, device=device, dtype=torch.float32
        )
        return 1.0 / self.rope_theta ** (exponents / self.head_dim)

    def materialize(self, device) -> None:
        self.inv_freq = self._frequencies(device)

    def _project(
        self, hidden: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = hidden.shape
        queries = (
            self.q_proj(hidden).view(batch, length, -1, self.head_dim).transpose(1, 2)
        )
        keys = (
            self.k_proj(hidden).view(batch, length, -1, self.head_dim).transpose(1, 2)
        )
        values = (
            self.v_proj(hidden).view(batch, length, -1, self.head_dim).transpose(1, 2)
        )
        queries = weightless_rms_norm(queries, self.eps) * self.qk_scale
        keys = weightless_rms_norm(keys, self.eps)
        if self.rope_theta:
            angles = positions.float()[..., None] * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
            cos = angles.cos().to(hidden.dtype).unsqueeze(1)
            sin = angles.sin().to(hidden.dtype).unsqueeze(1)
            queries = queries * cos + _rotate_half(queries) * sin
            keys = keys * cos + _rotate_half(keys) * sin
        return queries, keys, values

    def _mask(self, query_positions: Tensor, key_positions: Tensor) -> Tensor:
        distance = query_positions[..., :, None] - key_positions[..., None, :]
        allowed = distance >= 0
        if self.window is not None:
            allowed &= distance < self.window
        return allowed

    def _output(self, attended: Tensor, hidden: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        attended = attended * torch.sigmoid(self.gate_proj(hidden))
        return self.o_proj(attended)

    def forward(
        self, hidden: Tensor, positions: Tensor, state: AttentionState
    ) -> Tensor:
        """Attend one contiguous chunk of a single request."""

        queries, keys, values = self._project(hidden, positions)
        keys, values, start = state.extend(keys, values)
        key_positions = torch.arange(
            start, start + keys.shape[-2], device=hidden.device
        )
        mask = self._mask(positions[0], key_positions)
        attended = F.scaled_dot_product_attention(
            queries,
            keys.repeat_interleave(self.kv_groups, dim=1),
            values.repeat_interleave(self.kv_groups, dim=1),
            attn_mask=mask,
            scale=self.scale,
        )
        return self._output(attended, hidden)

    def forward_batch(
        self, hidden: Tensor, positions: Tensor, states: Sequence[AttentionState]
    ) -> Tensor:
        """Decode one token for each independently cached request."""

        batch, length, _ = hidden.shape
        if length != 1 or len(states) != batch:
            raise ValueError("batched attention requires one state and token per row")
        queries, new_keys, new_values = self._project(hidden, positions)
        rows = [
            state.extend(new_keys[i : i + 1], new_values[i : i + 1])
            for i, state in enumerate(states)
        ]
        width = max(keys.shape[-2] for keys, _, _ in rows)
        padded_keys, padded_values, key_positions = [], [], []
        for keys, values, start in rows:
            pad = width - keys.shape[-2]
            padded_keys.append(F.pad(keys, (0, 0, 0, pad)))
            padded_values.append(F.pad(values, (0, 0, 0, pad)))
            # Padding sits at future positions, which the causal mask removes.
            key_positions.append(
                torch.arange(start, start + width, device=hidden.device)
                + torch.where(
                    torch.arange(width, device=hidden.device) < keys.shape[-2],
                    0,
                    1 << 30,
                )
            )
        mask = self._mask(positions, torch.stack(key_positions))[:, None]
        attended = F.scaled_dot_product_attention(
            queries,
            torch.cat(padded_keys).repeat_interleave(self.kv_groups, dim=1),
            torch.cat(padded_values).repeat_interleave(self.kv_groups, dim=1),
            attn_mask=mask,
            scale=self.scale,
        )
        return self._output(attended, hidden)


class MLP(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer: int) -> None:
        super().__init__()
        width = config.hidden_size
        self.self_attn = Attention(config, layer)
        self.mlp = MLP(config)
        self.input_layernorm = CenteredRMSNorm(width, config.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(width, config.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(width, config.rms_norm_eps)
        self.post_feedforward_layernorm = CenteredRMSNorm(width, config.post_norm_eps)

    def _feed_forward(self, hidden: Tensor) -> Tensor:
        update = self.mlp(self.pre_feedforward_layernorm(hidden))
        return hidden + self.post_feedforward_layernorm(update)

    def forward(
        self, hidden: Tensor, positions: Tensor, state: AttentionState
    ) -> Tensor:
        update = self.self_attn(self.input_layernorm(hidden), positions, state)
        return self._feed_forward(hidden + self.post_attention_layernorm(update))

    def forward_batch(
        self, hidden: Tensor, positions: Tensor, states: Sequence[AttentionState]
    ) -> Tensor:
        update = self.self_attn.forward_batch(
            self.input_layernorm(hidden), positions, states
        )
        return self._feed_forward(hidden + self.post_attention_layernorm(update))


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer) for layer in range(config.num_hidden_layers)
        )
        self.norm = ScaledRMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed(self, tokens: Tensor) -> Tensor:
        return weightless_rms_norm(self.embed_tokens(tokens), self.config.rms_norm_eps)

    def forward(
        self, hidden: Tensor, positions: Tensor, state: MuseGlimmerState
    ) -> Tensor:
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            hidden = layer(hidden, positions, layer_state)
        return self.norm(hidden)

    def forward_batch(
        self, hidden: Tensor, positions: Tensor, states: Sequence[MuseGlimmerState]
    ) -> Tensor:
        for index, layer in enumerate(self.layers):
            hidden = layer.forward_batch(
                hidden, positions, [state.layers[index] for state in states]
            )
        return self.norm(hidden)


class VisionAdapter(nn.Module):
    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.out_hidden_size, config.projector_hidden_size, bias=False
        )
        self.fc2 = nn.Linear(
            config.projector_hidden_size, config.projector_hidden_size, bias=False
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return F.gelu(self.fc2(F.gelu(self.fc1(hidden))))


class MuseGlimmerModel(nn.Module):
    def __init__(
        self, config: MuseGlimmerConfig, *, include_vision: bool = True
    ) -> None:
        super().__init__()
        self.language_model = MuseGlimmerTextModel(config.text)
        if include_vision:
            self.vision_tower = MuseGlimmerVisionModel(config.vision)
            self.vision_adapter = VisionAdapter(config)
            self.vision_projection = nn.Linear(
                config.projector_hidden_size, config.text.hidden_size, bias=False
            )


class MuseGlimmerForConditionalGeneration(nn.Module):
    """Checkpoint-shaped Muse Glimmer with a request-state text forward."""

    def __init__(
        self, config: MuseGlimmerConfig, *, include_vision: bool = True
    ) -> None:
        super().__init__()
        self.config = config
        self.include_vision = include_vision
        self.model = MuseGlimmerModel(config, include_vision=include_vision)
        self.lm_head = nn.Linear(
            config.text.hidden_size, config.text.vocab_size, bias=False
        )

    def make_state(self) -> MuseGlimmerState:
        return MuseGlimmerState(self.config.text)

    def materialize_buffers(self, device) -> None:
        for module in self.modules():
            if isinstance(module, Attention):
                module.materialize(device)

    def image_features(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        """Project packed image patches to decoder-width embeddings."""

        if not self.include_vision:
            raise ValueError("this Muse model was loaded without its vision tower")
        model = self.model
        hidden = model.vision_tower(pixel_values, grid_thw)
        hidden = model.vision_projection(model.vision_adapter(hidden))
        return weightless_rms_norm(hidden, self.config.text.rms_norm_eps)

    def embed(
        self,
        tokens: Tensor,
        image_embeds: Tensor | None = None,
        image_mask: Tensor | None = None,
    ) -> Tensor:
        """Embed tokens, replacing image placeholders with vision features."""

        placeholders = (tokens == self.config.image_token_id) | (
            tokens == self.config.video_token_id
        )
        hidden = self.model.language_model.embed(tokens.masked_fill(placeholders, 0))
        if image_mask is not None and bool(image_mask.any()):
            if image_embeds is None or image_embeds.shape[0] != int(image_mask.sum()):
                raise ValueError("image features do not match image placeholder count")
            hidden = hidden.clone()
            hidden[image_mask] = image_embeds.to(hidden.dtype)
        return hidden

    def forward(
        self,
        tokens: Tensor,
        state: MuseGlimmerState,
        positions: Tensor,
        *,
        hidden: Tensor | None = None,
    ) -> Tensor:
        """Advance one request by a contiguous chunk and return final hidden rows."""

        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("tokens must contain exactly one sequence")
        if positions.shape != tokens.shape:
            raise ValueError("positions must have the same shape as tokens")
        if int(positions[0, 0]) != state.length:
            raise ValueError("chunk positions must continue the request state")
        hidden = self.embed(tokens) if hidden is None else hidden
        output = self.model.language_model(hidden, positions, state)
        state.length += tokens.shape[1]
        return output

    def forward_batch(
        self, tokens: Tensor, states: Sequence[MuseGlimmerState], positions: Tensor
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 1 or tokens.shape[0] != len(states):
            raise ValueError("batched decode requires one token and state per row")
        output = self.model.language_model.forward_batch(
            self.embed(tokens), positions, states
        )
        for state in states:
            state.length += 1
        return output

    def logits(self, hidden: Tensor) -> Tensor:
        cap = self.config.text.final_logit_softcapping
        logits = self.lm_head(hidden) * self.config.text.output_multiplier
        return torch.tanh(logits / cap) * cap
