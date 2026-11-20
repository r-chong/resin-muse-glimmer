# Portions are adapted from Hugging Face Transformers (Apache-2.0),
# src/transformers/models/muse_glimmer/modeling_muse_glimmer.py and
# src/transformers/vision_utils.py at v5.19.0.

"""Native Muse Glimmer vision tower for packed still-image patches.

Input patches arrive in raster order, one row per 14x14 patch, laid out as
temporal/channel/height/width. The tower runs window attention over 32x32
patch windows (every fourth layer and the last are full attention per image),
then folds each 2x2 block of patches into one decoder token.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resin_models_torch.muse_glimmer.config import MuseGlimmerVisionConfig


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _grids(grid_thw: Tensor) -> list[tuple[int, int, int]]:
    grids = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    if any(t < 1 or h < 1 or w < 1 for t, h, w in grids):
        raise ValueError("image grids must be positive")
    return grids  # type: ignore[return-value]


def image_segments(grid_thw: Tensor) -> list[int]:
    """Full-attention segment lengths: one per frame of each image."""

    return [h * w for t, h, w in _grids(grid_thw) for _ in range(t)]


def window_index(grid_thw: Tensor, window: int) -> tuple[Tensor, list[int]]:
    """Raster patches regrouped into ``window x window`` tiles, with tile sizes."""

    order: list[Tensor] = []
    lengths: list[int] = []
    offset = 0
    for t, h, w in _grids(grid_thw):
        index = torch.arange(t * h * w).reshape(t, h, w)
        pad_h = window - h % window
        pad_w = window - w % window
        tiles_h = (h + pad_h) // window
        tiles_w = (w + pad_w) // window
        padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
        padded = padded.reshape(t, tiles_h, window, tiles_w, window)
        padded = padded.permute(0, 1, 3, 2, 4).reshape(t, tiles_h * tiles_w, -1)
        for tile in padded.reshape(-1, window * window):
            valid = tile[tile != -100]
            if valid.numel():
                order.append(valid + offset)
                lengths.append(int(valid.numel()))
        offset += t * h * w
    return torch.cat(order), lengths


def raster_positions(grid_thw: Tensor) -> Tensor:
    """(row, column) of each raster patch, repeated per frame."""

    rows = []
    for t, h, w in _grids(grid_thw):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        rows.append(torch.stack((y.flatten(), x.flatten()), dim=-1).repeat(t, 1))
    return torch.cat(rows)


def interpolation_taps(grid_thw: Tensor, side: int) -> tuple[Tensor, Tensor]:
    """Bilinear taps into the square position table (align_corners=False,
    zero padding), matching ``F.grid_sample`` sampling of the learned grid."""

    rows, cols, heights, widths = [], [], [], []
    for t, h, w in _grids(grid_thw):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        rows.append(y.flatten().repeat(t))
        cols.append(x.flatten().repeat(t))
        heights.append(torch.full((t * h * w,), h))
        widths.append(torch.full((t * h * w,), w))

    def axis(index: Tensor, size: Tensor) -> tuple[Tensor, Tensor]:
        source = (index.float() + 0.5) * side / size.float() - 0.5
        floor = torch.floor(source)
        offsets = torch.arange(2)
        raw = floor.long()[:, None] + offsets
        taps = raw.clamp(0, side - 1)
        weights = (1 - (source[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
        weights = weights * ((raw >= 0) & (raw <= side - 1))
        return taps, weights

    h_taps, h_weights = axis(torch.cat(rows), torch.cat(heights))
    w_taps, w_weights = axis(torch.cat(cols), torch.cat(widths))
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def merge_index(grid_thw: Tensor, merge: int) -> Tensor:
    """Gather order that groups each ``merge x merge`` block contiguously."""

    order = []
    offset = 0
    for t, h, w in _grids(grid_thw):
        if h % merge or w % merge:
            raise ValueError("image grids must be divisible by the merge size")
        block = torch.arange(h * w).view(h // merge, merge, w // merge, merge)
        block = block.permute(0, 2, 1, 3).reshape(-1)
        block = (block[None] + torch.arange(t)[:, None] * h * w).reshape(-1)
        order.append(block + offset)
        offset += t * h * w
    return torch.cat(order)


class PatchEmbedder(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.side = config.pos_emb_height
        self.patch_embedding = nn.Linear(
            config.patch_width, config.hidden_size, bias=False
        )
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        weight = self.patch_embedding.weight
        embeddings = self.patch_embedding(pixel_values.to(weight.dtype))
        indices, weights = interpolation_taps(grid_thw, self.side)
        indices = indices.to(weight.device)
        weights = weights.to(weight.device)
        positions = (self.position_embedding_table(indices) * weights[:, :, None]).sum(
            1
        )
        return embeddings + positions.to(embeddings.dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)

    def forward(
        self, hidden: Tensor, segments: list[int], cos: Tensor, sin: Tensor
    ) -> Tensor:
        length = hidden.shape[0]
        shape = (length, self.num_heads, self.head_dim)
        queries = self.q_proj(hidden).view(shape)
        keys = self.k_proj(hidden).view(shape)
        values = self.v_proj(hidden).view(shape)
        dtype = queries.dtype
        rope_cos = cos.unsqueeze(-2).float()
        rope_sin = sin.unsqueeze(-2).float()
        queries = (
            queries.float() * rope_cos + _rotate_half(queries.float()) * rope_sin
        ).to(dtype)
        keys = (keys.float() * rope_cos + _rotate_half(keys.float()) * rope_sin).to(
            dtype
        )
        outputs = []
        start = 0
        for size in segments:
            rows = slice(start, start + size)
            outputs.append(
                F.scaled_dot_product_attention(
                    queries[rows].transpose(0, 1)[None],
                    keys[rows].transpose(0, 1)[None],
                    values[rows].transpose(0, 1)[None],
                    scale=self.scale,
                )[0].transpose(0, 1)
            )
            start += size
        return self.proj(torch.cat(outputs).reshape(length, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


class VisionLayer(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        # HF hard-codes these two epsilons rather than reading the config.
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self, hidden: Tensor, segments: list[int], cos: Tensor, sin: Tensor
    ) -> Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), segments, cos, sin)
        return hidden + self.mlp(self.norm2(hidden))


class MuseGlimmerVisionModel(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            VisionLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def rotary(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Axial 2-D RoPE: the same frequencies for both axes, over the full head."""

        spatial = self.config.head_dim // 2
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0, spatial, 2, dtype=torch.float32, device=positions.device
                )
                / spatial
            )
        )
        freqs = positions.float()[..., None] * inv_freq
        cos, sin = freqs.cos(), freqs.sin()

        def recompose(table: Tensor) -> Tensor:
            both = torch.cat((table[:, 0], table[:, 1]), dim=-1)
            return torch.cat((both, both), dim=-1).to(dtype)

        return recompose(cos), recompose(sin)

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        grid_thw = grid_thw.cpu()
        device = self.ln_pre.weight.device
        dtype = self.patch_embedder.patch_embedding.weight.dtype
        order, window_segments = window_index(grid_thw, self.config.window_patches)
        segments = image_segments(grid_thw)
        hidden = self.ln_pre(self.patch_embedder(pixel_values.to(device), grid_thw))
        order = order.to(device)
        hidden = hidden[order]
        # HF flips (row, col) to (col, row) and offsets by one before RoPE.
        positions = (raster_positions(grid_thw).flip(-1) + 1).to(device)[order]
        cos, sin = self.rotary(positions, dtype)
        for layer, kind in zip(self.layers, self.config.layer_types, strict=True):
            hidden = layer(
                hidden,
                segments if kind == "full_attention" else window_segments,
                cos,
                sin,
            )
        hidden = hidden[torch.argsort(order)]
        hidden = self.ln_post(hidden)
        merge = self.config.merge_size
        hidden = hidden[merge_index(grid_thw, merge).to(device)]
        width = hidden.shape[-1]
        return (
            hidden.view(-1, merge * merge, width)
            .permute(0, 2, 1)
            .reshape(-1, width * merge * merge)
        )
