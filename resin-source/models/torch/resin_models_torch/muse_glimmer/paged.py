"""Engine-managed paged KV for Muse Glimmer (Resin cache contract).

Two cache groups match the decoder's layer split:

- group 0, full HISTORY: the 13 full-attention (NoPE) layers;
- group 1, WINDOW (read window 2,048): the 39 sliding-window layers.

Because 39 = 3 x 13, one slab holds one window block (39 layers x 64 tokens)
or exactly three history blocks (13 layers x 64 tokens), with no padding. The
Engine owns allocation, retention, and lifetime; this module only maps the
Engine's page addresses onto one device arena.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from resin import CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.plans import BatchView, CacheGroupView
from resin_models_torch.muse_glimmer.config import MuseGlimmerTextConfig

BLOCK_TOKENS = 64
HISTORY_GROUP = 0
WINDOW_GROUP = 1


@dataclass(frozen=True)
class MuseCacheGeometry:
    full_layers: tuple[int, ...]
    sliding_layers: tuple[int, ...]
    kv_heads: int
    head_dim: int
    read_window: int
    window_tokens: int

    @classmethod
    def from_config(cls, config: MuseGlimmerTextConfig, *, chunk_tokens: int):
        full = tuple(
            i for i, t in enumerate(config.layer_types) if t == "full_attention"
        )
        sliding = tuple(
            i for i, t in enumerate(config.layer_types) if t == "sliding_attention"
        )
        if len(sliding) % len(full):
            raise ValueError(
                "slab packing assumes sliding layers are a multiple of full"
            )
        # A chunk of Q writes with read window W needs W + Q - 1 retained tokens.
        return cls(
            full,
            sliding,
            config.num_key_value_heads,
            config.head_dim,
            config.sliding_window,
            config.sliding_window + chunk_tokens - 1,
        )

    @property
    def history_blocks_per_slab(self) -> int:
        return len(self.sliding_layers) // len(self.full_layers)

    def slab_bytes(self, element_bytes: int = 2) -> int:
        return (
            len(self.sliding_layers) * 2 * BLOCK_TOKENS * self.kv_heads * self.head_dim
        ) * element_bytes

    def layout(self, parent_capacity: int) -> CacheLayout:
        return CacheLayout(
            parent_capacity=parent_capacity,
            groups=[
                CacheGroupLayout(
                    BLOCK_TOKENS, blocks_per_slab=self.history_blocks_per_slab
                ),
                CacheGroupLayout(
                    BLOCK_TOKENS,
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.window_tokens,
                    read_window_tokens=self.read_window,
                ),
            ],
        )


class MuseKvArena:
    """One device allocation, viewed per group. Slab 0 is the null slab."""

    def __init__(
        self, geometry: MuseCacheGeometry, parent_capacity: int, *, device, dtype,
        native_decode: bool = False, num_query_heads: int | None = None,
    ):
        self.geometry = geometry
        self.native_decode = native_decode
        self.num_query_heads = num_query_heads
        self._native_decoder = None
        self.decode_attention = None
        g = geometry
        slabs = parent_capacity + 1
        self.storage = torch.zeros(
            (slabs, len(g.sliding_layers), 2, BLOCK_TOKENS, g.kv_heads, g.head_dim),
            dtype=dtype,
            device=device,
        )
        self.window = self.storage  # [slab, layer, k/v, token, head, dim]
        self.history = self.storage.view(
            slabs * g.history_blocks_per_slab,
            len(g.full_layers),
            2,
            BLOCK_TOKENS,
            g.kv_heads,
            g.head_dim,
        )
        self._slot = {
            layer: (HISTORY_GROUP, i) for i, layer in enumerate(g.full_layers)
        }
        self._slot.update(
            {layer: (WINDOW_GROUP, i) for i, layer in enumerate(g.sliding_layers)}
        )

    def layer_views(self, layer: int) -> tuple[int, Tensor, Tensor]:
        """(group, keys, values), each indexed [page, token_in_block, head, dim]."""
        group, index = self._slot[layer]
        table = self.history if group == HISTORY_GROUP else self.window
        return group, table[:, index, 0], table[:, index, 1]

    def prepare_forward(self, rows, query_length: int) -> None:
        """Plan native one-token attention once before running all layers."""
        self.decode_attention = None
        if not self.native_decode or query_length != 1:
            return
        if self._native_decoder is None:
            from resin_models_torch.muse_glimmer.flashinfer import FlashinferPagedDecode

            if self.num_query_heads is None:
                raise ValueError("native decode needs the model's query-head count")
            self._native_decoder = FlashinferPagedDecode(
                self.geometry, self.num_query_heads,
                device=self.storage.device, dtype=self.storage.dtype,
            )
        self._native_decoder.plan(rows)
        self.decode_attention = self._native_decoder


@dataclass(frozen=True)
class GroupRow:
    pages: Tensor  # [blocks] packed page ids, chronological
    first_position: int  # absolute position of pages[0], token 0
    append_pages: Tensor  # [query] destination page per new token
    append_offsets: Tensor  # [query] offset within that page


@dataclass(frozen=True)
class PagedRow:
    """One scheduled sequence's cache addresses for both groups."""

    groups: tuple[GroupRow, GroupRow]
    end: int  # context + query length: positions >= end are unwritten


def paged_rows(batch: BatchView, parent_capacity: int, device) -> list[PagedRow]:
    views = [
        CacheGroupView.from_plan(
            batch,
            group_index=g,
            parent_capacity=parent_capacity,
            block_tokens=BLOCK_TOKENS,
        )
        for g in (HISTORY_GROUP, WINDOW_GROUP)
    ]
    packed = [view.packed_block_ids.astype(np.int64) for view in views]
    appended = [view.packed_append_ids.astype(np.int64) for view in views]
    rows = []
    for row in range(batch.num_sequences):
        tokens = batch.token_slice(row)
        groups = []
        for view, blocks, appends in zip(views, packed, appended):
            groups.append(
                GroupRow(
                    torch.as_tensor(blocks[view.block_slice(row)], device=device),
                    int(view.block_table_start_indices[row]) * BLOCK_TOKENS,
                    torch.as_tensor(appends[tokens], device=device),
                    torch.as_tensor(
                        view.append_block_offsets[tokens].astype(np.int64),
                        device=device,
                    ),
                )
            )
        end = int(batch.context_lengths[row]) + int(batch.query_lengths[row])
        rows.append(PagedRow((groups[0], groups[1]), end))
    return rows
