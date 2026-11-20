"""Native prefill qualification against an independent absolute-position mask.

Real GQA geometry and strided arena views cover initial/chunked prefill, page
and window crossings, unequal histories, and unwritten NaN page tails. Full
checkpoint HF parity remains a separate unchanged qualification gate.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from resin_models_torch.muse_glimmer.flashinfer import FlashinferPagedPrefill
from resin_models_torch.muse_glimmer.paged import (
    BLOCK_TOKENS,
    GroupRow,
    MuseCacheGeometry,
    MuseKvArena,
    PagedRow,
)


def _group(pages, first_position, query_start, end, device):
    pages = torch.tensor(pages, dtype=torch.long, device=device)
    append = torch.arange(
        query_start - first_position, end - first_position, device=device
    )
    return GroupRow(
        pages,
        first_position,
        pages[append // BLOCK_TOKENS],
        append % BLOCK_TOKENS,
    )


def _absolute_reference(queries, keys, values, *, query_start, key_start, window):
    """FP32 oracle with bounded temporary memory and explicit absolute masks."""
    key_positions = torch.arange(
        key_start, key_start + len(keys), device=queries.device
    )
    keys = keys.transpose(0, 1)[None].repeat_interleave(16, dim=1).float()
    values = values.transpose(0, 1)[None].repeat_interleave(16, dim=1).float()
    output = []
    for first in range(0, len(queries), 64):
        query = queries[first : first + 64]
        positions = torch.arange(
            query_start + first, query_start + first + len(query), device=queries.device
        )
        mask = key_positions[None] <= positions[:, None]
        if window is not None:
            mask &= key_positions[None] > positions[:, None] - window
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1)[None].float(),
            keys,
            values,
            attn_mask=mask[None, None],
            scale=128**-0.5,
        )
        output.append(attended[0].transpose(0, 1).to(torch.bfloat16))
    return torch.cat(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel qualification")
@pytest.mark.parametrize(
    ("query_length", "ends"),
    [
        (2, (2, 63, 64, 65, 2047, 2048, 2049, 8193)),
        (63, (63, 2049, 8193)),
        (64, (64, 2049, 8193)),
        (65, (65, 2049, 8193)),
        (2049, (2049, 8193)),
        (8192, (8192,)),
    ],
)
def test_native_paged_prefill_matches_absolute_attention(query_length, ends):
    pytest.importorskip("flashinfer")
    torch.manual_seed(456)
    device = "cuda"
    geometry = MuseCacheGeometry(
        full_layers=(0,),
        sliding_layers=(1, 2, 3),
        kv_heads=2,
        head_dim=128,
        read_window=2048,
        window_tokens=10239,
    )
    arena = MuseKvArena(geometry, 600, device=device, dtype=torch.bfloat16)
    arena.storage.fill_(float("nan"))
    rows, data = [], {0: [], 1: []}
    # Keep full and window data in disjoint slab ranges. History page n maps
    # to slab n // 3, so the first range cannot alias the second range.
    next_page = [3, 350]
    for end in ends:
        query_start = end - query_length
        earliest_window_key = max(0, query_start - geometry.read_window + 1)
        first_window = earliest_window_key // BLOCK_TOKENS * BLOCK_TOKENS
        groups = []
        for group, first in enumerate((0, first_window)):
            live = end - first
            count = math.ceil(live / BLOCK_TOKENS)
            pages = list(range(next_page[group], next_page[group] + count + 2))
            next_page[group] += count + 2
            groups.append(_group(pages, first, query_start, end, device))
            keys = torch.randn(live, 2, 128, device=device, dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            _, key_pages, value_pages = arena.layer_views(group)
            slots = torch.arange(live - query_length, device=device)
            page_ids = groups[-1].pages[slots // BLOCK_TOKENS]
            key_pages[page_ids, slots % BLOCK_TOKENS] = keys[:-query_length]
            value_pages[page_ids, slots % BLOCK_TOKENS] = values[:-query_length]
            # Reproduce the model's append-before-read order; every unwritten
            # slot and both reserved future pages remain NaN.
            key_pages[groups[-1].append_pages, groups[-1].append_offsets] = keys[
                -query_length:
            ]
            value_pages[groups[-1].append_pages, groups[-1].append_offsets] = values[
                -query_length:
            ]
            data[group].append((first, keys, values))
        rows.append(PagedRow(tuple(groups), end))

    prefiller = FlashinferPagedPrefill(
        geometry, 32, device=device, dtype=torch.bfloat16
    )
    prefiller.plan(rows, query_length)
    queries = torch.randn(
        len(rows), query_length, 32, 128, device=device, dtype=torch.bfloat16
    )
    for group in (0, 1):
        _, key_pages, value_pages = arena.layer_views(group)
        assert not key_pages.is_contiguous()
        actual = prefiller.run(
            group, queries.flatten(0, 1), key_pages, value_pages
        ).view_as(queries)
        expected = []
        for query, row, (first, keys, values) in zip(queries, rows, data[group]):
            expected.append(
                _absolute_reference(
                    query,
                    keys,
                    values,
                    query_start=row.end - query_length,
                    key_start=first,
                    window=geometry.read_window if group == 1 else None,
                )
            )
        expected = torch.stack(expected)
        assert torch.isfinite(actual).all(), "unwritten NaN tails entered attention"
        print(
            {
                "group": group,
                "query_length": query_length,
                "ends": ends,
                "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
            }
        )
        # Identical to the existing decode primitive tolerance. Full-model
        # greedy tokens and diagnostic logits keep their original gates.
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    assert prefiller.plan_calls == 2
    assert prefiller.layer_calls == 2
