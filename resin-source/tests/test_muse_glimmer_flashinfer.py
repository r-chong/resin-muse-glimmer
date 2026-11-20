"""Direct-paged decode checks, independent of the model's eager attention.

The GPU primitive case covers real GQA geometry, noncontiguous layer views,
unequal histories, page/window boundaries, evicted window starts and NaN tails.
Full-checkpoint HF parity remains a separate, unchanged qualification gate.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from resin_models_torch.muse_glimmer.flashinfer import (
    FlashinferPagedDecode,
    decode_page_lengths,
)
from resin_models_torch.muse_glimmer.paged import (
    BLOCK_TOKENS,
    GroupRow,
    MuseCacheGeometry,
    MuseKvArena,
    PagedRow,
)


def _group(pages, first_position=0, *, device="cpu"):
    empty = torch.empty(0, dtype=torch.long, device=device)
    return GroupRow(
        torch.tensor(pages, dtype=torch.long, device=device),
        first_position,
        empty,
        empty,
    )


def test_live_page_lengths_ignore_reserved_tail_pages_and_absolute_start():
    rows = [
        PagedRow((_group([3, 4, 5]), _group([7, 8], 2048)), end=2113),
    ]
    # The history fixture intentionally has insufficient coverage.
    with pytest.raises(ValueError, match="cover"):
        decode_page_lengths(rows, 0)
    assert decode_page_lengths(rows, 1) == ([2], [1])
    for length in (1, 63, 64, 65):
        row = PagedRow((_group([3, 4, 5]), _group([7, 8, 9])), end=length)
        counts, last = decode_page_lengths([row], 0)
        assert counts == [math.ceil(length / BLOCK_TOKENS)]
        assert last == [(length - 1) % BLOCK_TOKENS + 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel qualification")
def test_native_paged_decode_matches_independent_dense_attention():
    pytest.importorskip("flashinfer")
    torch.manual_seed(123)
    device = "cuda"
    geometry = MuseCacheGeometry(
        full_layers=(0,),
        sliding_layers=(1, 2, 3),
        kv_heads=2,
        head_dim=128,
        read_window=2048,
        window_tokens=10239,
    )
    arena = MuseKvArena(geometry, 320, device=device, dtype=torch.bfloat16)
    arena.storage.fill_(float("nan"))
    lengths = (1, 63, 64, 65, 2047, 2048, 2049, 8193)
    rows, data = [], {0: [], 1: []}
    next_page = [3, 150]  # disjoint physical slabs for full/window groups
    for end in lengths:
        first_window = (
            max(0, (end - geometry.read_window) // BLOCK_TOKENS) * BLOCK_TOKENS
        )
        groups = []
        for group, first in enumerate((0, first_window)):
            live = end - first
            count = math.ceil(live / BLOCK_TOKENS)
            # One future reserved page must never enter the native plan.
            pages = list(range(next_page[group], next_page[group] + count + 1))
            next_page[group] += count + 1
            groups.append(_group(pages, first, device=device))
            keys = torch.randn(live, 2, 128, device=device, dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            _, key_pages, value_pages = arena.layer_views(group)
            slots = torch.arange(live, device=device)
            page_ids = groups[-1].pages[slots // BLOCK_TOKENS]
            key_pages[page_ids, slots % BLOCK_TOKENS] = keys
            value_pages[page_ids, slots % BLOCK_TOKENS] = values
            data[group].append((keys, values))
        rows.append(PagedRow(tuple(groups), end))

    decoder = FlashinferPagedDecode(geometry, 32, device=device, dtype=torch.bfloat16)
    decoder.plan(rows)
    queries = torch.randn(len(rows), 32, 128, device=device, dtype=torch.bfloat16)
    for group in (0, 1):
        _, key_pages, value_pages = arena.layer_views(group)
        assert not key_pages.is_contiguous()
        actual = decoder.run(group, queries, key_pages, value_pages)
        expected = []
        for query, (keys, values) in zip(queries, data[group]):
            if group == 1:
                keys = keys[-geometry.read_window :]
                values = values[-geometry.read_window :]
            expected.append(
                F.scaled_dot_product_attention(
                    query[None, :, None, :].float(),
                    keys.transpose(0, 1)[None].repeat_interleave(16, dim=1).float(),
                    values.transpose(0, 1)[None].repeat_interleave(16, dim=1).float(),
                    scale=128**-0.5,
                )[0, :, 0].to(torch.bfloat16)
            )
        expected = torch.stack(expected)
        assert torch.isfinite(actual).all(), "unwritten NaN tails entered attention"
        print(
            {
                "group": group,
                "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
            }
        )
        # A primitive BF16 attention tolerance, not a replacement for the exact
        # greedy-token gate or the checkpoint's diagnostic logit comparison.
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    assert decoder.plan_calls == 2
    assert decoder.layer_calls == 2
