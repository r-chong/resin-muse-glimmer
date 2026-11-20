"""Changed-input and Engine-page qualification for stateless decode graphs."""

from __future__ import annotations

import copy

import pytest
import torch
from test_muse_glimmer_paged import _generate, _runner, _tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _models():
    from resin_models_torch.muse_glimmer.model import (
        MuseGlimmerForConditionalGeneration,
    )

    torch.manual_seed(41)
    eager = MuseGlimmerForConditionalGeneration(_tiny_config(), include_vision=False)
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(0.0, 0.5)
    eager = eager.eval().to(device="cuda", dtype=torch.bfloat16)
    eager.materialize_buffers("cuda")
    return eager, copy.deepcopy(eager)


def test_decode_region_graphs_use_changed_inputs_and_positions():
    from resin_models_torch.muse_glimmer.execution import MuseDecodeGraphs

    eager, graphed = _models()
    graphs = MuseDecodeGraphs(graphed)
    graphs.capture()
    with torch.inference_mode():
        # Change both activations and absolute RoPE positions, alternate graph
        # batches, and force eager fallbacks without padding the GEMM geometry.
        for batch, length, position in [
            (1, 1, 3),
            (8, 1, 17),
            (2, 1, 65),
            (4, 1, 7),
            (1, 1, 2050),
            (3, 1, 11),
            (1, 5, 17),
        ]:
            hidden = torch.randn(batch, length, 32, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(position, position + length, device="cuda")[
                None
            ].expand(batch, -1)
            for expected, actual in zip(
                eager.model.language_model.layers,
                graphed.model.language_model.layers,
                strict=True,
            ):
                want = expected.self_attn._project(hidden, positions)
                got = tuple(
                    value.clone()
                    for value in actual.self_attn._project(hidden, positions)
                )
                assert all(torch.equal(a, b) for a, b in zip(want, got, strict=True))
                want_ff = expected._feed_forward(hidden)
                got_ff = actual._feed_forward(hidden).clone()
                assert torch.equal(want_ff, got_ff)
    assert graphs.counts["decode_regions"]["replay_count"] > 0
    assert graphs.counts["decode_regions"]["eager_fallback_count"] > 0
    graphs.close()


def _gpu_paged_runner(model, chunk):
    from resin import RunnerConfig

    runner = _runner(model, "local", chunk)
    runner.device = torch.device("cuda")
    runner.kv_cache = "engine"
    runner._configure_engine_cache(chunk, 0.9, 0, 64)
    runner.runner_config = RunnerConfig(
        continuation_prefill=True,
        prefill_chunk_size=chunk,
        cache_layout=runner.cache_layout,
    )
    return runner


def test_decode_region_graphs_preserve_engine_paged_generation():
    from resin_models_torch.muse_glimmer.execution import MuseDecodeGraphs

    eager, graphed = _models()
    graphs = MuseDecodeGraphs(graphed)
    graphs.capture()
    prompts = [
        [(7 * i + 3) % 90 + 1 for i in range(67)],
        [(5 * i + 1) % 90 + 1 for i in range(3)],
        [(3 * i + 2) % 90 + 1 for i in range(17)],
        [(3 * i + 9) % 90 + 1 for i in range(31)],
    ]
    # Includes exact-batch graph execution, a three-row eager fallback, a
    # 64-token page boundary, unequal histories and a request/window crossover.
    for selected, chunk in [(prompts, 5), (prompts[:3], 64), (prompts[:1], 5)]:
        expected = _generate(_gpu_paged_runner(eager, chunk), selected, 12)
        actual = _generate(_gpu_paged_runner(graphed, chunk), selected, 12)
        assert actual == expected
    assert graphs.counts["decode_regions"]["replay_count"] > 0
    graphs.close()
