"""CPU checks for Muse Glimmer's Engine-managed (paged) KV path.

A tiny random model with Muse's layer pattern (3 sliding : 1 full, NoPE full
layers) runs through the real Resin Engine twice: once with request-owned
tensors (``kv_cache="local"``) and once with a declared ``CacheLayout`` and
Engine pages (``kv_cache="engine"``). Both must produce the greedy tokens of a
full-context reference forward, across window boundaries and chunked prefill.
"""

from __future__ import annotations

import pytest
import torch

SLIDING_WINDOW = 8


def _tiny_config():
    from resin_models_torch.muse_glimmer.config import (
        MuseGlimmerConfig,
        MuseGlimmerTextConfig,
        MuseGlimmerVisionConfig,
    )

    layers = ("sliding_attention",) * 3 + ("full_attention",)
    text = MuseGlimmerTextConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        sliding_window=SLIDING_WINDOW,
        layer_types=layers * 2,
        layer_rope_theta=(10_000.0, 10_000.0, 10_000.0, 0.0) * 2,
        final_logit_softcapping=20.0,
        output_multiplier=0.5,
    )
    vision = MuseGlimmerVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        layer_norm_eps=1e-5,
        rope_theta=10_000.0,
        layer_types=("full_attention",),
    )
    return MuseGlimmerConfig(
        text=text,
        vision=vision,
        image_token_id=95,
        video_token_id=96,
        out_hidden_size=32,
        projector_hidden_size=16,
    )


@pytest.fixture(scope="module")
def model():
    from resin_models_torch.muse_glimmer.model import (
        MuseGlimmerForConditionalGeneration,
    )

    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(_tiny_config(), include_vision=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.5)
    model.materialize_buffers("cpu")
    return model.eval()


def _runner(model, kv_cache: str, chunk: int):
    from resin import RunnerConfig
    from resin.backend.sampling import TorchSampler
    from resin_models_torch.muse_glimmer import MuseGlimmerTorchRunner

    runner = MuseGlimmerTorchRunner.__new__(MuseGlimmerTorchRunner)
    runner.device = torch.device("cpu")
    runner._model = model
    runner.config = model.config
    runner.include_vision = False
    runner.vocab_size = model.config.text.vocab_size
    runner.kv_cache = kv_cache
    runner.cache_layout = runner.cache_capacity_plan = None
    runner._arena = runner._binding = None
    if kv_cache == "engine":
        runner._configure_engine_cache(chunk, 0.9, 0, 64)
    runner.runner_config = RunnerConfig(
        continuation_prefill=True,
        prefill_chunk_size=chunk,
        cache_layout=runner.cache_layout,
    )
    runner._sampler = TorchSampler()
    runner._requests = {}
    runner.forward_calls = runner.vision_calls = 0
    return runner


def _reference_tokens(model, prompt, steps):
    """Greedy tokens from full-context forwards (no cache reuse)."""
    tokens = list(prompt)
    with torch.inference_mode():
        for _ in range(steps):
            state = model.make_state()
            hidden = model(
                torch.tensor([tokens]), state, torch.arange(len(tokens))[None]
            )
            tokens.append(int(model.logits(hidden[:, -1]).argmax()))
    return tokens[len(prompt) :]


def _generate(runner, prompts, steps):
    from resin import Engine, SamplingParams

    engine = Engine(runner, context_length=128, max_running_requests=len(prompts))
    leases = [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(
                max_output_tokens=steps, temperature=0.0, stop_token_ids=[]
            ),
        )
        for prompt in prompts
    ]
    for _ in range(500):
        if not engine.has_work():
            break
        engine.step()
    return [list(lease.take_result()) for lease in leases]


def test_engine_layout_matches_muse_geometry(model):
    from resin import CacheRetention

    runner = _runner(model, "engine", chunk=5)
    history, window = runner.cache_layout.groups
    assert history.blocks_per_slab == 3  # 6 sliding layers / 2 full layers
    assert window.retention == CacheRetention.WINDOW
    assert window.read_window_tokens == SLIDING_WINDOW
    assert window.window_tokens == SLIDING_WINDOW + 5 - 1


@pytest.mark.parametrize("chunk", [5, 64])
def test_engine_pages_match_local_state_and_reference(model, chunk):
    prompts = [
        [(7 * i + 3) % 90 + 1 for i in range(30)],  # crosses the 8-token window
        [(5 * i + 1) % 90 + 1 for i in range(3)],  # shorter than one block
        [(3 * i + 2) % 90 + 1 for i in range(17)],
    ]
    steps = 12
    expected = [_reference_tokens(model, prompt, steps) for prompt in prompts]
    local = _generate(_runner(model, "local", chunk), prompts, steps)
    engine_runner = _runner(model, "engine", chunk)
    paged = _generate(engine_runner, prompts, steps)
    assert local == expected
    assert paged == expected
    assert engine_runner.live_cache_count == 0
