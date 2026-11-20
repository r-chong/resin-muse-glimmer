"""Existing Engine token budgets change publication, not Muse's computation.

Short CPU prompts scale the GPU ablation's 8192-token budget to 16 tokens.
The runner's prefill chunk remains 8192, including its cache retention guard.
These checks prove request ordering/lifetime and tokens, not GPU performance.
"""

from __future__ import annotations

import pytest
import torch
from test_muse_glimmer_paged import _reference_tokens, _tiny_config

from resin import BatchPhase, Engine, SamplingParams


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


@pytest.fixture
def runner_factory(model, monkeypatch, tmp_path):
    from resin_models_torch.muse_glimmer import reference

    monkeypatch.delenv("RESIN_MUSE_MAX_BATCH_TOKENS", raising=False)
    monkeypatch.setattr(reference, "resolve_model_path", lambda *_: tmp_path)
    monkeypatch.setattr(reference, "load_model", lambda *_, **__: model)

    def make(**kwargs):
        runner = reference.MuseGlimmerTorchRunner(
            device="cpu",
            dtype=torch.float32,
            include_vision=False,
            prefill_chunk_size=8192,
            # The 8192-token retention guard itself requires >512 tiny pages.
            cache_parent_capacity=1024,
            kv_workspace_bytes=0,
            **kwargs,
        )
        runner.plans = []
        forward = runner.forward

        def record(plan):
            runner.plans.append(plan)
            return forward(plan)

        runner.forward = record
        return runner

    return make


def _submit(engine, prompts, steps):
    return [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(
                max_output_tokens=steps, temperature=0.0, stop_token_ids=[]
            ),
        )
        for prompt in prompts
    ]


def _drain(engine, limit=100):
    for _ in range(limit):
        if not engine.has_work():
            return
        engine.step()
    pytest.fail("Engine failed to complete within the bounded step count")


@pytest.mark.parametrize("batch", [1, 8])
def test_budget_publishes_prefills_individually_and_preserves_decode_width(
    model, runner_factory, monkeypatch, batch
):
    prompts = [[(i * 7 + row * 11) % 90 + 1 for i in range(16)] for row in range(batch)]
    expected = [_reference_tokens(model, prompt, 4) for prompt in prompts]
    observations = []
    for budget in (None, 16):
        if budget is not None:
            monkeypatch.setenv("RESIN_MUSE_MAX_BATCH_TOKENS", str(budget))
        runner = runner_factory()
        engine = Engine(runner, context_length=128, max_running_requests=batch)
        leases = _submit(engine, prompts, 4)
        first_steps = {}
        for step in range(1, 20):
            if not engine.has_work():
                break
            for event in engine.step():
                first_steps.setdefault(event.request_id, step)
        assert not engine.has_work()
        assert [lease.take_result() for lease in leases] == expected
        assert runner.live_cache_count == 0
        assert runner.runner_config.prefill_chunk_size == 8192
        decode = [plan for plan in runner.plans if plan.phase == BatchPhase.DECODE]
        assert decode and all(plan.num_sequences == batch for plan in decode)
        observations.append([first_steps[lease.id] for lease in leases])
        if budget is not None:
            prefills = runner.plans[:batch]
            assert [plan.num_sequences for plan in prefills] == [1] * batch
            assert [len(plan.token_ids) for plan in prefills] == [16] * batch
        engine.close()
    assert observations == [[1] * batch, list(range(1, batch + 1))]


def test_budget_preserves_mixed_prompt_continuations(
    model, runner_factory, monkeypatch
):
    monkeypatch.setenv("RESIN_MUSE_MAX_BATCH_TOKENS", "16")
    runner = runner_factory()
    prompts = [[(i * 7 + 3) % 90 + 1 for i in range(n)] for n in (30, 3, 17)]
    expected = [_reference_tokens(model, prompt, 12) for prompt in prompts]
    engine = Engine(runner, context_length=128, max_running_requests=3)
    leases = _submit(engine, prompts, 12)
    _drain(engine)
    assert [lease.take_result() for lease in leases] == expected
    assert all(len(plan.token_ids) <= 16 for plan in runner.plans)
    assert runner.live_cache_count == 0
    engine.close()


def test_cancelled_partial_and_ready_requests_release_pages(
    model, runner_factory, monkeypatch
):
    monkeypatch.setenv("RESIN_MUSE_MAX_BATCH_TOKENS", "16")
    runner = runner_factory()
    prompts = [[(i * 7 + 3) % 90 + 1 for i in range(n)] for n in (30, 3, 17, 16)]
    engine = Engine(runner, context_length=128, max_running_requests=4)
    partial, stopped, ongoing, ready = _submit(engine, prompts, 4)
    # The first row is only partially prefetched, so no sampled token is ready.
    assert engine.step() == []
    assert engine.cancel(partial)
    assert engine.cancel(ready)
    _drain(engine)
    assert partial.done and partial.cancelled and partial.take_result() == []
    assert ready.done and ready.cancelled and ready.take_result() == []
    assert all(ready.id not in plan.sequence_ids for plan in runner.plans)
    assert sum(partial.id in plan.sequence_ids for plan in runner.plans) == 1
    assert stopped.take_result() == _reference_tokens(model, prompts[1], 4)
    assert ongoing.take_result() == _reference_tokens(model, prompts[2], 4)
    assert runner.live_cache_count == 0
    engine.close()


def test_explicit_budget_takes_precedence_and_stop_releases_request(
    model, runner_factory, monkeypatch
):
    monkeypatch.setenv("RESIN_MUSE_MAX_BATCH_TOKENS", "32")
    runner = runner_factory(max_batch_tokens=16)
    prompt = [(i * 7 + 3) % 90 + 1 for i in range(16)]
    stop = _reference_tokens(model, prompt, 1)[0]
    engine = Engine(runner, context_length=128, max_running_requests=2)
    first = engine.submit(
        prompt,
        sampling_params=SamplingParams(
            max_output_tokens=4, temperature=0.0, stop_token_ids=[stop]
        ),
    )
    second = _submit(engine, [prompt], 4)[0]
    events = engine.step()
    assert len(events) == 1 and events[0].request_id == first.id
    assert events[0].finished and events[0].finish_reason == "stop"
    assert first.done and not second.done
    assert first.take_result() == [stop]
    assert runner.live_cache_count == 0
    _drain(engine)
    assert second.take_result() == _reference_tokens(model, prompt, 4)
    assert runner.live_cache_count == 0
    assert runner.runner_config.max_batch_tokens == 16
    engine.close()


@pytest.mark.parametrize("budget", ["0", "-1", "abc", "1.5"])
def test_invalid_environment_budget_fails_before_model_loading(monkeypatch, budget):
    from resin_models_torch.muse_glimmer import reference

    monkeypatch.setenv("RESIN_MUSE_MAX_BATCH_TOKENS", budget)
    monkeypatch.setattr(
        reference,
        "resolve_model_path",
        lambda *_: pytest.fail("invalid configuration reached checkpoint loading"),
    )
    with pytest.raises(ValueError, match="must be a positive integer"):
        reference.MuseGlimmerTorchRunner(device="cpu")
