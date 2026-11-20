"""Text-only Muse Glimmer parity against saved Transformers greedy references.

GPU-only. Set RESIN_TEST_CHECKPOINT to the pinned checkpoint directory and
RESIN_MUSE_REFERENCES to comma-separated ``.npz`` files written by
``baseline/hf_reference.py`` or ``baseline/hf_reference_cases.py`` (input_ids,
step_logits, generated_ids; optionally probe_positions/probe_logits).

Each test prints one ``MUSE_PARITY_REPORT`` JSON line before asserting so a
failing run still records its evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = os.environ.get("RESIN_TEST_CHECKPOINT")
REFERENCES = [p for p in os.environ.get("RESIN_MUSE_REFERENCES", "").split(",") if p]
STEPS = int(os.environ.get("RESIN_MUSE_STEPS", "16"))
# Optional directory for full teacher-forced logit rows (diagnostics).
DUMP_DIR = os.environ.get("RESIN_MUSE_DUMP_DIR")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT and REFERENCES), reason="needs a GPU checkpoint and HF references"
)


def _load(path: str) -> dict:
    data = np.load(path)
    ref = {key: data[key] for key in data.files}
    ref["name"] = Path(path).stem
    ref["prompt"] = ref["input_ids"].astype(np.int64).tolist()
    ref["tokens"] = ref["generated_ids"].astype(np.int64).tolist()[:STEPS]
    ref["step_logits"] = ref["step_logits"][: len(ref["tokens"])].astype(np.float32)
    return ref


def _report(test: str, **fields) -> dict:
    print("MUSE_PARITY_REPORT " + json.dumps({"test": test, **fields}), flush=True)
    return fields


def _token_summary(actual: list[int], expected: list[int]) -> dict:
    first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if first is None and len(actual) != len(expected):
        first = min(len(actual), len(expected))
    return {
        "matches": sum(a == b for a, b in zip(actual, expected)),
        "first_mismatch": first,
    }


def _logit_summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(actual - expected)
    ranked = np.sort(expected, axis=-1)
    return {
        "argmax_matches": int((actual.argmax(-1) == expected.argmax(-1)).sum()),
        "rows": int(len(expected)),
        "max_abs_diff": [round(float(v), 4) for v in diff.max(-1)],
        "mean_abs_diff": [round(float(v), 5) for v in diff.mean(-1)],
        "reference_top1_margin": [
            round(float(v), 4) for v in ranked[:, -1] - ranked[:, -2]
        ],
    }


@pytest.fixture(scope="module")
def runner():
    import torch

    from resin_models_torch.muse_glimmer import MuseGlimmerTorchRunner

    assert torch.cuda.is_available()
    return MuseGlimmerTorchRunner(
        Path(CHECKPOINT), revision=None, device="cuda", include_vision=False
    )


@pytest.fixture(scope="module")
def references():
    return [_load(path) for path in REFERENCES]


def _generate(runner, prompts, steps, *, chunk=None, max_running=8):
    """Greedy generation through the Engine; returns per-request tokens."""
    from resin import Engine, RunnerConfig, SamplingParams

    previous = runner.runner_config
    if chunk is not None:
        runner.runner_config = RunnerConfig(
            continuation_prefill=True,
            prefill_chunk_size=chunk,
            cache_layout=previous.cache_layout,
        )
    try:
        engine = Engine(
            runner,
            context_length=max(len(p) for p in prompts) + max(steps) + 8,
            max_running_requests=max_running,
        )
        leases = [
            engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    max_output_tokens=n, temperature=0.0, stop_token_ids=[]
                ),
            )
            for prompt, n in zip(prompts, steps)
        ]
        limit = sum(len(p) for p in prompts) + sum(steps) + 64
        for _ in range(limit):
            if not engine.has_work():
                break
            engine.step()
        assert all(lease.done for lease in leases), "engine did not finish"
        return [list(lease.take_result()) for lease in leases]
    finally:
        runner.runner_config = previous


def _teacher_forced(runner, ref) -> tuple[np.ndarray, np.ndarray | None]:
    """Step logits after the prompt, plus logits at reference probe positions."""
    import torch

    model = runner._model
    state = model.make_state()
    prompt = ref["prompt"]
    with torch.inference_mode():
        hidden = model(
            torch.tensor([prompt], device="cuda"),
            state,
            torch.arange(len(prompt), device="cuda")[None],
        )
        probes = None
        if "probe_positions" in ref:
            index = torch.as_tensor(ref["probe_positions"], device="cuda")
            probes = (
                model.logits(hidden[0].index_select(0, index)).float().cpu().numpy()
            )
        rows = [model.logits(hidden[:, -1]).float()[0]]
        for token in ref["tokens"][:-1]:
            hidden = model(
                torch.tensor([[token]], device="cuda"),
                state,
                torch.tensor([[state.length]], device="cuda"),
            )
            rows.append(model.logits(hidden[:, -1]).float()[0])
    return torch.stack(rows).cpu().numpy(), probes


def test_single_request_tokens_and_logits(runner, references):
    """One prefill plus greedy decode per reference, including window-crossing prompts."""
    failures = []
    for ref in references:
        (tokens,) = _generate(
            runner, [ref["prompt"]], [len(ref["tokens"])], max_running=1
        )
        steps, probes = _teacher_forced(runner, ref)
        if DUMP_DIR:
            Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(DUMP_DIR) / f"{ref['name']}.npz",
                step_logits=steps,
                reference_tokens=np.array(ref["tokens"]),
                engine_tokens=np.array(tokens),
                **(
                    {"probe_logits": probes, "probe_positions": ref["probe_positions"]}
                    if probes is not None
                    else {}
                ),
            )
        summary = _report(
            "single_request",
            reference=ref["name"],
            prompt_tokens=len(ref["prompt"]),
            engine_tokens=tokens,
            reference_tokens=ref["tokens"],
            engine=_token_summary(tokens, ref["tokens"]),
            steps=_logit_summary(steps, ref["step_logits"]),
            probe_positions=ref["probe_positions"].tolist()
            if probes is not None
            else [],
            probes=_logit_summary(probes, ref["probe_logits"])
            if probes is not None
            else None,
        )
        if summary["engine"]["first_mismatch"] is not None:
            failures.append(f"{ref['name']}: engine tokens diverge")
        if summary["steps"]["argmax_matches"] != summary["steps"]["rows"]:
            failures.append(f"{ref['name']}: teacher-forced argmax mismatch")
        if (
            probes is not None
            and summary["probes"]["argmax_matches"] != summary["probes"]["rows"]
        ):
            failures.append(f"{ref['name']}: probe argmax mismatch")
    assert not failures, failures
    assert runner.live_cache_count == 0


def test_chunked_prefill_across_window_matches_reference(runner, references):
    """Engine prefill in chunks that straddle the 2,048-token sliding window."""
    long_refs = [ref for ref in references if len(ref["prompt"]) > 2048]
    if not long_refs:
        pytest.skip("no reference prompt longer than the sliding window")
    failures = []
    for chunk in (1000, 2047):
        for ref in long_refs:
            (tokens,) = _generate(
                runner,
                [ref["prompt"]],
                [len(ref["tokens"])],
                chunk=chunk,
                max_running=1,
            )
            summary = _report(
                "chunked_prefill",
                reference=ref["name"],
                chunk=chunk,
                prompt_tokens=len(ref["prompt"]),
                engine=_token_summary(tokens, ref["tokens"]),
            )
            if summary["engine"]["first_mismatch"] is not None:
                failures.append(f"{ref['name']} chunk={chunk}")
    assert not failures, failures


@pytest.mark.parametrize("concurrency", [2, 4, 8])
def test_concurrent_requests_batch_and_match_reference(runner, references, concurrency):
    """Concurrent greedy requests (mixed prompt lengths) through one Engine."""
    import numpy as np

    from resin.backend.plans import BatchView

    original = runner.forward
    batch_sizes: list[int] = []

    def counted(plan):
        # Record Engine decode batching independently of the KV path.
        batch = BatchView.from_plan(plan)
        if batch.num_sequences and np.all(batch.query_lengths == 1):
            batch_sizes.append(batch.num_sequences)
        return original(plan)

    runner.forward = counted
    try:
        chosen = [references[i % len(references)] for i in range(concurrency)]
        outputs = _generate(
            runner,
            [ref["prompt"] for ref in chosen],
            [len(ref["tokens"]) for ref in chosen],
            max_running=concurrency,
        )
    finally:
        del runner.forward
    results = [
        {"reference": ref["name"], **_token_summary(out, ref["tokens"])}
        for ref, out in zip(chosen, outputs)
    ]
    _report(
        "concurrent",
        concurrency=concurrency,
        batched_decode_calls=len(batch_sizes),
        max_decode_batch=max(batch_sizes, default=0),
        requests=results,
    )
    assert runner.live_cache_count == 0
    assert max(batch_sizes, default=0) == concurrency, (
        "batched decode path not exercised"
    )
    assert all(r["first_mismatch"] is None for r in results), results

"""Untimed logit diagnostics captured from the runner actually under test."""
import json
from pathlib import Path

import numpy as np
import torch




def test_executing_path_logits(runner, references):
    original = runner._sampler
    for ref in references:
        captured = []

        def sampler(logits, plan):
            captured.append(logits.detach().float().cpu().numpy().copy())
            return original(logits, plan)

        runner._sampler = sampler
        try:
            (tokens,) = _generate(runner, [ref['prompt']], [len(ref['tokens'])], max_running=1)
        finally:
            runner._sampler = original
        actual = np.concatenate(captured, axis=0)
        expected = ref['step_logits']
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        summary = _logit_summary(actual, expected)
        log_p = torch.from_numpy(expected).double().log_softmax(-1)
        log_q = torch.from_numpy(actual).double().log_softmax(-1)
        kl = (log_p.exp() * (log_p-log_q)).sum(-1)
        summary['kl_reference_candidate'] = kl.tolist()
        summary['scope'] = 'actual runner.forward logits passed to sampler'
        summary['reference'] = ref['name']
        summary['tokens_match'] = tokens == ref['tokens']
        print('MUSE_EXECUTING_PATH_REPORT ' + json.dumps(summary), flush=True)
        Path(os.environ['RESIN_MUSE_DUMP_DIR']).mkdir(exist_ok=True)
        np.savez_compressed(str(Path(os.environ['RESIN_MUSE_DUMP_DIR']) / ('executing-' + ref['name'] + '.npz')), step_logits=actual, reference_tokens=ref['tokens'], engine_tokens=tokens)
        assert summary['tokens_match']
        assert summary['argmax_matches'] == summary['rows']
