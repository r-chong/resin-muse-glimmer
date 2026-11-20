"""Untimed logit diagnostics captured from the runner actually under test."""

# Fixture re-exports are consumed by pytest.
# ruff: noqa: PLC0414

import json
import os
from pathlib import Path

import numpy as np
import torch
from test_muse_glimmer_hf_parity import (
    _generate,
    _logit_summary,
)
from test_muse_glimmer_hf_parity import (
    pytestmark as pytestmark,
)
from test_muse_glimmer_hf_parity import (
    references as references,
)
from test_muse_glimmer_hf_parity import (
    runner as runner,
)


def test_executing_path_logits(runner, references):
    original = runner._sampler
    for ref in references:
        captured = []

        def sampler(logits, plan, captured=captured):
            captured.append(logits.detach().float().cpu().numpy().copy())
            return original(logits, plan)

        runner._sampler = sampler
        try:
            (tokens,) = _generate(
                runner, [ref["prompt"]], [len(ref["tokens"])], max_running=1
            )
        finally:
            runner._sampler = original
        actual = np.concatenate(captured, axis=0)
        expected = ref["step_logits"]
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        summary = _logit_summary(actual, expected)
        log_p = torch.from_numpy(expected).double().log_softmax(-1)
        log_q = torch.from_numpy(actual).double().log_softmax(-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        summary["kl_reference_candidate"] = kl.tolist()
        summary["scope"] = "actual runner.forward logits passed to sampler"
        summary["reference"] = ref["name"]
        summary["tokens_match"] = tokens == ref["tokens"]
        print("MUSE_EXECUTING_PATH_REPORT " + json.dumps(summary), flush=True)
        dump_dir = Path(
            os.environ.get("RESIN_MUSE_DUMP_DIR", "/experiment/diagnostics")
        )
        dump_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            dump_dir / ("executing-" + ref["name"] + ".npz"),
            step_logits=actual,
            reference_tokens=ref["tokens"],
            engine_tokens=tokens,
        )
        assert summary["tokens_match"]
        assert summary["argmax_matches"] == summary["rows"]
