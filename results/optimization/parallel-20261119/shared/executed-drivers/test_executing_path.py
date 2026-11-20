"""Untimed logit diagnostics captured from the runner actually under test."""
import json
from pathlib import Path

import numpy as np
import torch

from test_muse_glimmer_hf_parity import runner, references, _generate, _logit_summary


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
        log_p = torch.from_numpy(expected).log_softmax(-1)
        log_q = torch.from_numpy(actual).log_softmax(-1)
        kl = (log_p.exp() * (log_p-log_q)).sum(-1)
        summary['kl_reference_candidate'] = kl.tolist()
        summary['scope'] = 'actual runner.forward logits passed to sampler'
        summary['reference'] = ref['name']
        summary['tokens_match'] = tokens == ref['tokens']
        print('MUSE_EXECUTING_PATH_REPORT ' + json.dumps(summary), flush=True)
        Path('/experiment/diagnostics').mkdir(exist_ok=True)
        np.savez_compressed('/experiment/diagnostics/executing-' + ref['name'] + '.npz', step_logits=actual, reference_tokens=ref['tokens'], engine_tokens=tokens)
        assert summary['tokens_match']
        assert summary['argmax_matches'] == summary['rows']
