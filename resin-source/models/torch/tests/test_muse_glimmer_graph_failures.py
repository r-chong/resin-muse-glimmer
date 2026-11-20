"""Graph-region validation cannot make a partly executed model retryable."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from resin.backend.cuda.failures import GraphInputError, PostAcceptCudaError
from resin_models_torch.muse_glimmer.execution import MuseDecodeGraphs


def _rejecting_graphs():
    def reject(*_args):
        raise GraphInputError("simulated region staging rejection")

    graphs = MuseDecodeGraphs.__new__(MuseDecodeGraphs)
    graphs.device = torch.device("cpu")
    graphs.batches = (1,)
    graphs.fallbacks = Counter()
    graphs.regions = [
        SimpleNamespace(
            inputs=(((1, 32), torch.float32),),
            outputs=(((1, 32), torch.float32),),
            operation=lambda value: value,
        )
    ]
    graphs.executor = SimpleNamespace(mode=lambda _: SimpleNamespace(run=reject))
    return graphs


def test_region_input_rejection_quarantines_the_model_forward(monkeypatch):
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    graphs = _rejecting_graphs()
    with pytest.raises(PostAcceptCudaError, match="earlier layers") as failure:
        graphs.invoke(0, torch.zeros(1, 1, 32))
    assert failure.value._resin_post_accept
    assert isinstance(failure.value.__cause__, GraphInputError)
    assert synchronized == [torch.device("cpu")]


def test_region_input_rejection_keeps_unquiescent_resources_quarantined(monkeypatch):
    def fail_sync(_device):
        raise RuntimeError("simulated synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    with pytest.raises(PostAcceptCudaError, match="remain quarantined") as failure:
        _rejecting_graphs().invoke(0, torch.zeros(1, 1, 32))
    assert failure.value._resin_post_accept
