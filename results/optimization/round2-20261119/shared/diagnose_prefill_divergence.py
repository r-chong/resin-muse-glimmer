"""Untimed BF16 diagnostic; run HF, frozen C and prefill in separate processes.

Teacher forcing intentionally holds the decoded context identical after a
candidate's argmax changes. It does not convert a failed greedy comparison into
a pass, establish output quality, or relax any existing qualification threshold.
The model and checkpoint are read only. Only --output and its .npz are written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

PROBES = (0, 27, 28, 47)


def prompt_tokens(expected_hash):
    import numpy as np
    import torch

    generator = torch.Generator(device="cpu").manual_seed(41)
    complete = torch.randint(1000, 30000, (8451,), generator=generator).tolist()
    digest = hashlib.sha256(np.array(complete, dtype=np.int64).tobytes()).hexdigest()
    assert digest == expected_hash, (digest, expected_hash)
    return complete[:8192], digest


class Collector:
    def __init__(self, forced):
        self.forced = forced
        self.rows = []
        self.probes = {}

    def observe(self, logits):
        import torch

        step = len(self.rows)
        assert step < len(self.forced)
        assert logits.ndim == 1 and torch.isfinite(logits).all()
        raw_dtype = str(logits.dtype)
        values = logits.detach().float()
        scores, ids = values.topk(5)
        scores, ids = scores.cpu().tolist(), ids.cpu().tolist()
        argmax = int(values.argmax().item())
        tied = (values == values[argmax]).nonzero().flatten()
        row = {
            "output_step_zero_based": step,
            "context_tokens": 8192 + step,
            "raw_logit_dtype": raw_dtype,
            "top5_token_ids": ids,
            "top5_logits": scores,
            "top1_minus_top2": scores[0] - scores[1],
            "argmax_token_id": argmax,
            "exact_max_tie_count": int(tied.numel()),
            "exact_max_tie_ids_first8": tied[:8].cpu().tolist(),
            "forced_token_id": self.forced[step],
            "forced_token_logit": float(values[self.forced[step]].item()),
            "argmax_matches_forced": argmax == self.forced[step],
        }
        if step in PROBES:
            self.probes[step] = values.cpu().numpy().copy()
            row["focal_token_logits"] = {
                str(token): float(values[token].item())
                for token in (589, 2931, 5094, 4097)
            }
        self.rows.append(row)
        return self.forced[step]


def run_hf(args, prompt, collector):
    import torch
    from transformers import MuseGlimmerForConditionalGeneration

    # The pinned image has no accelerate. CPU loading then one ordinary .to()
    # avoids changing that image or requiring device_map/accelerate. Attention
    # selection remains the Transformers default; no override is supplied.
    model = (
        MuseGlimmerForConditionalGeneration.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, local_files_only=True
        )
        .to("cuda:0")
        .eval()
    )
    assert all(p.device.type == "cuda" for p in model.parameters())
    details = {
        "model_class": type(model).__name__,
        "module_file": inspect.getfile(type(model)),
        "load_method": "from_pretrained BF16 on CPU then .to(cuda:0)",
        "attention_override": None,
        "attention_implementation": {
            name: getattr(config, "_attn_implementation", None)
            for name, config in (
                ("root", model.config),
                ("text", model.config.text_config),
                ("vision", model.config.vision_config),
            )
        },
    }
    current = torch.tensor([prompt], device="cuda:0", dtype=torch.long)
    mask = torch.ones_like(current)
    cache = None
    with torch.inference_mode():
        for _ in collector.forced:
            output = model(
                input_ids=current,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            forced = collector.observe(output.logits[0, -1])
            current = torch.tensor([[forced]], device="cuda:0", dtype=torch.long)
            mask = torch.cat((mask, torch.ones_like(current)), dim=1)
    return details


def run_resin(args, prompt, collector):
    import torch
    from resin import Engine, SamplingParams
    from resin_models_torch.muse_glimmer import MuseGlimmerTorchRunner

    runner = MuseGlimmerTorchRunner(
        args.checkpoint,
        revision=None,
        dtype=torch.bfloat16,
        include_vision=True,
        prefill_chunk_size=8192,
        kv_cache="engine",
    )
    engine = Engine(
        runner,
        context_length=8451,
        max_running_requests=8,
        prefix_cache=False,
        enable_overlap=False,
    )

    def teacher_forced_sampler(logits, plan):
        assert logits.ndim == 2 and logits.shape[0] == 1
        token = collector.observe(logits[0])
        return torch.tensor([token], device=logits.device, dtype=torch.long)

    runner._sampler = teacher_forced_sampler
    lease = engine.submit(
        prompt,
        sampling_params=SamplingParams(
            max_output_tokens=len(collector.forced),
            temperature=0.0,
            top_p=1.0,
            seed=41,
            stop_token_ids=[],
        ),
    )
    while engine.has_work():
        engine.step(allow_overlap=False)
    assert list(lease.take_result()) == collector.forced
    assert runner.kv_cache == "engine" and runner.live_cache_count == 0
    arena = runner._arena
    assert arena is not None and arena.native_decode
    native_prefill = getattr(arena, "native_prefill", False)
    assert native_prefill == (args.mode == "prefill")
    counters = {}
    for name in ("_native_decoder", "_native_prefiller"):
        attention = getattr(arena, name, None)
        counters[name] = {
            key: getattr(attention, key, 0) for key in ("plan_calls", "layer_calls")
        }
    assert counters["_native_decoder"]["layer_calls"] > 0
    if args.mode == "prefill":
        assert counters["_native_prefiller"]["layer_calls"] == 52
    source = Path(inspect.getfile(type(runner)))
    return {
        "model_class": type(runner).__name__,
        "module_file": str(source),
        "runner_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "kv_cache": runner.kv_cache,
        "native_decode": arena.native_decode,
        "native_prefill": native_prefill,
        "actual_attention_counters": counters,
        "engine_context_capacity": 8451,
        "engine_max_running_requests": 8,
        "live_batch": 1,
        "prefill_chunk_tokens": 8192,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hf", "C", "prefill"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--source-commit", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    assert torch.cuda.is_available(), "This diagnostic requires root's bounded H100 job"
    continuation = json.loads(args.continuation.read_text())
    forced = continuation["forced_ids"]
    assert len(forced) == 48 and all(type(token) is int for token in forced)
    prompt, digest = prompt_tokens(continuation["prompt_sha256"])
    collector = Collector(forced)
    started = time.perf_counter()
    details = (run_hf if args.mode == "hf" else run_resin)(args, prompt, collector)
    torch.cuda.synchronize()
    assert len(collector.rows) == 48 and tuple(collector.probes) == PROBES
    versions = {}
    for package in (
        "torch",
        "transformers",
        "flashinfer-python",
        "numpy",
        "accelerate",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "mode": args.mode,
        "source_commit_declared": args.source_commit,
        "checkpoint": args.checkpoint,
        "prompt_sha256_full_8451_tokens": digest,
        "actual_prompt_tokens": 8192,
        "continuation_source": continuation["source"],
        "forced_ids": forced,
        "probe_steps_zero_based": PROBES,
        "dtype": "bfloat16 weights and KV; capture logits in float32",
        "scope": "Untimed identical teacher-forced context; no greedy/quality qualification conclusion",
        "sampling_effect": "Forced C b1 continuation only; untouched model logits captured before sampler return",
        "details": details,
        "packages": versions,
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "matmul_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sdp_math_fp16_bf16_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "elapsed_s_including_load": time.perf_counter() - started,
        "rows": collector.rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        probe_steps=np.array(PROBES, dtype=np.int64),
        logits=np.stack([collector.probes[step] for step in PROBES]),
    )
    print("PREFILL_DIVERGENCE_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
