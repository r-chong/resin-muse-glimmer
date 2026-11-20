"""Characterize logit differences against the HF (sdpa) reference.

Compares any number of candidates (Resin dump, HF eager control) with the
reference, on the same rows: greedy decode steps (teacher-forced on the
reference tokens) and prompt probe positions. Reports BF16-ULP-normalized
differences, top-1 / top-5 agreement, margin sensitivity, and KL divergence.

    python harness/logit_analysis.py --reference REF.npz \
        --candidate resin=DUMP.npz --candidate hf_eager=EAGER.npz [--json OUT]

Rows are compared only while the candidate's own greedy tokens agree with the
reference (HF eager runs free-running greedy, not teacher-forced).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def bf16_ulp(values: np.ndarray) -> np.ndarray:
    """Spacing of BF16 (8 significant bits) at each value's magnitude."""
    magnitude = np.maximum(np.abs(values), np.float32(2.0**-126))
    return np.exp2(np.floor(np.log2(magnitude)) - 7)


def is_bf16(values: np.ndarray) -> bool:
    bits = values.astype(np.float32).view(np.uint32)
    return bool(np.all((bits & 0xFFFF) == 0))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def compare(ref: np.ndarray, cand: np.ndarray) -> dict:
    diff = np.abs(cand - ref)
    ulps = diff / bf16_ulp(np.maximum(np.abs(ref), np.abs(cand)))
    order = np.argsort(-ref, axis=-1, kind="stable")  # ties -> lowest id, like argmax
    top1, top2 = order[:, 0], order[:, 1]
    rows = np.arange(len(ref))
    margin = ref[rows, top1] - ref[rows, top2]
    cand_margin = cand[rows, top1] - cand[rows, top2]
    top5_ref = np.sort(order[:, :5], axis=-1)
    top5_cand = np.sort(np.argsort(-cand, axis=-1, kind="stable")[:, :5], axis=-1)
    lp_ref, lp_cand = log_softmax(ref), log_softmax(cand)
    kl = (np.exp(lp_ref) * (lp_ref - lp_cand)).sum(-1)
    # Largest differences by logit magnitude band (softcap saturates near ±20).
    bands = {}
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 20.01)):
        mask = (np.abs(ref) >= lo) & (np.abs(ref) < hi)
        if mask.any():
            bands[f"|x|∈[{lo},{int(hi)})"] = {
                "count": int(mask.sum()),
                "max_abs_diff": round(float(diff[mask].max()), 4),
                "p99_ulps": round(float(np.percentile(ulps[mask], 99)), 2),
            }
    top_rows = ref[rows, top1]
    return {
        "rows": int(len(ref)),
        "logits_bitwise_equal_fraction": round(float((diff == 0).mean()), 4),
        "max_abs_diff": round(float(diff.max()), 4),
        "mean_abs_diff": round(float(diff.mean()), 5),
        "ulps": {q: round(float(np.percentile(ulps, p)), 2)
                 for q, p in (("p50", 50), ("p90", 90), ("p99", 99), ("p99.99", 99.99))}
                | {"max": round(float(ulps.max()), 2)},
        "top1_logit_abs_diff_max": round(float(np.abs(cand[rows, top1] - top_rows).max()), 4),
        "top1_agreement": int((cand.argmax(-1) == top1).sum()),
        "top5_set_agreement": int((top5_ref == top5_cand).all(-1).sum()),
        "margin_shift_max": round(float(np.abs(cand_margin - margin).max()), 4),
        "min_reference_margin": round(float(margin.min()), 4),
        "low_margin_rows": [
            {"row": int(i), "ref_margin": round(float(margin[i]), 4),
             "cand_margin": round(float(cand_margin[i]), 4),
             "same_top1": bool(cand[i].argmax() == top1[i])}
            for i in np.argsort(margin)[:5]
        ],
        "kl_ref_to_cand": {"mean": float(kl.mean()), "max": float(kl.max())},
        "by_magnitude": bands,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=path.npz")
    parser.add_argument("--json")
    args = parser.parse_args()
    ref = np.load(args.reference)
    ref_tokens = ref["generated_ids"].astype(np.int64)
    out = {"reference": args.reference, "reference_is_bf16": is_bf16(ref["step_logits"])}
    for item in args.candidate:
        name, path = item.split("=", 1)
        cand = np.load(path)
        tokens = cand["engine_tokens"] if "engine_tokens" in cand else cand["generated_ids"]
        tokens = tokens.astype(np.int64)
        agree = 0
        while agree < min(len(tokens), len(ref_tokens)) and tokens[agree] == ref_tokens[agree]:
            agree += 1
        steps = min(len(cand["step_logits"]), len(ref["step_logits"]))
        if "engine_tokens" not in cand:  # free-running: rows valid until divergence
            steps = min(steps, agree + 1)
        result = {
            "candidate_is_bf16": is_bf16(cand["step_logits"]),
            "greedy_tokens_agree_prefix": agree,
            "greedy_tokens_compared": int(min(len(tokens), len(ref_tokens))),
            "decode_steps": compare(ref["step_logits"][:steps], cand["step_logits"][:steps]),
        }
        if "probe_logits" in cand and "probe_logits" in ref:
            result["prompt_probes"] = compare(ref["probe_logits"], cand["probe_logits"])
        out[name] = result
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
