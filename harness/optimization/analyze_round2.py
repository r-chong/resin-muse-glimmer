"""Validate and summarize follow-up runs without mixing timing boundaries."""

import argparse
import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def describe(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
        "runs": len(values),
    }


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    i = int(x)
    j = min(i + 1, len(values) - 1)
    return values[i] + (values[j] - values[i]) * (x - i)


def summarize(completion):
    grouped = defaultdict(list)
    for record in completion["records"]:
        grouped[record["phase"], record["context_length"], record["batch_size"]].append(
            record
        )
    rows = []
    for (phase, context, batch), records in sorted(grouped.items()):
        row = {
            "phase": phase,
            "context_length": context,
            "batch_size": batch,
            "end_to_end_output_tokens_s_gpu": describe(
                [r["completion_output_tokens_s_gpu"] for r in records]
            ),
            "mean_ttft_ms": describe(
                [1000 * statistics.mean(r["ttft_s"]) for r in records]
            ),
            "max_ttft_ms": describe([1000 * max(r["ttft_s"]) for r in records]),
        }
        if phase == "decode":
            row.update(
                mean_per_request_tpot_ms=describe(
                    [1000 * statistics.mean(r["per_request_tpot_s"]) for r in records]
                ),
                max_per_request_tpot_ms=describe(
                    [1000 * max(r["per_request_tpot_s"]) for r in records]
                ),
                p90_inter_token_ms=describe(
                    [
                        1000
                        * percentile(
                            [
                                x
                                for intervals in r["completion_intervals_s"]
                                for x in intervals
                            ],
                            0.9,
                        )
                        for r in records
                    ]
                ),
                max_first_to_second_token_ms=describe(
                    [1000 * max(a[1] - a[0] for a in r["arrivals_s"]) for r in records]
                ),
                mean_first_to_second_token_ms=describe(
                    [
                        1000 * statistics.mean(a[1] - a[0] for a in r["arrivals_s"])
                        for r in records
                    ]
                ),
                post_warmup_output_tokens_s_gpu=describe(
                    [r["post_warmup_output_tokens_s_gpu"] for r in records]
                ),
            )
        rows.append(row)
    return {
        "rows": rows,
        "prompt_sha256": completion["prompt_sha256"],
        "reference_checks": completion["references"],
        "startup_s": completion["startup_s"],
    }


def validate(completion):
    records = completion["records"]
    assert len(records) == 18
    assert all(
        r["first_mismatch"] is None and r["matches"] == 16
        for r in completion["references"]
    )
    assert len(completion["references"]) == 3
    for record in records:
        batch = record["batch_size"]
        steps = 256 if record["phase"] == "decode" else 1
        assert record["output_tokens"] == batch * steps
        assert len(record["tokens"]) == len(record["arrivals_s"]) == batch
        assert all(len(x) == steps for x in record["tokens"])
        assert all(len(x) == steps for x in record["arrivals_s"])
        assert all(all(y >= x for x, y in pairwise(a)) for a in record["arrivals_s"])
        assert all(a[-1] <= record["elapsed_s"] for a in record["arrivals_s"])
        assert (
            abs(
                record["output_tokens"] / record["elapsed_s"]
                - record["completion_output_tokens_s_gpu"]
            )
            < 1e-8
        )
    keys = {
        (r["phase"], r["context_length"], r["batch_size"], r["repeat"]) for r in records
    }
    assert len(keys) == 18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    report = {
        "status": "in_progress",
        "baseline_commit": "66bb70a63470581f1ced548265e839da374836aa",
        "timing": "Submit all requests through CPU-visible final token, includes prefill, no HTTP or detokenization",
        "pairs": {},
        "validation": {"case_records": 0, "all_available_records_validated": True},
        "limitations": [
            "Each candidate/control case shares one physical H100 and software stack. Delivery batch eight uses a separate recovery allocation after a timeout; both sides are replaced together. The control block precedes the candidate block; three repeated cohorts per case, not randomized alternating blocks.",
            "Different pairs use different H100 allocations. Host dispatch and allocation variability may affect absolute decode times; compare each candidate to its own paired control.",
            "The stock vLLM reference is from the prior round, with its own pinned stack and separate physical H100, using identical model, tokens, precision, output budget and completion boundary.",
            "Greedy parity on saved references and kernel/scheduler tests does not establish dataset-level output quality.",
            "Full-width token-count checks do not by themselves prove generated-token identity; cross-variant token agreement is reported separately.",
        ],
    }
    for lane in ["graphs", "prefill", "delivery"]:
        path = root / lane / "pair-001/result.json"
        if lane == "delivery" and (root / "delivery/combined/result.json").exists():
            path = root / "delivery/combined/result.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        legs = {leg["name"]: leg for leg in raw.get("completions", [])}
        pair = {
            "source": str(path.relative_to(root)),
            "hardware": raw["hardware"],
            "gpu_function_body_s": raw["gpu_function_elapsed_s"],
            "qualification": {
                n: {k: v[k] for k in ["returncode", "elapsed_s"]}
                for n, v in raw["qualification"].items()
            },
            "variants": {},
        }
        if "hardware_by_case" in raw:
            pair["hardware_by_case"] = raw["hardware_by_case"]
            pair["synthesis_note"] = raw["synthesis_note"]
        if lane == "prefill":
            qualification_path = root / "prefill/qual-001/result.json"
            qualification = json.loads(qualification_path.read_text())
            qualified = next(v for v in qualification["variants"] if v["name"] == lane)
            measured = next(v for v in raw["variants"] if v["name"] == lane)
            assert qualified["source_sha256"] == measured["source_sha256"]
            assert qualified["commit"] == measured["commit"]
            pair["qualification_source"] = str(qualification_path.relative_to(root))
            pair["qualification"] = {
                n: {k: v[k] for k in ["returncode", "elapsed_s"]}
                for n, v in qualification["qualification"].items()
            }
        for name, leg in legs.items():
            assert leg["returncode"] == 0, (lane, name, "benchmark failed")
            completion = leg["completion"]
            validate(completion)
            report["validation"]["case_records"] += len(completion["records"])
            pair["variants"][name] = summarize(completion)
        if set(pair["variants"]) != {"C", lane}:
            pair["complete"] = False
            report["pairs"][lane] = pair
            continue
        assert pair["qualification"]
        assert all(x["returncode"] == 0 for x in pair["qualification"].values())
        control, candidate = pair["variants"]["C"], pair["variants"][lane]
        assert control["prompt_sha256"] == candidate["prompt_sha256"]
        changes = []
        for a, b in zip(control["rows"], candidate["rows"], strict=True):
            assert (a["phase"], a["context_length"], a["batch_size"]) == (
                b["phase"],
                b["context_length"],
                b["batch_size"],
            )
            changes.append(
                {
                    "phase": a["phase"],
                    "context_length": a["context_length"],
                    "batch_size": a["batch_size"],
                    "metric_change_percent": {
                        k: 100 * (b[k]["median"] / a[k]["median"] - 1)
                        for k in a
                        if isinstance(a[k], dict)
                    },
                }
            )
        original = {
            (r["phase"], r["context_length"], r["batch_size"], r["repeat"]): r
            for r in legs["C"]["completion"]["records"]
        }
        disagreements = []
        for record in legs[lane]["completion"]["records"]:
            key = (
                record["phase"],
                record["context_length"],
                record["batch_size"],
                record["repeat"],
            )
            for request, (a, b) in enumerate(
                zip(original[key]["tokens"], record["tokens"], strict=True)
            ):
                mismatch = next(
                    (i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y),
                    None,
                )
                if mismatch is not None:
                    disagreements.append(
                        {"case": key, "request": request, "first_mismatch": mismatch}
                    )
        decode8 = next(
            r for r in changes if r["phase"] == "decode" and r["batch_size"] == 8
        )
        latency_changes = [
            r["metric_change_percent"][metric]
            for r in changes
            for metric in (
                [
                    "mean_per_request_tpot_ms",
                    "p90_inter_token_ms",
                    "max_per_request_tpot_ms",
                ]
                if r["phase"] == "decode"
                else ["mean_ttft_ms"]
            )
        ]
        gain = decode8["metric_change_percent"]["end_to_end_output_tokens_s_gpu"]
        pair.update(
            complete=True,
            changes=changes,
            synthetic_token_disagreements=disagreements,
            qualification_scope={
                "saved_reference_and_affected_contract_tests_passed": True,
                "full_synthetic_generated_sequences_match_control": not disagreements,
                "strict_full_sequence_status": "pass"
                if not disagreements
                else "unresolved; investigate numerical sensitivity versus semantic error",
            },
            success_gate={
                "minimum_batch8_throughput_gain_percent": 10,
                "maximum_latency_regression_percent": 5,
                "batch8_throughput_gain_percent": gain,
                "worst_latency_change_percent": max(latency_changes),
                "performance_pass": gain >= 10 and max(latency_changes) <= 5,
                "latency_scope": "4K/8K prefill mean TTFT plus b1/2/4/8 mean/max per-request TPOT and p90 inter-token latency; first-to-second gaps separately reported",
            },
        )
        report["pairs"][lane] = pair
    old = root.parent / "parallel-20261119/D/vllm-002/result.json"
    if old.exists():
        raw = json.loads(old.read_text())
        validate(raw["completion"])
        report["stock_vllm_reference"] = {
            "hardware": raw["hardware"],
            "version": raw["version"],
            "completion": summarize(raw["completion"]),
            "source": str(old),
        }
    if len(report["pairs"]) == 3 and all(
        x.get("complete") for x in report["pairs"].values()
    ):
        report["status"] = "complete"
    numeric = root / "shared/numeric-diagnosis.json"
    if numeric.exists():
        report["numeric_diagnosis"] = json.loads(numeric.read_text())
    report["protocol"] = json.loads((root / "protocol.json").read_text())
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pairs": {n: p.get("success_gate") for n, p in report["pairs"].items()},
                "case_records": report["validation"]["case_records"],
            }
        )
    )


if __name__ == "__main__":
    main()
