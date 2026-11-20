"""Summarize raw optimization records without mixing timing boundaries."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    s = sorted(values)
    x = (len(s) - 1) * q
    i = int(x)
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (x - i)


def describe(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
        "values": values,
    }


def internal_summary(all_records):
    groups = defaultdict(list)
    for r in all_records:
        groups[(r["phase"], r["context_length"], r["batch_size"])].append(r)
    rows = []
    for (phase, context, batch), records in sorted(groups.items()):
        row = {
            "phase": phase,
            "context_length": context,
            "batch_size": batch,
            "runs": len(records),
        }
        if phase == "decode":
            row["output_tokens_s_gpu"] = describe(
                [
                    r["committed_tokens"] / r["window_seconds"] / r["gpu_count"]
                    for r in records
                ]
            )
            row["mean_tpot_ms"] = describe(
                [1000 * r["window_seconds"] / len(r["step_seconds"]) for r in records]
            )
            row["p90_tpot_ms"] = describe(
                [1000 * percentile(r["step_seconds"], 0.9) for r in records]
            )
            row["graph_counts"] = [r.get("graph_counts", {}) for r in records]
        else:
            row["ttft_ms"] = describe(
                [1000 * statistics.median(r["prefill_seconds"]) for r in records]
            )
        rows.append(row)
    return rows


def completion_summary(completion):
    groups = defaultdict(list)
    for r in completion["records"]:
        groups[(r["phase"], r["context_length"], r["batch_size"])].append(r)
    rows = []
    for (phase, context, batch), records in sorted(groups.items()):
        row = {
            "phase": phase,
            "context_length": context,
            "batch_size": batch,
            "runs": len(records),
            "end_to_end_output_tokens_s_gpu": describe(
                [r["completion_output_tokens_s_gpu"] for r in records]
            ),
            "mean_ttft_ms": describe(
                [1000 * statistics.mean(r["ttft_s"]) for r in records]
            ),
        }
        if phase == "decode":
            row["mean_per_request_tpot_ms"] = describe(
                [1000 * statistics.mean(r["per_request_tpot_s"]) for r in records]
            )
            row["post_warmup_completion_window_output_tokens_s_gpu"] = describe(
                [r["post_warmup_output_tokens_s_gpu"] for r in records]
            )
        rows.append(row)
    return {
        "startup_s": completion["startup_s"],
        "prompt_sha256": completion["prompt_sha256"],
        "rows": rows,
        "reference_checks": completion["references"],
        "timing": completion["timing"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    root = args.directory
    report = {
        "status": "in_progress",
        "frozen_commit": "73532b9",
        "timing_boundaries": {
            "internal": "Resin synchronized-step sum, warmup excluded; full prefill cohorts",
            "completion": "submit all requests through CPU-visible final token; includes prefill; no HTTP or detokenization",
        },
        "initial_candidates": {},
        "paired_validation": {},
        "completion_comparison": {},
        "limitations": [
            "Historical frozen numbers vary substantially from the fresh frozen-source allocation; speedup claims use paired validation only.",
            "One agent per optimization path; this is an optimization feasibility experiment, not a causal comparison of agent skill or frameworks.",
            "GSM8K prompt parity is not dataset-level GSM8K accuracy.",
            "Images, prefix caching, speculation, overlap and memory pressure are outside the qualified scope.",
            "Stock vLLM and Resin use their own pinned dependency stacks and separate physical H100s for the external reference. Resin internal finalists share a physical H100.",
            "Stock vLLM disables image/video inputs; Resin benchmark retains unused vision weights. Both execute the same text-only forward workload.",
        ],
    }
    baseline = root / "baseline/run-002/result.json"
    if baseline.exists():
        r = json.loads(baseline.read_text())
        report["fresh_baseline"] = {
            "hardware": r["hardware"],
            "rows": internal_summary(r["benchmark"]),
            "process_returncode": r["commands"][0]["returncode"],
            "postprocessing": "all18 primary records and4 raw traces complete; missing remote merge helper repaired locally",
        }
    for lane, job in [("A", "a1-002"), ("B", "b1-001"), ("C", "c1-002")]:
        f = root / lane / job / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            report["initial_candidates"][lane] = {
                "hardware": r["hardware"],
                "gpu_function_body_s": r["gpu_function_elapsed_s"],
                "commands": [
                    {k: c[k] for k in ["name", "returncode", "elapsed_s"]}
                    for c in r["commands"]
                ],
                "rows": internal_summary(r.get("benchmark", [])),
                "source": str(f.relative_to(root)),
            }
    paired_files = sorted(root.glob("D/paired-*/result.json"))
    if paired_files:
        f = paired_files[-1]
        r = json.loads(f.read_text())
        grouped = defaultdict(list)
        for leg in r["legs"]:
            if leg["returncode"] == 0:
                grouped[leg["name"]].extend(leg.get("benchmark", []))
        report["paired_validation"] = {
            "source": str(f.relative_to(root)),
            "hardware": r["hardware"],
            "qualification": {
                n: {k: v[k] for k in ["returncode", "elapsed_s"]}
                for n, v in r["qualification"].items()
            },
            "variants": {n: internal_summary(rows) for n, rows in grouped.items()},
            "gpu_function_body_s": r["gpu_function_elapsed_s"],
            "orders": [
                {
                    "round": leg["round"],
                    "variant": leg["name"],
                    "order": leg["order"],
                    "returncode": leg["returncode"],
                }
                for leg in r["legs"]
            ],
        }
        base = {
            (x["phase"], x["context_length"], x["batch_size"]): x
            for x in report["paired_validation"]["variants"].get("baseline", [])
        }
        for name, rows in report["paired_validation"]["variants"].items():
            if name == "baseline":
                continue
            regressions = []
            batch8_gain = None
            for row in rows:
                key = (row["phase"], row["context_length"], row["batch_size"])
                control = base[key]
                if row["phase"] == "decode":
                    row["throughput_gain_percent"] = 100 * (
                        row["output_tokens_s_gpu"]["median"]
                        / control["output_tokens_s_gpu"]["median"]
                        - 1
                    )
                    if row["batch_size"] == 8:
                        batch8_gain = row["throughput_gain_percent"]
                    metrics = ["mean_tpot_ms", "p90_tpot_ms"]
                else:
                    metrics = ["ttft_ms"]
                row["latency_change_percent"] = {
                    m: 100 * (row[m]["median"] / control[m]["median"] - 1)
                    for m in metrics
                }
                regressions.extend(row["latency_change_percent"].values())
            report["paired_validation"].setdefault("success_gate", {})[name] = {
                "batch8_gain_percent": batch8_gain,
                "worst_latency_change_percent": max(regressions),
                "required_batch8_gain_percent": 10,
                "maximum_latency_regression_percent": 5,
                "performance_pass": batch8_gain is not None
                and batch8_gain >= 10
                and max(regressions) <= 5,
                "all_full_curve_repeats_present": len(rows) == 6
                and all(row["runs"] == 3 for row in rows),
            }
        for leg in r.get("completions", []):
            if leg.get("completion") and leg["returncode"] == 0:
                report["completion_comparison"][leg["name"]] = completion_summary(
                    leg["completion"]
                )
        if len(r["legs"]) == 12 and all(leg["returncode"] == 0 for leg in r["legs"]):
            report["status"] = "paired_internal_complete"
    split_pairs = sorted(root.glob("D/pair-*/result.json"))
    if split_pairs:
        report["paired_validation"] = {"pairs": {}}
        for f in split_pairs:
            r = json.loads(f.read_text())
            if r.get("legs"):
                candidate = next(
                    v["name"] for v in r["variants"] if v["name"] != "baseline"
                )
                grouped = defaultdict(list)
                for leg in r["legs"]:
                    if leg["returncode"] == 0:
                        grouped[leg["name"]].extend(leg.get("benchmark", []))
                summaries = {n: internal_summary(rows) for n, rows in grouped.items()}
                pair = {
                    "source": str(f.relative_to(root)),
                    "hardware": r["hardware"],
                    "variants": summaries,
                    "qualification": {
                        n: {k: v[k] for k in ["returncode", "elapsed_s"]}
                        for n, v in r["qualification"].items()
                    },
                    "orders": [
                        {
                            "round": l["round"],
                            "variant": l["name"],
                            "order": l["order"],
                            "returncode": l["returncode"],
                        }
                        for l in r["legs"]
                    ],
                    "gpu_function_body_s": r["gpu_function_elapsed_s"],
                }
                base = {
                    (x["phase"], x["context_length"], x["batch_size"]): x
                    for x in summaries.get("baseline", [])
                }
                regressions = []
                gain = None
                for row in summaries.get(candidate, []):
                    control = base.get(
                        (row["phase"], row["context_length"], row["batch_size"])
                    )
                    if not control:
                        continue
                    if row["phase"] == "decode":
                        row["throughput_gain_percent"] = 100 * (
                            row["output_tokens_s_gpu"]["median"]
                            / control["output_tokens_s_gpu"]["median"]
                            - 1
                        )
                        if row["batch_size"] == 8:
                            gain = row["throughput_gain_percent"]
                        metrics = ["mean_tpot_ms", "p90_tpot_ms"]
                    else:
                        metrics = ["ttft_ms"]
                    row["latency_change_percent"] = {
                        m: 100 * (row[m]["median"] / control[m]["median"] - 1)
                        for m in metrics
                    }
                    regressions.extend(row["latency_change_percent"].values())
                rows = summaries.get(candidate, [])
                pair["success_gate"] = {
                    "batch8_gain_percent": gain,
                    "worst_latency_change_percent": max(regressions)
                    if regressions
                    else None,
                    "performance_pass": gain is not None
                    and gain >= 10
                    and max(regressions) <= 5,
                    "all_full_curve_repeats_present": len(rows) == 6
                    and all(row["runs"] == 3 for row in rows)
                    and len(summaries.get("baseline", [])) == 6
                    and all(row["runs"] == 3 for row in summaries.get("baseline", [])),
                    "all_commands_passed": all(l["returncode"] == 0 for l in r["legs"])
                    and all(v["returncode"] == 0 for v in r["qualification"].values()),
                }
                report["paired_validation"]["pairs"][candidate] = pair
            for leg in r.get("completions", []):
                if leg.get("completion") and leg["returncode"] == 0:
                    report["completion_comparison"][leg["name"]] = completion_summary(
                        leg["completion"]
                    )
        pairs = report["paired_validation"]["pairs"]
        if len(pairs) == 3 and all(
            v["success_gate"]["all_full_curve_repeats_present"]
            and v["success_gate"]["all_commands_passed"]
            for v in pairs.values()
        ):
            report["status"] = "paired_internal_complete"
    for f in sorted(root.glob("D/vllm-*/result.json")):
        r = json.loads(f.read_text())
        if r.get("completion") and r.get("returncode") == 0:
            report["completion_comparison"]["stock_vllm"] = completion_summary(
                r["completion"]
            )
            report["stock_vllm"] = {
                k: r[k] for k in ["version", "hardware", "gpu_function_elapsed_s"]
            }
            report["stock_vllm"]["source"] = str(f.relative_to(root))
    if report["status"] == "paired_internal_complete" and {
        "baseline",
        "C",
        "stock_vllm",
    } <= set(report["completion_comparison"]):
        completions = report["completion_comparison"]
        report["completion_prompts_match"] = (
            len({v["prompt_sha256"] for v in completions.values()}) == 1
        )
        if report["completion_prompts_match"] and all(
            len(v["rows"]) == 6 and all(row["runs"] == 3 for row in v["rows"])
            for v in completions.values()
        ):
            report["status"] = "complete"
    report["limitations"].extend(
        [
            "Engine-construction startup_s is not a matched cold-start comparison: C initializes native attention lazily during untimed references/warmup, while vLLM constructs and compiles its engine eagerly.",
            "CUDA allocator peaks are process scoped; vLLM workers use a child process, so driver-process peaks are not compared as total engine memory.",
            "Actual-runner diagnostics cover the three saved reference prompts. Exact greedy/argmax checks pass; no new numerical acceptance threshold or dataset-level quality score is claimed.",
            "A1 timing is an initial unpaired measurement. A2 cumulative gains are paired against frozen control; the incremental A1-to-A2 contribution was not isolated on one GPU.",
            "Model-code diff size excludes tests and shared evaluation tooling. Shared setup, image builds and GPU queue delays are included in the recorded wall time; active agent runtime was not separately instrumented.",
            "No from-scratch performance comparison or vLLM/SGLang customization productivity experiment was performed.",
        ]
    )
    report["protocol"] = json.loads((root / "protocol.json").read_text())
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "paired_variants": list(
                    report["paired_validation"].get(
                        "pairs", report["paired_validation"].get("variants", {})
                    )
                ),
                "completion_backends": list(report["completion_comparison"]),
            }
        )
    )


if __name__ == "__main__":
    main()
