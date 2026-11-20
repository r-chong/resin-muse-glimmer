"""Analyze the validated C runner traces; never use profiled wall time as throughput."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


SOURCE = Path(__file__).resolve().parents[1] / "profile/traces/traces/3d24c619bd634e5ba3a04bbbb3afbf62"
OUTPUT = Path(__file__).with_name("c-profile-analysis.json")


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total, current_end = 0.0, float("-inf")
    for start, end in intervals:
        total += max(0.0, end - max(start, current_end))
        current_end = max(current_end, end)
    return total


def summary(events):
    return {
        "count": len(events),
        "summed_duration_ms": sum(e["dur"] for e in events) / 1000,
        "interval_union_ms": union_us(events) / 1000,
    }


def gpu_summary(events):
    s = summary(events)
    if not events:
        return s
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    envelope = merged[-1][1] - merged[0][0]
    s.update(
        {
            "envelope_ms": envelope / 1000,
            "inactive_inside_envelope_ms": (envelope - union_us(events)) / 1000,
            "active_fraction_of_envelope": union_us(events) / envelope,
            "gaps_gt_10us": sum(g > 10 for g in gaps),
            "gaps_gt_100us": sum(g > 100 for g in gaps),
            "largest_gaps_ms": sorted((g / 1000 for g in gaps), reverse=True)[:10],
        }
    )
    return s


def category(names):
    # Exclusive categories use CPU operator ancestry rather than kernel names.
    # Their duration totals are descriptive; they are not additive savings.
    if "aten::repeat_interleave" in names:
        return "GQA_repeat_interleave"
    if "aten::pad" in names or "aten::constant_pad_nd" in names:
        return "padding"
    if "aten::stack" in names:
        return "stacking"
    if "aten::index_put_" in names or "aten::_index_put_impl_" in names:
        return "page_writes_index_put"
    if "aten::index" in names:
        return "page_gathers_index"
    if "aten::scaled_dot_product_attention" in names:
        return "SDPA"
    if "aten::linear" in names or "aten::mm" in names:
        return "linear_GEMM"
    if names.intersection(
        {
            "aten::arange", "aten::where", "aten::ge", "aten::lt",
            "aten::bitwise_and", "aten::bitwise_and_",
        }
    ):
        return "mask_position_candidate_ops"
    return "other"


results = []
for path in sorted(SOURCE.glob("*rank000.json.gz")):
    trace = json.load(gzip.open(path, "rt"))
    events = trace["traceEvents"]
    cpu_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")
    ]
    by_thread = defaultdict(list)
    for event in cpu_events:
        by_thread[event["pid"], event["tid"]].append(event)
    info = {}
    for thread in by_thread.values():
        stack = []
        for event in sorted(thread, key=lambda e: (e["ts"], -e["dur"])):
            end = event["ts"] + event["dur"]
            while stack and (
                event["ts"] >= stack[-1]["ts"] + stack[-1]["dur"]
                or end > stack[-1]["ts"] + stack[-1]["dur"] + 0.01
            ):
                stack.pop()
            ancestors = [*stack, event]
            forwards = [
                e for e in ancestors if e["name"] == "benchmark.runner.forward"
            ]
            info[event.get("args", {}).get("External id")] = {
                "event": event,
                "names": {e["name"] for e in ancestors},
                "forward_id": forwards[-1]["args"]["External id"] if forwards else None,
            }
            stack.append(event)
    forwards = sorted(
        (e for e in cpu_events if e["name"] == "benchmark.runner.forward"),
        key=lambda e: e["ts"],
    )
    gpu_events = [
        e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    runtime_by_correlation = {
        e.get("args", {}).get("correlation"): e for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    per_forward = []
    attributed = set()
    for index, forward in enumerate(forwards):
        fid = forward["args"]["External id"]
        fcpu = [v["event"] for v in info.values() if v["forward_id"] == fid]
        counts = Counter(e["name"] for e in fcpu if e["cat"] == "cpu_op")
        fgpu = []
        groups = defaultdict(list)
        for gpu_index, event in enumerate(gpu_events):
            owner = info.get(event.get("args", {}).get("External id"))
            if owner is None:
                launch = runtime_by_correlation.get(event.get("args", {}).get("correlation"))
                if (launch and launch["pid"] == forward["pid"]
                    and launch["tid"] == forward["tid"]
                    and forward["ts"] <= launch["ts"]
                    and launch["ts"] + launch["dur"] <= forward["ts"] + forward["dur"] + 0.01):
                    owner = {"forward_id": fid, "names": set()}
            if owner and owner["forward_id"] == fid:
                attributed.add(gpu_index)
                fgpu.append(event)
                groups["native_paged_attention" if "flashinfer::" in event["name"] else category(owner["names"])].append(event)
        kernels = [e for e in fgpu if e["cat"] == "kernel"]
        top_kernels = defaultdict(list)
        for event in kernels:
            top_kernels[event["name"]].append(event)
        named_cpu = defaultdict(list)
        for event in fcpu:
            if event["cat"] == "cpu_op":
                named_cpu[event["name"]].append(event)
        relevant_cpu_names = (
            "aten::linear", "aten::scaled_dot_product_attention", "aten::repeat_interleave",
            "aten::index", "aten::index_put_", "aten::pad", "aten::stack", "aten::arange",
            "aten::where", "aten::to", "aten::_to_copy", "aten::copy_",
        )
        all_forward_intervals = (
            forward["ts"], forward["ts"] + forward["dur"]
        )
        runtime_events = [
            e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and e["ts"] >= all_forward_intervals[0]
            and e["ts"] + e["dur"] <= all_forward_intervals[1] + 0.01
        ]
        runtime_by_name = defaultdict(list)
        for event in runtime_events:
            runtime_by_name[event["name"]].append(event)
        gqa_configurations = Counter(
            (
                event["name"],
                tuple(event["args"].get("grid", [])),
                tuple(event["args"].get("block", [])),
            )
            for event in groups.get("GQA_repeat_interleave", [])
            if event["cat"] == "kernel"
        )
        inferred_widths = []
        if path.name.startswith("decode-"):
            batch = int(re.search(r"-b(\d+)-", path.name).group(1))
            for (name, grid, block), count in gqa_configurations.items():
                if "elementwise_kernel<128, 4," in name and len(grid) == 3:
                    inferred_widths.append(
                        {
                            "expanded_KV_width_tokens": grid[0] * 128 * 4 // (batch * 32 * 128),
                            "kernel_count": count,
                            "grid": list(grid),
                            "inference": "Direct-copy launch has 128 threads x 4 elements, for batch x 32 expanded heads x tokens x 128 dimensions. Input shapes were not recorded.",
                        }
                    )
        per_forward.append(
            {
                "index": index,
                "host_forward_duration_ms": forward["dur"] / 1000,
                "complete_forward_evidence": {
                    "SDPA_calls": counts["aten::scaled_dot_product_attention"],
                    "linear_calls": counts["aten::linear"],
                    "native_attention_main_kernels": sum("BatchPrefillWithPagedKVCacheKernel" in e["name"] for e in kernels),
                    "expected": "52 SDPA calls in prefill or 52 native attention main kernels in decode; 417 linear calls",
                    "complete": counts["aten::linear"] == 417 and (
                        counts["aten::scaled_dot_product_attention"] == 52 if path.name.startswith("prefill-")
                        else sum("BatchPrefillWithPagedKVCacheKernel" in e["name"] for e in kernels) == 52),
                },
                "kernels": summary(kernels),
                "all_GPU_activity": gpu_summary(fgpu),
                "GPU_categories_by_CPU_operator_ancestry": {
                    name: {
                        "all_activity": summary(group_events),
                        "kernels": summary([e for e in group_events if e["cat"] == "kernel"]),
                        "memcpy": summary([e for e in group_events if e["cat"] == "gpu_memcpy"]),
                    }
                    for name, group_events in sorted(groups.items())
                },
                "CPU_operator_inclusive_intervals": {
                    name: summary(named_cpu[name]) for name in relevant_cpu_names
                },
                "longest_host_SDPA_calls": [
                    {
                        "duration_ms": event["dur"] / 1000,
                        "start_relative_to_forward_ms": (event["ts"] - forward["ts"]) / 1000,
                        "overlapping_lazy_function_loads": [
                            {
                                "duration_ms": overhead["dur"] / 1000,
                                "start_relative_to_SDPA_ms": (overhead["ts"] - event["ts"]) / 1000,
                            }
                            for overhead in events
                            if overhead.get("cat") == "overhead"
                            and overhead["name"] == "Lazy Function Loading"
                            and overhead["ts"] < event["ts"] + event["dur"]
                            and overhead["ts"] + overhead["dur"] > event["ts"]
                        ],
                    }
                    for event in sorted(
                        named_cpu["aten::scaled_dot_product_attention"],
                        key=lambda e: -e["dur"],
                    )[:5]
                ],
                "GQA_copy_launch_width_inference": inferred_widths,
                "SDPA_kernel_names": dict(Counter(
                    event["name"] for event in groups.get("SDPA", [])
                    if event["cat"] == "kernel" and "cudnn" in event["name"]
                )),
                "CUDA_API_intervals": {
                    name: summary(vals) for name, vals in sorted(runtime_by_name.items())
                },
                "top_kernels_by_summed_duration": [
                    {"name": name, **summary(vals)}
                    for name, vals in sorted(
                        top_kernels.items(), key=lambda item: -sum(e["dur"] for e in item[1])
                    )[:20]
                ],
                "GPU_activity_beyond_host_forward_end": summary(
                    [e for e in fgpu if e["ts"] + e["dur"] > all_forward_intervals[1] + 0.01]
                ),
            }
        )
    all_kernel_names = Counter(e["name"] for e in events if e.get("cat") == "kernel")
    results.append(
        {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "resin_timing": trace.get("resin_timing"),
            "observed_complete_forward_count": sum(
                f["complete_forward_evidence"]["complete"] for f in per_forward
            ),
            "forward_annotations": len(forwards),
            "all_kernels": summary([e for e in gpu_events if e["cat"] == "kernel"]),
            "all_GPU_activity": gpu_summary(gpu_events),
            "unattributed_GPU_activity": summary(
                [e for i, e in enumerate(gpu_events) if i not in attributed]
            ),
            "per_forward": per_forward,
        }
    )

report = {
    "schema_version": 1,
    "source_commit": "66bb70a63470581f1ced548265e839da374836aa",
    "scope": "Four untimed profiler captures of validated C, one H100, two full forwards each",
    "method": {
        "durations": "Milliseconds; source trace durations are microseconds.",
        "forward_attribution": "Match GPU External id through CPU operator ancestry; missing External ids fall back to correlation-matched CUDA launch events on the forward host thread inside its interval.",
        "native_attention": "GPU kernel names containing flashinfer:: grouped separately; main-kernel count excludes merge kernels.",
        "complete": "417 linear calls and either 52 SDPA calls (prefill) or 52 native attention main kernels (decode).",
        "activity": "Union of GPU kernel/memcpy/memset intervals; sums and envelope separately reported.",
    },
    "limitations": [
        "Profiles follow throughput windows and include instrumentation overhead; wall times are not throughput measurements.",
        "CPU intervals nest and may include GPU waits; do not add CPU categories or infer removable launch overhead from GPU gaps alone.",
        "Kernel category durations describe the captured execution; they do not predict additive speedups.",
        "Results are not a paired timing comparison with the original baseline or stock vLLM.",
    ],
    "traces": results,
}
assert len(results) == 4
assert all(t["observed_complete_forward_count"] == 2 for t in results)
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
for trace in results:
    print(Path(trace["source"]).name)
    for f in trace["per_forward"]:
        print(json.dumps({"host_ms":f["host_forward_duration_ms"],"kernel_count":f["kernels"]["count"],"kernel_sum_ms":f["kernels"]["summed_duration_ms"],"gpu_active_ms":f["all_GPU_activity"]["interval_union_ms"],"gpu_envelope_ms":f["all_GPU_activity"]["envelope_ms"],"category_kernel_ms":{k: v["kernels"]["summed_duration_ms"] for k,v in f["GPU_categories_by_CPU_operator_ancestry"].items()}}))
