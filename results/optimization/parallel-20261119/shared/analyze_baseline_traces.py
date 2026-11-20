"""Read only the coordinator's frozen baseline traces; write lane A analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


SOURCE = Path(
    "/Users/reese/code/resin/results/optimization/parallel-20261119/baseline/"
    "raw-profiles/traces/6f70e63fea9142ab9eb3213ca4ac34b7"
)
OUTPUT = Path(__file__).with_name("baseline-profile-analysis.json")


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
for path in sorted(SOURCE.glob("*.json.gz")):
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
            if owner and owner["forward_id"] == fid:
                attributed.add(gpu_index)
                fgpu.append(event)
                groups[category(owner["names"])].append(event)
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
                    "expected": "52 attention layers and 52*8+1=417 linear calls",
                    "complete": counts["aten::scaled_dot_product_attention"] == 52
                    and counts["aten::linear"] == 417,
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
    "scope": "Only the shared frozen baseline traces supplied by the coordinator",
    "method": {
        "time_unit": "All report durations are milliseconds; source trace ts/dur are microseconds.",
        "forward_count": "Count benchmark.runner.forward annotations, then independently verify each contains 52 aten::scaled_dot_product_attention calls and 417 aten::linear calls.",
        "GPU_attribution": "Match GPU External id to CPU operator ancestry, then to its enclosing benchmark.runner.forward annotation.",
        "GPU_union": "Union of kernel, memcpy and memset intervals on the one GPU; separate kernel sums/union retained.",
        "CPU_intervals": "Inclusive operator/API intervals may nest, overlap GPU work, and include GPU queue waits. They are not CPU compute measurements or additive totals.",
        "mask_category": "Mask/position candidate category groups arange/where/ge/lt/bitwise-and ancestry. It can include other uses; no source-line stack was recorded.",
    },
    "limitations": [
        "Profiler captures are untimed passes after throughput measurement and include instrumentation overhead. Their wall intervals must not be substituted for unprofiled throughput.",
        "GPU inactivity inside an envelope is unattributed; it is not automatically host compute or removable launch overhead.",
        "Kernel/API category durations are descriptive and not additive predicted speedups.",
        "No comparison to other allocations or candidate traces is made here.",
    ],
    "observations": [
        "All four files contain exactly two complete forwards, independently verified with 52 attention and 417 linear calls per forward.",
        "Batch-8 decode spends 57.7–58.5 ms of summed GPU kernel time in 104 repeat_interleave copies per forward; page gathers cost 3.95–4.00 ms, with 832 indexed gathers per forward. This supports targeting dense KV expansion and repeated row work, but does not establish candidate speedups.",
        "The explicit-mask baseline actually dispatches cuDNN flash-attention kernels at both decode and prefill. It is not a math-attention fallback in these traces.",
        "The second forward in each decode trace has a larger expansion grid, consistent with the padded key width increasing from 8448 to 8512 tokens at a 64-token page boundary. Its two longest GPU gaps occur inside host cuDNN attention calls, with Lazy Function Loading events in those calls. This supports new-shape cuDNN setup as a contributor; the whole gap is not proven to be lazy loading or removable host overhead.",
        "Decode host forward times vary 110–200 ms at batch 1 and 190–269 ms at batch 8, while kernel work within each batch remains nearly constant. These profiler intervals include tracing and new-shape setup and should not replace unprofiled throughput windows.",
        "Prefill is GPU-active for about 98–99% of its GPU envelope. At 8K, linear GEMMs account for 515–518 ms of summed kernel time, SDPA for 195 ms, and other mostly pointwise work for about 294 ms. Pure launch cleanup alone cannot be inferred to remove those execution costs.",
    ],
    "traces": results,
}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
for trace in results:
    print(Path(trace["source"]).name, "complete forwards", trace["observed_complete_forward_count"])
    for forward in trace["per_forward"]:
        print(json.dumps({
            "forward": forward["index"],
            "host_ms": forward["host_forward_duration_ms"],
            "kernel_count": forward["kernels"]["count"],
            "kernel_sum_ms": forward["kernels"]["summed_duration_ms"],
            "kernel_union_ms": forward["kernels"]["interval_union_ms"],
            "GPU_active_union_ms": forward["all_GPU_activity"]["interval_union_ms"],
            "GPU_envelope_ms": forward["all_GPU_activity"]["envelope_ms"],
            "GPU_inactive_ms": forward["all_GPU_activity"]["inactive_inside_envelope_ms"],
            "categories_kernel_ms": {k: v["kernels"]["summed_duration_ms"] for k,v in forward["GPU_categories_by_CPU_operator_ancestry"].items()},
        }))
    print("unattributed_GPU", trace["unattributed_GPU_activity"])
print(OUTPUT)
