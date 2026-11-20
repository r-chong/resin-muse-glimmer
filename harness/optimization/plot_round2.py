"""Export a presentation figure from validated common-boundary results."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/resin-round2-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    report = json.loads((root / "analysis.json").read_text())
    assert report["status"] == "complete"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (curve, gains) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    graph_pair = report["pairs"]["graphs"]
    sources = [
        ("Validated C · paired control", graph_pair["variants"]["C"], "#777b88", "o"),
        ("C + CUDA graphs", graph_pair["variants"]["graphs"], "#2563eb", "o"),
        (
            "Stock vLLM · earlier reference",
            report["stock_vllm_reference"]["completion"],
            "#08916e",
            "s",
        ),
    ]
    for label, completion, color, marker in sources:
        rows = sorted(
            (r for r in completion["rows"] if r["phase"] == "decode"),
            key=lambda r: r["batch_size"],
        )
        x = [r["batch_size"] for r in rows]
        stat = [r["end_to_end_output_tokens_s_gpu"] for r in rows]
        y = [s["median"] for s in stat]
        curve.errorbar(
            x,
            y,
            yerr=[
                [s["median"] - s["min"] for s in stat],
                [s["max"] - s["median"] for s in stat],
            ],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.3,
            capsize=3,
        )
        curve.annotate(
            f"{y[-1]:.1f}",
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            weight="bold",
            va="center",
        )
    curve.set(
        title="Graphs narrow the throughput gap",
        xlabel="Concurrent requests",
        ylabel="Output tokens / second / GPU",
    )
    curve.set_xticks([1, 2, 4, 8])
    curve.set_xlim(0.6, 9.4)
    curve.set_ylim(0, 200)
    curve.grid(axis="y", alpha=0.18)
    curve.legend(frameon=False, loc="upper left", fontsize=9)

    lanes = ["graphs", "prefill", "delivery"]
    labels = ["C + graphs", "Native prefill", "Earlier first tokens"]
    values = [
        report["pairs"][n]["success_gate"]["batch8_throughput_gain_percent"]
        for n in lanes
    ]
    colors = ["#2563eb", "#acb5c6", "#acb5c6"]
    gains.barh(labels, values, color=colors, height=0.5)
    for i, value in enumerate(values):
        gains.text(max(value, 0) + 0.6, i, f"{value:+.1f}%", va="center", weight="bold")
    gains.invert_yaxis()
    gains.axvline(0, color="#b6bdc8", linewidth=0.8)
    gains.set(
        title="Each change versus its own C control",
        xlabel="Batch-eight completion throughput gain",
    )
    gains.set_xlim(min(-1.5, min(values) - 2), max(values) + 8)
    gains.grid(axis="x", alpha=0.15)
    gains.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    fig.suptitle(
        "Resin optimization · second round",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.07,
        0.905,
        "Same Muse Glimmer checkpoint · BF16 · one H100 · 8,192 input + 256 output tokens",
        fontsize=10,
        color="#4b5563",
    )
    fig.text(
        0.07,
        0.055,
        "Medians of three cohorts; error bars show observed ranges. Timing includes prefill through CPU-visible final token.\n"
        "Each candidate/control pair shares one GPU allocation. Stock vLLM uses its own stack and an earlier, separate H100 allocation.",
        fontsize=9,
        color="#4b5563",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.81, wspace=0.37)
    fig.savefig(root / "throughput.png", dpi=190, facecolor="white")
    fig.savefig(root / "throughput.svg", facecolor="white")


if __name__ == "__main__":
    main()
