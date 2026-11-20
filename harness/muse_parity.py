"""Run the text-only Muse Glimmer parity test through Resin on one Modal H100.

Uses Resin's own qualification function (deploy/modal_bench.py::qualify) from
the patched ``resin-source`` checkout. The HF reference ``.npz`` must already
be on the ``resin-benchmark-traces-v1`` volume (mounted at /bench-traces).

    resin-source/.venv/bin/python harness/muse_parity.py \
        --reference /bench-traces/muse-reference/e0b3ae6b/gsm8k_test_0001.npz \
        --reference /bench-traces/muse-reference/69d7fbf4/long_window.npz --steps 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESIN = ROOT / "resin-source"
TEST = "tests/test_muse_glimmer_hf_parity.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, action="append", help="reference .npz on the trace volume (repeatable)")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "parity")
    args = parser.parse_args()

    sys.path.insert(0, str(RESIN))
    # Resin locks modal 1.5.3, which cannot build with this workspace's
    # default image builder (2026.10); 2025.06 is its newest supported.
    os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
    import modal

    from deploy import modal_bench

    # resin-source/ is vendored in this repo; identify it by this repo's commit.
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "resin-source"], text=True).strip())
    with modal.enable_output(), modal_bench.app.run():
        result = modal_bench.qualify.with_options(gpu="H100!").remote(
            "muse-glimmer-30b",
            [TEST, "-s"],
            environment={"RESIN_MUSE_REFERENCES": ",".join(args.reference), "RESIN_MUSE_STEPS": str(args.steps)},
        )
    report = [
        json.loads(line.split("MUSE_PARITY_REPORT ", 1)[1])
        for line in result["output"].splitlines()
        if "MUSE_PARITY_REPORT " in line
    ]
    record = {
        "repo_commit": source,
        "resin_source_dirty": dirty,
        "reference": args.reference,
        "steps": args.steps,
        "returncode": result["returncode"],
        "elapsed_s": result["elapsed_s"],
        "report": report,
        "output": result["output"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "output"}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
