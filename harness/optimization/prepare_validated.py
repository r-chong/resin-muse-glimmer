"""Prepare immutable sources for the validated C/graphs comparison; no remote calls."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

CONTROL_COMMIT = "66bb70a63470581f1ced548265e839da374836aa"


def prepare(directory: Path, candidate_ref: str) -> Path:
    repository = Path(__file__).resolve().parents[2]
    candidate = subprocess.check_output(
        ["git", "rev-parse", "--verify", candidate_ref + "^{commit}"],
        cwd=repository,
        text=True,
    ).strip()
    subprocess.run(
        ["git", "cat-file", "-e", CONTROL_COMMIT + "^{commit}"],
        cwd=repository,
        check=True,
    )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    variants = []
    for name, commit in [("C", CONTROL_COMMIT), ("graphs", candidate)]:
        snapshot = directory / "snapshots" / name
        snapshot.mkdir(parents=True)
        archive = subprocess.check_output(
            ["git", "archive", commit, "resin-source"], cwd=repository
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(snapshot, filter="data")
        root = snapshot / "resin-source"
        tests = ["tests/test_muse_glimmer_flashinfer.py"]
        variant = {
            "name": name,
            "source": str(root),
            "commit": commit,
            "qualify": True,
            "completion": True,
            "extra_tests": tests,
        }
        if name == "graphs":
            if not (
                root / "models/torch/resin_models_torch/muse_glimmer/execution.py"
            ).is_file():
                raise ValueError("candidate must contain the validated graph runner")
            variant["environment"] = {"RESIN_MUSE_DECODE_EXECUTION": "graphs"}
            tests[:0] = [
                "models/torch/tests/test_muse_glimmer_graph_failures.py",
                "models/torch/tests/test_muse_glimmer_graphs.py",
            ]
        variants.append(variant)
    config = {
        "base_source": variants[1]["source"],
        "completion_only": True,
        "variants": variants,
    }
    spec = directory / "spec.json"
    spec.write_text(json.dumps(config, indent=2) + "\n")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--candidate-ref", default="HEAD")
    args = parser.parse_args()
    print(prepare(args.directory, args.candidate_ref))


if __name__ == "__main__":
    main()
