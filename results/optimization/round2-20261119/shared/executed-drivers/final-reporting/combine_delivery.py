"""Replace the incomplete delivery batch-eight case with its paired recovery."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory.resolve()
    initial_dir = root / "delivery/pair-001/recovered"
    recovery_dir = root / "delivery/b8-recovery-001"
    initial = json.loads((initial_dir / "result.json").read_text())
    recovery = json.loads((recovery_dir / "result.json").read_text())
    assert initial["qualification"]["delivery"]["returncode"] == 0
    for name in ["C", "delivery"]:
        a = next(v for v in initial["variants"] if v["name"] == name)
        b = next(v for v in recovery["variants"] if v["name"] == name)
        assert a["source_sha256"] == b["source_sha256"]
        assert a.get("environment", {}) == b.get("environment", {})
    combined = copy.deepcopy(initial)
    combined["completions"] = []
    combined["gpu_function_elapsed_s"] = (
        initial["gpu_function_elapsed_s"] + recovery["gpu_function_elapsed_s"]
    )
    combined["synthesis_note"] = (
        "The initial job timed out at 900s after qualification and all non-b8 cases. "
        "Its partial b8 candidate records are excluded. Replace both control and "
        "candidate b8 records with three fresh cohorts from a second matched allocation. "
        "Source, weights, workload, timing and repeats are unchanged. Raw failures retained."
    )
    combined["hardware_by_case"] = {
        "prefill_and_decode_b1_b2_b4": initial["hardware"],
        "decode_b8": recovery["hardware"],
    }
    combined["raw_sources"] = [str(initial_dir), str(recovery_dir)]
    for name in ["C", "delivery"]:
        original = json.loads((initial_dir / f"completion-{name}.json").read_text())
        fresh_leg = next(leg for leg in recovery["completions"] if leg["name"] == name)
        assert fresh_leg["returncode"] == 0
        fresh = fresh_leg["completion"]
        assert fresh["prompt_sha256"] == original["prompt_sha256"]
        previous = [
            r
            for r in original["records"]
            if not (r["phase"] == "decode" and r["batch_size"] == 8)
        ]
        assert len(previous) == 15
        assert len(fresh["records"]) == 3
        assert all(
            r["phase"] == "decode"
            and r["context_length"] == 8192
            and r["batch_size"] == 8
            for r in fresh["records"]
        )
        original["records"] = previous + fresh["records"]
        combined["completions"].append(
            {
                "name": name,
                "returncode": 0,
                "completion": original,
                "source_note": combined["synthesis_note"],
            }
        )
    output = root / "delivery/combined"
    output.mkdir(exist_ok=True)
    (output / "result.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "records": 36, "initial_timeout_retained": True}
        )
    )


if __name__ == "__main__":
    main()
