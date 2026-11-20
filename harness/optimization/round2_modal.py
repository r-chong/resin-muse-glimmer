"""Stage immutable round-two variants for the established validation executor."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import modal
from paired import diagnostic_fragment, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    config = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
    source = Path(config["base_source"]).resolve()
    sys.path.insert(0, str(source))
    from deploy.modal_image import build_modal_image

    image = (
        build_modal_image(source)
        .add_local_file(
            source / "scripts/bench.py", "/root/scripts/bench.py", copy=True
        )
        .add_local_file(
            source / "scripts/merge_bench_traces.py",
            "/root/scripts/merge_bench_traces.py",
            copy=True,
        )
        .uv_pip_install("pytest>=8,<9")
        .add_local_file(
            Path(__file__).with_name("completion_bench.py"),
            "/root/completion_bench.py",
            copy=True,
        )
    )
    fragment = diagnostic_fragment()
    for local, remote in config.get("extra_local_files", {}).items():
        image = image.add_local_file(Path(local), remote, copy=True)
    for variant in config["variants"]:
        name = variant["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("variant names must be alphanumeric")
        src = Path(variant["source"]).resolve()
        image = image.add_local_dir(
            src / "models/torch/resin_models_torch",
            f"/variants/{name}/resin_models_torch",
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc"],
        )
        image = image.add_local_dir(src / "tests", f"/variants/{name}/tests", copy=True)
        image = image.add_local_dir(
            src / "models/torch/tests",
            f"/variants/{name}/models/torch/tests",
            copy=True,
        )
        combined = args.output / f"qualified-{name}.py"
        combined.write_text(
            (src / "tests/test_muse_glimmer_hf_parity.py").read_text() + "\n" + fragment
        )
        image = image.add_local_file(
            combined,
            f"/variants/{name}/tests/test_muse_glimmer_qualified.py",
            copy=True,
        )
        variant["source_sha256"] = {
            str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((src / "models/torch/resin_models_torch").rglob("*.py"))
        }
    volume_name = "resin-opt-" + args.job_id
    config["volume"] = volume_name
    config["checkpoint"] = (
        "/models/resin-muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69"
    )
    config.setdefault("completion_only", True)
    config["harness_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("round2_modal.py", "paired.py", "completion_bench.py")
    }
    (args.output / "request.json").write_text(json.dumps(config, indent=2))
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    app = modal.App("resin-opt-" + args.job_id)
    worker = app.function(
        image=image,
        gpu="H100!",
        cpu=32,
        memory=(131072, 262144),
        timeout=args.timeout,
        retries=0,
        max_containers=1,
        scaledown_window=2,
        single_use_containers=True,
        volumes={
            "/models/resin-muse-glimmer-30b": modal.Volume.from_name(
                "resin-muse-glimmer-30b-models-v1"
            ).with_mount_options(read_only=True),
            "/bench-traces": modal.Volume.from_name(
                "resin-benchmark-traces-v1"
            ).with_mount_options(read_only=True),
            "/experiment": volume,
        },
    )(execute)
    with modal.enable_output(), app.run():
        call = worker.spawn(config)
        (args.output / "function-call.json").write_text(
            json.dumps({"object_id": call.object_id})
        )
        result = call.get()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpu_elapsed_s": result["gpu_function_elapsed_s"],
                "qualification": {
                    n: r["returncode"] for n, r in result["qualification"].items()
                },
                "completion_results": [
                    (r["name"], r["returncode"]) for r in result.get("completions", [])
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
