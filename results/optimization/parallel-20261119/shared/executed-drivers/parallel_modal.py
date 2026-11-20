"""Isolated, bounded Modal transport for the parallel Resin experiment.

Measurement remains in Resin's shared benchmark; this file stages immutable
source snapshots, read-only inputs and per-job output namespaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def execute(config):
    import importlib.metadata
    import platform
    import subprocess
    import time
    from pathlib import Path
    started = time.time()
    checkpoint = "/models/resin-muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69"
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/tmp/huggingface", "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
        "TRITON_CACHE_DIR": "/tmp/triton", "XDG_CACHE_HOME": "/tmp/cache",
        "RESIN_TEST_CHECKPOINT": checkpoint,
        "RESIN_MUSE_REFERENCES": ",".join([
            "/bench-traces/muse-reference/e0b3ae6b/gsm8k_test_0001.npz",
            "/bench-traces/muse-reference/69d7fbf4/long_window.npz",
            "/bench-traces/muse-reference/69d7fbf4/long_8k.npz",
        ]),
        "RESIN_MUSE_STEPS": "16", "RESIN_MUSE_DUMP_DIR": "/experiment/diagnostics",
        "PYTHONPATH": "/opt/resin/tests",
    })
    environment.update(config.get("environment", {}))
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    if not hardware.startswith("NVIDIA H100 80GB HBM3,"):
        raise RuntimeError("unexpected hardware: " + hardware)
    result = {"hardware": hardware, "python": platform.python_version(), "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, "commands": [], "started_unix": started}

    def run(command, name):
        t = time.monotonic()
        lines = []
        with subprocess.Popen(command, cwd="/root", env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                lines.append(line)
                print(line, end="", flush=True)
            process.wait()
        Path("/experiment", name + ".log").write_text("".join(lines))
        record = {"name": name, "command": command, "returncode": process.returncode, "elapsed_s": time.monotonic() - t, "output": "".join(lines)}
        result["commands"].append(record)
        return process.returncode == 0

    try:
        qualified = True
        if config["qualify"]:
            qualified = run(["python", "-m", "pytest", "-o", "pythonpath=", *["/opt/resin/" + p for p in config.get("extra_tests", [])], "/opt/resin/tests/test_muse_glimmer_qualified.py", "-s", "-v", "--tb=short"], "qualification")
        if config["bench"] and qualified:
            command = ["python", "/root/scripts/bench.py", "--backend", "local", "--model", "muse-glimmer-30b", "--checkpoint", checkpoint, "--prompt-source", "synthetic", "--phases", "prefill,decode", "--prefill-context-lengths", "4096,8192", "--prefill-batch-sizes", "1", "--decode-context-lengths", "8192", "--decode-batch-sizes", config["batches"], "--warmup-steps", "5", "--measure-steps", "250", "--repeats", str(config["repeats"]), "--prefill-graph", "disabled", "--decode-graph", "disabled", "--no-overlap", "--output", "/experiment/benchmark.jsonl"]
            if config["profile"]:
                command += ["--trace-dir", "/experiment/traces", "--trace-after-sweep", "--trace-batch-sizes", "1,8"]
            run(command, "benchmark")
            path = Path("/experiment/benchmark.jsonl")
            if path.exists(): result["benchmark"] = [json.loads(line) for line in path.read_text().splitlines()]
        result["gpu_function_elapsed_s"] = time.time() - started
        Path("/experiment/result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        import modal
        modal.Volume.from_name(config["artifacts_volume"]).commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--extra-test", action="append", default=[])
    args = parser.parse_args()
    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
    sys.path.insert(0, str(source))
    import modal
    from deploy.modal_image import build_modal_image

    diagnostic = Path(__file__).with_name("test_executing_path.py").read_text()
    diagnostic = diagnostic.replace("from test_muse_glimmer_hf_parity import runner, references, _generate, _logit_summary", "")
    combined = args.output / "test_muse_glimmer_qualified.py"
    combined.write_text((source / "tests/test_muse_glimmer_hf_parity.py").read_text() + "\n" + diagnostic)

    image = (
        build_modal_image(source)
        .add_local_file(source / "scripts/bench.py", "/root/scripts/bench.py", copy=True)
        .uv_pip_install("pytest>=8,<9")
        .add_local_dir(source / "tests", "/opt/resin/tests", copy=True)
        .add_local_dir(source / "models/torch/tests", "/opt/resin/models/torch/tests", copy=True)
        .add_local_file(combined, "/opt/resin/tests/test_muse_glimmer_qualified.py", copy=True)
    )
    models = modal.Volume.from_name("resin-muse-glimmer-30b-models-v1").with_mount_options(read_only=True)
    references = modal.Volume.from_name("resin-benchmark-traces-v1").with_mount_options(read_only=True)
    artifacts = modal.Volume.from_name("resin-opt-" + args.job_id, create_if_missing=True)
    app = modal.App("resin-opt-" + args.job_id)

    worker = app.function(
        image=image, gpu="H100!", cpu=32, memory=262144,
        timeout=args.timeout, retries=0, max_containers=1,
        scaledown_window=2, single_use_containers=True,
        volumes={"/models/resin-muse-glimmer-30b": models, "/bench-traces": references, "/experiment": artifacts},
    )(execute)

    provenance = {"source": str(source), "lane": args.lane, "job_id": args.job_id, "config": {k: v for k, v in vars(args).items() if k not in {"source", "output"}}, "source_files": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py')) if '.venv' not in p.parts and '__pycache__' not in p.parts}}
    (args.output / "request.json").write_text(json.dumps(provenance, indent=2))
    began = time.monotonic()
    with modal.enable_output(), app.run():
        call = worker.spawn({"artifacts_volume": "resin-opt-" + args.job_id, "qualify": args.qualify, "bench": args.bench, "profile": args.profile, "repeats": args.repeats, "batches": args.batches, "extra_tests": args.extra_test, "environment": dict(value.split("=", 1) for value in args.env)})
        (args.output / "function-call.json").write_text(json.dumps({"object_id": call.object_id}))
        result = call.get()
    result["local_elapsed_s"] = time.monotonic() - began
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    if "benchmark" in result:
        (args.output / "benchmark.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result['benchmark']))
    print(json.dumps({"job": args.job_id, "output": str(args.output), "gpu_elapsed_s": result['gpu_function_elapsed_s'], "commands": [{k: r[k] for k in ['name','returncode','elapsed_s']} for r in result['commands']]}))


if __name__ == "__main__":
    main()
