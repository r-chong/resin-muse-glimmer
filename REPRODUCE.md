# Reproducing the Muse Glimmer 30B on Resin results

Pinned inputs:
- Model: `meta-models/Muse-Glimmer-30B` @ `cec94fcfd6512f2cfe708138c32533b346ea4f69`
- Resin: `1fe19de12380cd0f5cf9c05bdd06213272217df7`, plus [`resin-patches/`](resin-patches/)
- Hardware: one Modal `H100!` (NVIDIA H100 80GB HBM3), BF16

The promoted runner is native paged decode plus stateless CUDA graph regions,
originally qualified at `6fd2dda4b0677bea4e539135f016c6b40a5c8653`.
Graph dispatch is opt-in; the primary procedure below selects it explicitly.
Native prefill and the delivery-budget experiment are not part of this runner.

## Primary: paired completion benchmark

Use a full clone, including source history and tags. Install the local
environment in section 1 and authenticate Modal. The job expects these existing
workspace volumes:

| Volume | Required contents |
|---|---|
| `resin-muse-glimmer-30b-models-v1` | Checkpoint directory `cec94fcfd6512f2cfe708138c32533b346ea4f69` |
| `resin-benchmark-traces-v1` | `muse-reference/e0b3ae6b/gsm8k_test_0001.npz`, `muse-reference/69d7fbf4/long_window.npz`, and `muse-reference/69d7fbf4/long_8k.npz` |

Weights and references are mounted read-only. If these volumes are absent,
prepare or transfer the checkpoint and golden references using sections 2–3.
For an exact replay, use the recorded reference artifacts; regeneration is a
new HF reference capture and must preserve the recorded prompts/token IDs.
The saved references were generated with Transformers 5.19.0. The pinned
Resin stack does not load that HF model class directly; qualification compares
against the saved token/logit arrays.

Prepare immutable source snapshots without calling Modal or allocating a GPU:

```bash
resin-source/.venv/bin/python harness/optimization/prepare_validated.py \
  --directory /tmp/resin-validated-replay
```

This snapshots the committed candidate `HEAD` and the native-C control
`66bb70a63470581f1ced548265e839da374836aa`, and writes a portable spec using
the new snapshot paths. Uncommitted source edits are not included. Choose a new
directory for each preparation; existing directories are never overwritten.

Launch the paid, bounded one-H100 comparison only when ready:

```bash
resin-source/.venv/bin/python harness/optimization/round2_modal.py \
  --spec /tmp/resin-validated-replay/spec.json \
  --output /tmp/resin-validated-replay/run \
  --job-id "validated-$(date -u +%Y%m%dT%H%M%SZ)" --timeout 1200
```

Both variants receive correctness qualification before measurement. The graph
candidate uses `RESIN_MUSE_DECODE_EXECUTION=graphs`; the completion adapter
calls `capture_cuda_graphs()` before timing. Supported graph batches are
1/2/4/8. There is no native-prefill or delivery-budget environment override.

The primary workload and timing are unchanged from the measured experiment:

- Synthetic tokens from seed 41, identical prompts for both variants.
- Prefill: 4,096/8,192 tokens, one request and one output token.
- Completion: 8,192 input tokens, concurrency 1/2/4/8, 256 output tokens per
  request, greedy sampling, no early EOS, prefix cache, speculation, or overlap.
- One full warmup cohort and three measured cohorts per case.
- CPU-visible request submission through the final token, including prefill;
  no HTTP or detokenization. Post-warmup windows are separate diagnostics.
- One physical H100 and one software stack for both sides. The control block
  precedes the candidate block; repetitions are not randomized alternating blocks.

The historical paired graph job qualified the candidate and reused its already
validated C control. This replay additionally requalifies C. The 1,200-second
bound adds headroom over the original 900-second allocation limit; it changes
neither measured workloads nor timing boundaries. Retries are disabled.

The output `result.json` retains hardware identity, package versions, source
hashes, qualification return codes, complete token/arrival arrays, and all
three repeats. Check all qualification and completion return codes before
using medians. Count actual committed tokens; compare the generated sequences
as well as throughput and per-request latency.

To repeat the external stock reference independently:

```bash
resin-source/.venv/bin/python harness/optimization/vllm_modal.py \
  --output /tmp/resin-stock-replay \
  --job-id "stock-$(date -u +%Y%m%dT%H%M%SZ)"
```

This uses the unmodified `vllm==0.34.0` wheel with its own stack and a separate
H100 allocation, the same checkpoint/reference volumes, and the same completion
driver. It disables image/video inputs; Resin retains unused vision weights.
Both execute the text-only workload. It is not a same-allocation software-stack
comparison. Inspect recorded packages rather than assuming all transitive
dependencies of a newly installed wheel match the historical allocation.

Published records: [first round](results/optimization/parallel-20261119/analysis.json)
and [second round](results/optimization/round2-20261119/analysis.json).
Raw archives, profiler traces, logs and NPZ dumps are intentionally excluded from
Git; [artifact-storage.json](results/optimization/artifact-storage.json) explains
their storage and the scope of checksum manifests.

## 0. Resin source

`resin-source/` already includes the promoted implementation. The original
baseline starts with pristine upstream in commit `a1736eb`, followed by the
historical fork patches. Those patches reproduce the frozen baseline, not the
later C/graph additions. Check the original patch export with:

```bash
scripts/apply_resin_patches.sh /tmp/resin-check
git -C /tmp/resin-check rev-parse HEAD^{tree}
git rev-parse baseline/frozen-20261119:resin-source  # same baseline source tree
```

## 1. Local environment (macOS or Linux; no local GPU needed)

Building Resin needs Rust ≥ 1.95. The Apple MLX package is skipped; all model
execution happens on Modal.

```bash
rustup toolchain install 1.95.0 --profile minimal
cd resin-source
RUSTUP_TOOLCHAIN=1.95.0 uv sync --no-install-package resin-models-mlx
cargo test -p resin-recipe            # recipe and template parity (CPU)
cd ..
```

Authenticate Modal once with `resin-source/.venv/bin/modal setup`. Resin
locks `modal==1.5.3`. If your workspace's default image builder is newer,
export `MODAL_IMAGE_BUILDER_VERSION=2025.06`; the harness scripts already
default to it.

```bash
export MODAL_IMAGE_BUILDER_VERSION=2025.06
```

Run the focused runner/cache/scheduler checks locally without model weights:

```bash
RESIN_MUSE_DECODE_EXECUTION=eager \
PYTHONPATH=resin-source:resin-source/models/torch \
resin-source/.venv/bin/python -m pytest \
  resin-source/models/torch/tests/test_muse_glimmer_paged.py \
  resin-source/models/torch/tests/test_muse_glimmer_graph_failures.py \
  resin-source/models/torch/tests/test_muse_glimmer_graphs.py \
  resin-source/tests/test_muse_glimmer_flashinfer.py \
  resin-source/tests/test_scheduler.py -q
```

On the CPU integration machine this produced 35 passes and three CUDA-only
skips. It supplements the recorded H100 qualification; it does not replace it.

## 2. Checkpoint (CPU worker, about 60 GB, about 4 min)

```bash
cd resin-source
MODAL_IMAGE_BUILDER_VERSION=2025.06 .venv/bin/python scripts/bench.py \
  --backend modal --model muse-glimmer-30b --download-only
cd ..
```

## 3. HF references (independent Transformers 5.19.0 runs on H100)

The short reference (`baseline/output/e0b3ae6b…`, 82-token GSM8K prompt,
128 greedy steps) came from `baseline/hf_reference.py`. A new short capture
and the long prompts (3,829 and 6,896 tokens) can be generated with:

```bash
resin-source/.venv/bin/modal run baseline/hf_reference.py --max-new-tokens 128
resin-source/.venv/bin/modal run baseline/hf_reference_cases.py --steps 16 --cases-filter long_window,long_8k
```

Upload the `.npz` files to Resin's trace volume using the directory names
each command prints. Use `<short-run>` for the short capture
and `<long-run>` for the long capture. Preserve the original reference files
for an exact replay rather than overwriting them with a different capture.

```bash
V=resin-benchmark-traces-v1
resin-source/.venv/bin/modal volume put -f $V baseline/output/<short-run>/gsm8k_test_0001.npz /muse-reference/e0b3ae6b/gsm8k_test_0001.npz
resin-source/.venv/bin/modal volume put -f $V baseline/output/<long-run>/long_window.npz /muse-reference/69d7fbf4/long_window.npz
resin-source/.venv/bin/modal volume put -f $V baseline/output/<long-run>/long_8k.npz /muse-reference/69d7fbf4/long_8k.npz
```

The saved references use the template line `Current date: 2026-11-16`. The
parity test feeds their token IDs directly, so the date does not affect it.

## 4. Standalone correctness on H100

This older helper uses eager dispatch on the current runner. The primary
paired procedure above is the qualification path for the promoted graph mode.

Single request, probes around the 2,048-token window, chunked prefill, and
concurrency 2/4/8:

```bash
resin-source/.venv/bin/python harness/muse_parity.py \
  --reference /bench-traces/muse-reference/e0b3ae6b/gsm8k_test_0001.npz \
  --reference /bench-traces/muse-reference/69d7fbf4/long_window.npz \
  --reference /bench-traces/muse-reference/69d7fbf4/long_8k.npz --steps 16
```

This writes `results/parity/<timestamp>.json`. Reference run:
`results/parity/20261119T200741.json` (5/5 passed).

## 5. Internal decode/profile benchmark (secondary)

This helper runs the current runner with eager dispatch. Its synchronized
internal decode window excludes prefill; do not compare that number directly
with the primary completion throughput or stock vLLM completion results.

```bash
harness/muse_bench.sh   # results/bench/<timestamp>.jsonl
```

This runs prefill at 4,096 and 8,192 tokens (batch 1), and decode from an
8,192-token context at batch 1/2/4/8 with 5 warmup + 250 measured steps.
Prompts are synthetic. Reference run: `results/bench/20261119T201630.jsonl`;
summary in [`reports/implementation-log.md`](reports/implementation-log.md).
That record predates the promoted implementation. Use the frozen baseline tag
to repeat the historical source rather than interpreting a current-main run as
a baseline replay.

## Cost

The paired job is capped at 1,200 seconds on one H100; the independent stock
job is capped at 900 seconds. CPU image builds, checkpoint preparation,
container startup and other resources are billed separately. Actual job times
and metered account snapshots for the measured rounds are in their protocols;
the second round used about 51 reported H100 function-minutes across parallel
jobs, including the retained timeout and recovery. These are not invoice totals.

## Preserved references

```bash
git show baseline/frozen-20261119:resin-source/models/torch/resin_models_torch/muse_glimmer/reference.py
git show backup/main-before-optimized-runner-20261120:REPRODUCE.md
```

The baseline tag resolves to `73532b9aa69f15d91b232dc47191c6a2cf077dbd`;
the pre-promotion main backup resolves to
`cbf5dc9a6b9d77f2270786f1ef7576b2ce1d7bab`. Existing experimental source
hashes remain pinned in the measurement records. The history-cleanup map
documents the equivalent revisions for any rewritten artifact/report commits.
