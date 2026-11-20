# Reproducing the Muse Glimmer 30B on Resin results

Pinned inputs:
- Model: `meta-models/Muse-Glimmer-30B` @ `cec94fcfd6512f2cfe708138c32533b346ea4f69`
- Resin: `1fe19de12380cd0f5cf9c05bdd06213272217df7`, plus [`resin-patches/`](resin-patches/)
- Hardware: one Modal `H100!` (NVIDIA H100 80GB HBM3), BF16

## 0. Resin source

`resin-source/` in this repo is already the patched tree: pristine upstream in
commit `a1736eb`, then the fork commits. To check that the patches alone
reproduce it:

```bash
scripts/apply_resin_patches.sh /tmp/resin-check
git -C /tmp/resin-check rev-parse HEAD^{tree}   # equals: git rev-parse HEAD:resin-source
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

## 2. Checkpoint (CPU worker, about 60 GB, about 4 min)

```bash
cd resin-source
MODAL_IMAGE_BUILDER_VERSION=2025.06 .venv/bin/python scripts/bench.py \
  --backend modal --model muse-glimmer-30b --download-only
cd ..
```

## 3. HF references (independent Transformers 5.19.0 runs on H100)

The short reference (`baseline/output/e0b3ae6b…`, 82-token GSM8K prompt,
128 greedy steps) came from `baseline/hf_reference.py`. The long prompts
(3,829 and 6,896 tokens) come from:

```bash
.venv/bin/modal run baseline/hf_reference_cases.py --steps 16 --cases-filter long_window,long_8k
```

Upload the `.npz` files to Resin's trace volume. Replace `<run>` with the
directory name the command prints.

```bash
V=resin-benchmark-traces-v1
resin-source/.venv/bin/modal volume put -f $V baseline/output/e0b3ae6b507041dfb2347ff01b9db3a3/gsm8k_test_0001.npz /muse-reference/e0b3ae6b/gsm8k_test_0001.npz
resin-source/.venv/bin/modal volume put -f $V baseline/output/<run>/long_window.npz /muse-reference/<run8>/long_window.npz
resin-source/.venv/bin/modal volume put -f $V baseline/output/<run>/long_8k.npz /muse-reference/<run8>/long_8k.npz
```

The saved references use the template line `Current date: 2026-11-16`. The
parity test feeds their token IDs directly, so the date does not affect it.

## 4. Correctness on H100

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

## 5. Performance on H100

```bash
harness/muse_bench.sh   # results/bench/<timestamp>.jsonl
```

This runs prefill at 4,096 and 8,192 tokens (batch 1), and decode from an
8,192-token context at batch 1/2/4/8 with 5 warmup + 250 measured steps.
Prompts are synthetic. Reference run: `results/bench/20261119T201630.jsonl`;
summary in [`reports/implementation-log.md`](reports/implementation-log.md).

## Cost

Approximate GPU time for the full sequence: about 20 H100-minutes, plus CPU
time for image builds and the checkpoint copy.
