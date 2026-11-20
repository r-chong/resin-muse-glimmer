# Muse Glimmer 30B on Resin: implementation log

Evidence for the take-home research questions: how much work it takes to get
a correct, servable Resin engine; who intervened, and why; what broke; and
where performance would come from. All times are local (EST) on 2026-11-19.
They come from file timestamps, commit times, and run artifacts in this repo.
The agent was Claude Code (Opus 5.5). "Owner" is the experiment owner.

## Timeline

| Time | Event | Evidence |
| --- | --- | --- |
| ~18:45 | Session start: survey the repo, Resin `add-resin-model` skill, HF modeling code, checkpoint config | `MUSE_MODAL_PREPARATION.md` mtime 18:42 |
| 18:56 | Tokenizer, chat template and dependency pins recorded | [`pins/README.md`](../pins/README.md) |
| ~18:58 | Owner architecture decisions: torch reference runner first, native vision, target-only, native Rust recipe | [`DECISIONS.md`](../DECISIONS.md) |
| 19:00–19:19 | Recipe, bindings, torch runner (text + vision), registry entry, parity test written; recipe tests pass locally | fork commits authored 19:19 |
| ~19:20 | Owner narrows scope to text-only parity on H100 | DECISIONS |
| 19:22–19:48 | Packaging, local toolchain fixes, Modal image build, 59.6 GB checkpoint copy (248 s) | download log |
| 19:49 | First H100 run: **Resin infra blocker** (torch package missing from image) | `results/parity/20261119T194925.json` |
| 19:56 | **First passing parity**: 82-token prefill + 16 decode steps through the Engine match HF | `results/parity/20261119T195604.json` |
| 20:02–20:04 | HF references for prompts beyond the window (3,829 and 6,896 tokens) | `baseline/output/69d7fbf4…` |
| 20:07 | **Window, chunked-prefill and concurrency parity pass** (5/5) | `results/parity/20261119T200741.json` |
| 20:12 | Benchmark blocked by a **second Resin infra bug** (`bench.py`) | infra patch 0002 |
| 20:16–20:21 | Benchmark completes after a third Resin infra fix (graph capture assumed) | `results/bench/20261119T201630.jsonl` |

From session start to the first correct Engine output took **about 71 minutes**
of wall clock. Roughly 30 of those minutes were waiting on Modal: image build
and checkpoint copy. Writing the model-specific code took about 20 minutes. The
runner passed parity on its first execution once the infrastructure issue was
fixed. No model-math bug was found.

## Size of the model-specific work

From the patch series (added lines; see
[`resin-patches/README.md`](../resin-patches/README.md)):

- **Text-only path (minimum servable engine):**
  - runner: `config.py` 176, `model.py` 448, `reference.py` 308
  - recipe: Rust ATEM recipe 616 (it includes tests and long tool-prompt constants)
  - registry entry: 59
  - Python recipe and media bindings: about 160
  - Roughly 1,750 lines in total, plus tests.
- **Image path (written, not validated on GPU):** `vision.py` 274,
  `media/muse.rs` 283, and the `MuseGlimmerMedia` binding.
- **Resin infrastructure fixes:** 2 patches (Modal image: 8 lines; `bench.py`:
  about 10 lines). Neither is Muse-specific; any eager or torch-backend preset
  would hit both.

## Owner interventions

| # | Intervention | Effect |
| --- | --- | --- |
| 1 | Four architecture decisions (runner, vision, speculation, recipe) | Required by `AGENTS.md`; the agent asked and did not choose |
| 2 | "We're not running on MLX; we run on Modal H100" | Agent had tried a full local `uv sync`; switched to local CPU/Rust only |
| 3 | Narrowed scope: smallest text-only runner, existing reference, report the first blocker | Stopped the expanding image/long-context capture before any GPU spend |
| 4 | Keep a decision log; semantic commits | `DECISIONS.md`; Conventional Commits throughout |
| 5 | Work belongs in `resin-benchmarks`; a baseline exists | Resin changes exported as patches, later vendored |
| 6 | Ignore the separate vibe-coded engine | Out of scope for this log |
| 7 | Keep the fork; no library-mode migration | Layout fixed |
| 8 | Prioritized window, batching and throughput evidence; separate infra from model | This log, infra/model patch split |
| 9 | Vendor `resin-source/`, then push | Done (`a1736eb` = pristine upstream) |

## Bugs and blockers

**Resin infrastructure (would affect any similar model):**

1. The Modal image shipped only `resin_models_torch/__init__.py`, so the worker
   raised `ModuleNotFoundError`. Fixed by infra patch 0001.
2. `bench.py` assumed every non-MLX runner is a `CudaRunner`. It compared an
   unset `RunnerConfig.max_batch_tokens` (allowed by the contract) with an int,
   then called `capture_cuda_graphs()` unconditionally. This took two H100
   attempts to surface, each failing before measurement. Fixed by infra patch
   0002.
3. Resin's locked `modal==1.5.3` rejects the workspace's 2026.10 image builder.
   Worked around with `MODAL_IMAGE_BUILDER_VERSION=2025.06`; no patch.
4. Resin's locked Transformers (5.14.1) predates Muse (added in 5.15.0). The
   runner was designed to read configs directly, so no lock bump was needed.
5. Building Resin needs Rust ≥ 1.95 (`atomic_try_update`). The default dev
   group builds the Apple MLX package, which needs Apple's Metal compiler.

**Model integration (all caught by tests before GPU):**

6. Misread the template's tool-result branch. Caught by the Jinja-rendered HF
   fixture test.
7. HF `smart_resize` breaks aspect-ratio ties by Python `set` order. Reproduced
   CPython's tuple hash and set probing; 514 generated cases match.
8. A hand-computed test expectation was wrong (2072 vs 2044). Caught by
   comparing with Python.

**Agent process errors (no data lost):** zsh glob/word-splitting surprises, an
invalid `cherry-pick -q`, and `commit -C` pointing at the wrong repo. Each was
caught right away; tree hashes were verified after the history rewrites.

## Correctness evidence

See [`DECISIONS.md`](../DECISIONS.md). Summary:
- Greedy tokens match HF on prompts of 82, 3,829 and 6,896 tokens (16 steps each).
- Logits at the probed positions around 2,048 have the same argmax as HF;
  max abs logit difference ≤ 1.0 after softcapping.
- Chunked prefill across the window matches.
- Concurrency 2, 4 and 8 through the batched decode path matches.

## Performance (PyTorch reference runner, 1× H100 80GB HBM3, BF16)

Run: `harness/muse_bench.sh` → Resin `scripts/bench.py`, repo commit `07e6803`.
- Prompts are synthetic token controls, not a chat corpus. Sampling is greedy.
- Graphs and overlap are off: the runner has neither.
- Prefill uses 1 warmup and 3 measured cohorts; the times are complete-prompt latency.
- Decode starts from an 8,192-token context and runs 5 warmup + 250 measured
  steps, so each request generates about 256 tokens (context 8,197 → 8,447).
- The metadata records `dirty: true`. That is from uncommitted owner files
  outside `resin-source/`; `resin-source/` itself was clean.

| Prefill (batch 1) | Time to first token | Input tok/s/GPU |
| --- | ---: | ---: |
| 4,096 tokens | 0.484 s | 8,455 |
| 8,192 tokens | 1.039 s | 7,885 |

| Decode, 8K context | Output tok/s/GPU | Per-user tok/s | TPOT (ms) | Step p50 / p90 (ms) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 8.4 | 8.36 | 119.6 | 118.6 / 124.3 |
| 2 | 14.4 | 7.22 | 138.5 | 135.2 / 141.9 |
| 4 | 25.0 | 6.25 | 160.0 | 153.5 / 175.7 |
| 8 | 43.9 | 5.49 | 182.0 | 180.6 / 188.9 |

Rough roofline, unprofiled (a first estimate, not a measurement):

- **Decode is far from the bandwidth floor.** One step streams about 56 GB of
  BF16 weights, about 17 ms at H100's ~3.35 TB/s. Batch-1 TPOT is 120 ms,
  about 7× over that floor. Each added request costs about 9 ms per step.
  That fits per-request host and memory overhead: `torch.cat` KV growth,
  `repeat_interleave` GQA expansion (about 1.7 GB of expanded full-attention
  K/V per request per step at 8K), padding and masks, and eager launches.
  These are items 1–3 below. A Perfetto trace would confirm the split; none
  was captured.
- **Prefill is reasonably efficient for eager code.** 8,192 tokens is about
  4.4×10¹⁴ dense FLOPs, so 1.04 s is about 420 TFLOP/s, roughly 40% of H100
  dense BF16 peak. Most of the time is in cuBLAS GEMMs.
- **No external comparison yet.** vLLM, SGLang and the from-scratch engine
  were not run under this workload, so these numbers say nothing about Resin
  versus alternatives. They characterize only the correctness-first runner.

## Optimization opportunities (not done; no premature optimization)

The runner is deliberately eager and plain-PyTorch. Expected costs, in rough
order of impact:

1. **No CUDA graphs.** 52 layers × about 15 kernel launches per decode step,
   all from Python.
2. **KV growth by `torch.cat` each step, plus `repeat_interleave` GQA
   expansion.** Copies and expands every retained key and value per layer
   per step (16× for GQA 32/2).
3. **Padded batched decode with materialized boolean masks.** Pads each request
   to the longest history; the SDPA mask path cannot use FlashAttention.
4. **Engine paging unused.** Request state lives in Python, so no prefix
   caching or cache-pool admission control.
5. **Prefill runs one request at a time,** with dense
   `[L, L]` masks for the sliding layers.
6. **Unfused small ops.** FP32 norms (four per layer plus QK norms), RoPE,
   sigmoid gate, softcap.

The path Resin's skill prescribes is a `CudaRunner` with FlashInfer paged
attention: `window_left = 2047` for sliding layers, no RoPE on full layers. It
would add graph-captured decode and Engine-owned pages; Gemma 4's runner has the
same full/sliding split. That is an owner decision ("then a separate CUDA
runner later").
