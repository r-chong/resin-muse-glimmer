# Decision log

Running log for the Muse Glimmer 30B on Resin investigation. **Owner** marks
decisions made by the experiment owner (architecture, scope, budget).
**Agent** marks routine, reversible execution choices made within those
decisions. Newest entries are at the bottom.

## 2026-11-19

### Constraints (owner)

- Model `meta-models/Muse-Glimmer-30B` @ `cec94fcfd6512f2cfe708138c32533b346ea4f69`.
- Resin @ `1fe19de12380cd0f5cf9c05bdd06213272217df7`.
- BF16 weights and KV, unquantized. One H100 80GB HBM3 on Modal. Spending cap is $200.
- Text and image inputs, text output.
- Target workload: prefill at 4,096 and 8,192 tokens (batch 1); decode from an
  8,192-token context, 256 new tokens, concurrency 1/2/4/8; image check with one
  fixed image, recording its token count.
- Record tokenizer, chat template and dependency versions. Done in [`pins/README.md`](pins/README.md).

### Architecture (owner)

| Question | Decision |
| --- | --- |
| Runner implementation | **Torch reference first**: plain-PyTorch runner in `resin_models_torch`, then a separate CUDA runner later |
| Vision tower | **Native** reimplementation (no HF model code at runtime) |
| Speculative decoding | **Target-only**; speculation recorded as an open item |
| Chat format recipe | **Native Rust recipe** in `crates/resin-recipe` with Python bindings |
| Execution platform | **Modal H100**; MLX/Apple is out of scope |

### Scope change (owner)

- Stop expanding multimodal coverage and reference fixtures. Priority: the
  smallest working **text-only** runner on H100, with one prefill and several
  decode steps through Resin, compared to the **existing** HF reference.
  Report the first concrete blocker. Defer images, long context, optimization
  and benchmarks.
- Keep this log, and commit in small semantic (Conventional Commits) units.

### Execution choices (agent)

- **Runner shape.** Follows Resin's `Lfm2TorchRunner` pattern: an Engine `Runner`
  with request-owned attention state (not Engine pages), SDPA with explicit
  causal/sliding masks, and no CUDA graphs. Sliding layers keep only the last
  2,047 keys. Uniform one-token decode steps run as one batched forward; prefill
  runs one request at a time.
- **Numerics.** Follow the Transformers reference: FP32 norms cast back to BF16,
  RoPE tables cast to BF16, and softcapped logits in BF16
  (`20·tanh(0.196·x/20)`).
- **Registry.** `muse-glimmer-30b` uses `backend="torch"` and `gpu="H100!"` (no
  silent upgrade to H200). Its own Modal volume is
  `resin-muse-glimmer-30b-models-v1`.
- **Transformers lock unchanged (5.14.1).** It cannot load Muse (added in
  v5.15.0), so the Resin path reads `config.json` and `processor_config.json`
  directly and uses `tokenizers` for the recipe. No Transformers code runs
  inside Resin.
- **Recipe date line.** The template stamps `strftime_now`. The recipe takes
  `current_date`, and the Python factory defaults it to today, as Transformers
  does. Parity fixtures pass a fixed date.
- **Image resize tie-break.** HF `smart_resize` breaks equal-aspect ties by Python
  `set` iteration order. The Rust port reproduces CPython's tuple hash and set
  probing. 514 generated cases match, including square-image ties.
- **Local toolchain.** Resin core needs Rust ≥ 1.95 (`atomic_try_update`).
  Installed `1.95.0` as an extra rustup toolchain; the default toolchain is
  unchanged. Rustup self-updated 1.29.0 → 1.29.3. Local `uv sync` skips
  `resin-models-mlx`, which needs Apple's Metal compiler.
- **Stopped** the extra HF capture (`baseline/hf_reference_cases.py`) after the
  scope change. It was cancelled during its image build, before any GPU was allocated.
  The script stays in the repo, unused.
- **Parity reference.** `baseline/output/e0b3ae6b507041dfb2347ff01b9db3a3`
  (GSM8K prompt, 82 tokens, 128 greedy steps, HF 5.19.0 / torch 2.9.1 on H100).
  The first 16 steps are compared.

- **Where work lives.** `resin-benchmarks` (this repo) is the repo of record.
  Resin changes are made on branch `muse-glimmer-torch` in the gitignored
  `resin-source/` checkout. They are exported to [`resin-patches/`](resin-patches/) as a
  `git format-patch` series on `1fe19de`, one patch per semantic commit. The
  HF baseline in `baseline/` is the comparison target.

- **Modal image builder.** The active Modal profile is `resin`; its default image
  builder is 2026.10. Resin's locked `modal==1.5.3` rejects that version, so
  Resin launches set `MODAL_IMAGE_BUILDER_VERSION=2025.06`, its newest supported
  version. Resin's lock is unchanged.
- **Parity harness.** [`harness/muse_parity.py`](harness/muse_parity.py) runs
  Resin's own `qualify` Modal function on `H100!` with the parity test. The raw
  result goes to `results/parity/`.

- **Reference prompt date.** The saved HF references (`baseline/output/*`) were
  rendered with the wall-clock `Current date: 2026-11-16`. The Resin parity test
  feeds the saved input IDs, so it is unaffected. Any recipe-to-reference
  token check must render with `current_date="2026-11-16"`. The same issue
  failed the from-scratch baseline (`vibe-muse-glimmer-engine`) run at
  19:29:32: it took the first date in the prompt (the knowledge cutoff,
  `2026-01-04`). Its fix `bab6b42` was verified locally to recover `2026-11-16`
  from that exact prompt. That fixed version has not yet been rerun on GPU.

- **Baseline rerun result (2026-11-19, run `20261120T003007Z-502c4891`, commit
  `bab6b42`).** It ran on an H100 80GB HBM3. Prompt IDs match the fixture (82
  tokens, date 2026-11-16). Tokens match for all 128 greedy steps, and the
  logits are bitwise equal (max abs error 0.0). The function body took 102 s,
  90 s of it loading weights. Peak allocation was 59.6 GB. Scope of this
  evidence: the baseline calls Transformers' own decoder layers, embedding,
  rotary, norm and `lm_head`. It replaces only the KV cache, masks and decode
  loop, so bitwise equality is expected. The run validates the cache, mask
  and loop logic, not independent forward math. With 82 + 128 = 210 positions,
  it never crosses the 2,048-token sliding window. I reused the run already in
  flight instead of launching a duplicate.

- **First Resin blocker (fixed).** Resin's Modal image copies only
  `resin_models_torch/__init__.py`, so the worker raised
  `ModuleNotFoundError: resin_models_torch.muse_glimmer`. Fix: patch 0006 ships
  the full torch package, placed after the Rust-build layer so the cached
  layers are reused.
- **Resin text parity passed (2026-11-19, `results/parity/20261119T195604.json`,
  resin-source `cdaa69f8`, clean).** H100, BF16, PyTorch reference runner, one
  prefill (82 tokens) plus 16 greedy decode steps through the Resin Engine.
  All 16 tokens match the HF reference `e0b3…`, and teacher-forced argmax
  matches 16/16. Max abs logit difference is 0.25–0.53 per step (mean
  0.010–0.079). The reference top-1 margin is ≥ 12.3 on every step. These
  are not bitwise equal: the runner reimplements the math rather than calling
  HF layers. Not yet covered: the sliding-window boundary (only 98
  positions), batch > 1, and steps beyond 16.

### Layout (owner)

- **Vendor the fork (owner).** `resin-source/` is no longer gitignored. It is
  tracked in this repo: pristine upstream `1fe19de` in `a1736eb`, followed by
  the infra and model commits replayed with their original messages. The fork's
  former nested `.git` is kept as the ignored backup `.resin-fork.git/`.
  Patches are generated with `git format-patch --relative=resin-source`.
- **Keep the fork.** Resin changes stay on the `muse-glimmer-torch` branch of
  `resin-source` (base `1fe19de`), exported to `resin-patches/`. The native Rust
  recipe and the shared registry/bench/serve tooling stay as they are.
  `resin-benchmarks` holds docs, pins, the harness and results.

- **Sliding-window and batching parity passed (2026-11-19,
  `results/parity/20261119T200741.json`, 5/5 tests, 96 s on H100).** New HF
  references `baseline/output/69d7fbf4…` (transformers 5.19.0, sdpa, H100):
  `long_window` (3,829 tokens) and `long_8k`, which came out at **6,896
  tokens, not 8K**.
  - Single request (Engine, 16 greedy steps): tokens match 16/16 on all three
    prompts. Teacher-forced argmax matches 16/16.
  - Probe positions 1024, 2046–2050, 3828 / 2047, 4096, 6143, 6895: argmax
    matches 11/11. Max abs logit difference is ≤ 1.0. The smallest
    reference margin at a probe is 5.69, and on decode steps it is 1.75.
  - Chunked Engine prefill (1,000- and 2,047-token chunks that straddle the
    window): 16/16 tokens match on both long prompts.
  - Concurrency 2/4/8 with mixed prompt lengths: every request matches its
    reference 16/16. The batched decode path ran at full width (max batch =
    concurrency).
  - Limits: 16 decode steps, greedy only, BF16 batch-shape numerics not
    stressed beyond these cases, no prompt of 8,192 or more tokens.

- **Benchmark (2026-11-19, `results/bench/20261119T201630.jsonl`).** Run through
  `harness/muse_bench.sh`, which calls Resin's `bench.py` with the brief's
  workload. Prompts are **synthetic token controls** (agent choice: content
  does not change dense-model compute for fixed lengths; no corpus download).
  Use a chat corpus if you want it. Results are in
  [`reports/implementation-log.md`](reports/implementation-log.md#performance-pytorch-reference-runner-1-h100-80gb-hbm3-bf16).
  Headline: TTFT 0.48 s at 4K and 1.04 s at 8K. Decode is 8.4 output tok/s/GPU
  at batch 1 (TPOT 120 ms) and 43.9 at batch 8 (per-user 5.5 tok/s).
- **Infra patch 0002** (`bench.py`) landed after the model commits; see
  `resin-patches/README.md`.

- **Logit discrepancy analysis (2026-11-19).** See
  [`reports/correctness-analysis.md`](reports/correctness-analysis.md).
  - Resin-vs-HF differences are BF16 implementation noise. They are no
    larger than HF's own eager-vs-sdpa control, which itself diverges at a
    margin-0.125 step.
  - Resin matches all 128 greedy GSM8K tokens, including an exact tie in the
    reference.
  - Recommendation: make exact-token assertions margin-aware.
- **Resin coverage.** The runner declares no `CacheLayout`, so the scheduler
  and bench paths are exercised. The paged KV pool, prefix cache, graphs and
  overlap are **not**.

### Open items

- Speculative decoding (target-only for now).
- Image path: implemented in recipe and runner, but not yet validated on GPU.
- Sliding-window boundary and long-context parity: not yet tested.
- Optimized CUDA runner: deferred (owner decision). No external baseline
  (vLLM/SGLang/from-scratch) has been measured under the same workload.
