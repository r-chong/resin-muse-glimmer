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

### Open items

- Speculative decoding (target-only for now).
- Image path: implemented in recipe and runner, but not yet validated on GPU.
- Sliding-window boundary and long-context parity: not yet tested.
- Optimized CUDA runner and benchmarks: deferred.
