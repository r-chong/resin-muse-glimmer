# Correctness analysis: logit discrepancies and Resin coverage

The oracle is the independent Transformers 5.19.0 run (`sdpa` attention,
BF16, H100) in `baseline/output/`. This note answers two questions:

1. What explains the Resin-versus-HF logit differences? Are they BF16 rounding,
   and do they affect low-margin predictions?
2. Which Resin abstractions does the current runner actually exercise?

## 1. Logit discrepancies

### Method

- **Resin side:** the Engine generated tokens greedily, and the
  same runner model produced teacher-forced logits on the reference tokens.
  Rows: 128 decode steps on GSM8K (82-token prompt), 16 steps each on
  `long_window` (3,829 tokens) and `long_8k` (6,896 tokens), plus the prompt
  probes at 1024, 2046–2050, 2047, 4096, 6143, and the last position.
  Run: `results/parity/20261119T203333.json`. Dumps on the Modal volume:
  `/muse-dumps/resin-128`.
- **Control:** the HF reference again, with only the attention kernel changed
  (`attn_implementation="eager"` instead of `sdpa`). Same weights, framework,
  token IDs (verified identical), GPU and dtype. Runs:
  `baseline/output/3403e274…` (GSM8K) and `baseline/output/9a66aaec…`
  (long prompts). This measures how far an implementation-only change moves
  BF16 logits on this checkpoint.
- **Analysis:** [`harness/logit_analysis.py`](../harness/logit_analysis.py).
  Results are in `results/logits/analysis_*.json`.
  - Differences are measured in BF16 units in the last place (ULPs) at each
    logit's magnitude.
  - Ties are broken by lowest token ID, as `argmax` does.
  - HF-eager decodes greedily on its own, so its rows are compared only up to
    its first token divergence.

### Results

| Prompt (rows) | Candidate | Greedy tokens match HF-sdpa | Top-1 match | Max \|Δlogit\| | ULPs p50 / p99 / p99.99 | Bitwise-equal logits | KL(HF‖cand) max |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| GSM8K (128) | **Resin** | **128/128** | 128/128 | 0.625 | 0 / 2 / 7 | 75.6% | 1.6e-2 |
| GSM8K (111) | HF eager | 110/128 (diverges at step 110) | 110/111 | 2.375 | 0 / 3 / 13 | 70.8% | 6.9e-3 |
| long_window (16 + 7 probes) | **Resin** | 16/16 | 16/16, probes 7/7 | 1.09 (probes 0.88) | 0 / 4 / 10 | 66.0% | 1.7e-3 |
| long_window (16 + 7 probes) | HF eager | 16/16 | 16/16, probes 7/7 | 1.09 (probes 3.09) | 0 / 5 / 10 | 59.8% | 3.0e-4 |
| long_8k (16 + 4 probes) | **Resin** | 16/16 | 16/16, probes 4/4 | 1.00 (probes 1.00) | 0 / 3 / 7 | 65.7% | 9.6e-4 |
| long_8k (16 + 4 probes) | HF eager | 16/16 | 16/16, probes 4/4 | 0.72 (probes 1.06) | 0 / 4 / 9 | 53.5% | 8.2e-3 |

(Max ULP values of 250–490 occur only at logits near 0, where a BF16 ULP is
tiny. Above |x| = 4 the p99 is ≤ 9 ULPs for both candidates; see
`by_magnitude` in the JSON.)

### What explains the differences

1. **Every logit is a BF16 value on both sides.** Muse computes
   `20·tanh(0.196·x/20)` in BF16 after a BF16 `lm_head`, and both HF and Resin
   return BF16 logits (verified bit patterns). A logit near 15 can only move in
   steps of 0.0625, and near 17–20 in steps of 0.125. A one-step rounding
   change at the end already shows up as 0.06–0.125.
2. **The rest is accumulated BF16 rounding through 52 layers, not a math
   error.** The observed differences are 2–9 ULPs at p99, at most about 1.1 at
   decode steps, and a median of 0 (66–76% of logits are bit-identical).
   - Swapping only HF's own attention kernel gives differences that are
     **equal or larger**: max 2.4 versus Resin's 0.63 on GSM8K, and 3.1 versus
     0.88 at the window probes.
   - Resin's attention path (explicit masks plus `repeat_interleave` GQA) is
     just a different kernel and summation order, like eager versus `sdpa`.
     Everything else (norms in FP32, the RoPE table cast, softcap order) follows
     HF exactly.
   - So the discrepancies sit **inside the implementation-noise band of the
     reference itself**. A systematic bug (wrong mask, RoPE, norm or scale)
     would show up as large, structured errors. It would also grow around the
     2,048 window; the probes at 2,046–2,050 show the same small differences.
3. **The softcap compresses differences at the top.** The top-1 logit
   differs by at most 0.25 for both candidates. Changes to the top-1/top-2
   margin are at most 0.56 for Resin and 0.63 for HF-eager.

### Do they affect low-margin predictions?

**Yes, in principle, for any BF16 implementation, HF included.** In practice
here they did not change any Resin token. The decisive evidence:

- **HF disagrees with itself at a low-margin step.** At GSM8K step 110, HF-sdpa
  has 15.375 against 15.25 (a margin of 0.125, two BF16 steps). HF-eager
  rounds both to **15.375, an exact tie**, picks the other token by lowest ID,
  and diverges for the remaining 18 steps.
- **The reference itself contains an exact tie.** At step 127, HF-sdpa has
  17.0 against 17.0 (margin 0), and greedy picks token 319 by lowest ID. Resin
  has 17.0 against 16.875 and picks the same token.
- **Resin at the low-margin steps:**

  | Step | HF margin | Resin margin | Result |
  | ---: | ---: | ---: | --- |
  | 110 | 0.125 | 0.0625 | same token |
  | 122 | 0.125 | 0.125 | same token |
  | 127 | 0.000 | 0.125 | same token |

  All 128 Engine tokens match.

Interpretation: on this checkpoint, greedy steps with a margin of a few BF16
ULPs (about 0.25 or less near logits of 15–20) are decided by rounding. Exact
token agreement there is luck, not a correctness property, for Resin and for HF
alike. The appropriate acceptance test is:
- exact tokens only while margins exceed about 2× the observed implementation
  noise (≈ 1.0 here);
- distribution-level agreement (KL ≤ about 1e-2, top-5 sets) elsewhere.

Resin passes both. The current pytest asserts exact tokens for every step. It
passed, but a future kernel change could flip a margin-0.125 step without any
bug. That assertion should be relaxed to be margin-aware before the CUDA runner
work.

## 2. Which Resin abstractions are exercised

The runner is an Engine `Runner` that declares **no `cache_layout`** in its
`RunnerConfig`. In that case the Rust `EngineCore` creates no `PagedKvResources`
(`crates/resin-core/src/engine.rs`, the `kv_resources` initialization), and the
Python binding skips `bind_resources` (`crates/resin-py/src/runner.rs`). KV
state lives in Python (`AttentionState`, per request and layer).

| Resin abstraction | Exercised? | Evidence |
| --- | --- | --- |
| Request admission, queueing, `max_running_requests` | **Yes** | Concurrency 2/4/8 tests; bench cohorts |
| `BatchPlan` construction: prefill rows, decode batching, positions, context lengths | **Yes** | Batched decode ran at full width (max batch = concurrency) |
| Chunked prefill and continuation (`prefill_chunk_size`, `continuation_prefill`) | **Yes** | 1,000- and 2,047-token chunks across the window match HF |
| Sampling parameters, greedy sampling, output budget, stop handling | **Yes** (greedy only) | Engine tokens match; `max_output_tokens` honored |
| Request lifetime and the `release` callback | **Yes** | `live_cache_count == 0` after every test |
| Benchmark harness (`resin.bench`, `scripts/bench.py`) | **Yes** | After 1 infra fix |
| **Paged KV cache pool, page tables, `cache_groups`** | **No** | No `cache_layout`, so no `PagedKvResources` |
| Sliding-window and full-attention cache groups in the pool | **No** | Window handled in Python state (trimmed to 2,047 keys) |
| Prefix cache and cache storage | **No** | Off by default, and impossible without a layout |
| KV-capacity admission, memory-pressure retraction, capacity preflight | **No** | These paths require `kv_resources`; bench recorded `cache_capacity: null` |
| CUDA graphs (`CudaRunner`, graph modes) | **No** | Eager runner; bench `capture_seconds ≈ 0` |
| Overlap scheduling (`enqueue` / completions) | **No** | No `enqueue`, so `overlap: false`, `timing: synchronized_steps` |
| Speculative decoding | **No** | Target-only (owner decision) |
| Native recipe and parser at runtime; HTTP server | **No** (recipe tested on CPU only) | Parity feeds saved token IDs |
| Media payloads and vision | **No** | Text-only scope |
| Distributed (TP/EP/DP) | **No** | One GPU |

**Bottom line:** the evidence so far validates Resin's **scheduler** (admission,
batch planning, chunked prefill, decode batching, sampling, lifecycle) and its
**benchmark harness**, driving a model-owned forward pass. It does **not** yet
validate Resin's main efficiency claims: the unified paged KV cache (including
the sliding-window cache groups), prefix reuse, graphs, and overlap. Those come
into play only with a `ManagedRunner` that declares a `CacheLayout`, as the
skill prescribes for the `CudaRunner` path. This is also why the measured decode
cost grows by about 9 ms per extra request: KV management is the runner's own
Python code, not Resin's.
