# Adopting Resin's cache contract for Muse Glimmer

Goal: replace the PyTorch runner's Python-owned KV (`AttentionState`, one
tensor per request per layer) with **Engine-managed pages**. The runner declares
a `CacheLayout`; the Engine allocates, retains, and evicts pages; the runner
only maps page addresses to tensors.

Started 2026-11-19 ~20:45 and time-boxed.

## Part 1: assessment (written before implementation)

### What Resin provides

These are read from the source.

- **Engine side, nothing for us to write:**
  - slab pool allocation, admission accounting across groups, window
    retention and eviction, chunk capping for windows;
  - prefix cache, copy-on-write, and leases
    (`crates/resin-core/src/engine.rs`, `docs/cache-pool.md`).
- **Contract types:**
  - `CacheLayout` / `CacheGroupLayout` (`HISTORY` vs `WINDOW` retention,
    `read_window_tokens`, `blocks_per_slab`);
  - `RunnerConfig(cache_layout=…)`, `bind_resources(ResourceBinding)`;
  - a per-batch `BatchPlan.cache_groups` with CSR block tables, window start
    indices, and per-token append addresses.
- **Python helpers, framework-neutral:**
  - `resin.backend.plans.CacheGroupView` validates addresses;
  - `resin.backend.resources.CachePoolBinding` performs the binding handshake;
  - `resin.backend.capacity.plan_cache_capacity` sizes the pool from free VRAM.
  - None of these require `CudaRunner`.

### What the model author must implement

1. **Layout declaration.** Two groups match Muse's layer split:

   | Group | Retention | Layers | Bytes per 64-token block (BF16 K+V) |
   | --- | --- | --- | ---: |
   | 0 | full `HISTORY` | 13 full-attention (NoPE) | 851,968 |
   | 1 | `WINDOW`, `read_window_tokens=2048` | 39 sliding (RoPE) | 2,555,904 |

   39 = 3 × 13, so one slab holds exactly **1 window block or 3 history blocks**
   (`blocks_per_slab = 1 / 3`) with **no padding**.
2. **The window retention rule.** A chunk of Q writes with read window W needs
   `window_tokens ≥ W + Q − 1`. With 8,192-token prefill chunks that is 10,239
   retained tokens: about 160 window blocks, or about 410 MB per live request,
   against about 82 MB of real sliding state. Smaller chunks need less memory
   but take more prefill steps. This is a memory/latency trade-off, and the
   Engine enforces it by capping chunks.
3. **The arena.** One `[P+1, slab]` BF16 tensor, with two views:
   `[P+1, 39 layers, K/V, 64, 2 heads, 128]` and `[(P+1)·3, 13 layers, …]`.
   Slab 0 is the reserved null slab.
4. **Capacity.** After loading weights, size `P` with `plan_cache_capacity`,
   reserving activation workspace.
5. **Forward-pass address mapping.** For each row and group:
   - write the new K and V to the append addresses;
   - gather the row's pages in block-table order;
   - compute key positions from `block_table_start_indices × 64`;
   - mask positions at or beyond `context + query_length` (reserved,
     unwritten slots), as well as the causal and window limits.
6. **`bind_resources`**, and dropping `release`-time KV cleanup, since the
   Engine owns lifetime.

### Can the PyTorch runner do this without a CUDA-runner rewrite?

**Yes.** Nothing in the contract requires `CudaRunner`, CUDA graphs, FlashInfer,
or custom kernels. The change is confined to:
- where K and V live (pages instead of Python tensors);
- how attention reads them (gather by page table instead of `torch.cat`).

The projection, norms, MLP, logits, sampling, and Engine integration stay
unchanged. Estimated cost is about 200–300 lines in the model package, with no
Rust and no Resin-infrastructure change expected.

### Risks

- **Address-mapping bugs** (window start index, offsets, reserved-span
  masking) would corrupt attention silently. Mitigation: the existing HF parity
  tests (window probes, chunked prefill across 2,048, concurrency 2/4/8) run
  unchanged through the paged path.
- **Window memory overhead** from the `W + Q − 1` rule at large chunks.
  Mitigation: measure capacity; keep the chunk size configurable.
- **Performance is unknown in either direction.** Per-layer page gathers replace
  `torch.cat`. No speedup is expected from paging alone without a paged
  attention kernel. Any change will be measured, not claimed.
- **Engine paths that now become live:** memory-pressure retraction and
  re-prefill, and the bench's capacity preflight. They are only exercised if
  capacity is hit; not targeted in this time-box.
- **Prefix caching** becomes *possible* (checkpoints align to the 64-token
  block size) but is not targeted in this time-box.
- **Image requests** need no change to the cache. They are out of scope.

## Part 2: implementation and results

The assessment held: the PyTorch runner adopted the contract **without a
CUDA-runner rewrite and without any Rust or Resin-infrastructure change.**
Elapsed time was about 25 minutes from assessment to a passing H100 run
(20:45 → 21:10), plus about 10 minutes of benchmarks.

### Code changes (model-specific; patches `model/0008`–`0010`)

| File | Change | Lines |
| --- | --- | ---: |
| `muse_glimmer/paged.py` (new) | Cache geometry, two-group `CacheLayout`, arena with two views, per-row page addresses from `BatchPlan.cache_groups` | 171 |
| `muse_glimmer/model.py` | `forward_paged`: write new K/V at the append addresses, gather pages by block table, mask positions past `end` | +62 |
| `muse_glimmer/reference.py` | `kv_cache="engine"` (default) or `"local"`; capacity via `plan_cache_capacity`; `bind_resources` via `CachePoolBinding`; route plans | +112 / −11 |
| `deploy/models.py` | Pass memory utilization; `kv_cache` spec field; `muse-glimmer-30b-local-kv` comparison preset | +35 |
| `models/torch/tests/test_muse_glimmer_paged.py` (new) | CPU test: a tiny Muse-shaped model through the real Engine, paged vs local vs full-context reference | 167 |
| `tests/test_muse_glimmer_hf_parity.py` | Keep `cache_layout` in the chunk override; count batch width per plan, independent of the KV path | about 25 net |

The Resin pieces used unchanged: `CacheLayout`, `CacheGroupLayout`,
`CacheRetention.WINDOW`, `RunnerConfig(cache_layout=…)`, `CacheGroupView`,
`CachePoolBinding`, `plan_cache_capacity`, and the Engine's allocator and
window retention.

### Correctness (the existing tests, now running through Engine pages)

- **CPU:** paged = local = full-context reference on prompts of 3, 17 and 30
  tokens. Window 8, chunks of 5 (crossing the window) and 64, 12 steps, 3
  concurrent requests.
  - Mutation check: shifting the window start by one block makes both cases
    fail, so the test detects addressing errors.
- **H100** (`results/parity/20261119T205303.json`, then concurrency rerun
  `3 passed`):
  - Single request: all 16 tokens match HF on 82-, 3,829- and 6,896-token
    prompts; teacher-forced argmax 16/16.
  - Chunked prefill (1,000 and 2,047 tokens) across the window: 16/16.
  - Concurrency 2, 4 and 8: every request matches 16/16, and the Engine
    decode batch reaches full width.
  - The first H100 run failed only in my test instrumentation: it counted
    calls to the local-path method. The fix is recorded below.

### Measured performance (same harness, workload, and H100 type)

`harness/muse_bench.sh`: synthetic prompts, greedy decoding, 8,192-token
decode context, 250 measured steps. All three runs used the same source
(`c75b420`) except the earlier local run (`07e6803`, local path unchanged since).

| | Local KV (20:16) | Local KV (21:03) | **Engine pages (20:59)** |
| --- | ---: | ---: | ---: |
| Prefill 4,096 TTFT | 0.484 s | 0.477 s | 0.494 s |
| Prefill 8,192 TTFT | 1.039 s | 1.040 s | 1.050 s |
| Decode b=1: tok/s/GPU (TPOT) | 8.4 (119.6 ms) | 8.3 (120.5 ms) | **11.2 (89.2 ms)** |
| Decode b=2 | 14.4 (138.5 ms) | 14.0 (143.1 ms) | **20.0 (100.1 ms)** |
| Decode b=4 | 25.0 (160.0 ms) | 26.1 (153.4 ms) | **30.3 (131.9 ms)** |
| Decode b=8 | 43.9 (182.0 ms) | 44.3 (180.6 ms) | **53.6 (149.3 ms)** |

Raw data: `results/bench/20261119T201630.jsonl`,
`local-20261119T210305.jsonl`, `paged-20261119T205919.jsonl`.

- **Decode:** throughput is 21–43% higher with Engine pages (time per token
  down 14–28%) at every batch size. The two local runs agree within about 3%,
  so this difference is well outside the observed run-to-run spread. It is
  one paged run, and the batch-4 p90 is noisy (163 ms against a p50 of
  122 ms).
- **Prefill:** unchanged within 1–3%.
- **Cause: not profiled.** A plausible explanation is that the local path
  re-concatenates (`torch.cat`) each layer's whole KV history every step and
  slices the window, while the paged path writes one token and gathers pages.
  This is untested. This is still a correctness-first runner (eager, padded
  SDPA, `repeat_interleave`), so it remains far from the roughly 17 ms
  weight-bandwidth floor.

### Memory: what the contract costs

- **Pool:** sized by `plan_cache_capacity` after loading weights: 24.4 GB
  free, 0.90 utilization, 8 GiB activation reserve. That gives 2,870 slabs ×
  2.56 MB = **7.34 GB**.
- **Per request at an 8,448-token peak:** 176 slabs (132 window plus 44
  history), about **450 MB**, so about 16 such requests fit. The old Python
  path held about 194 MB per request: 112 MB full layers plus 82 MB for the
  trimmed sliding window.
- **Why:** the `window_tokens ≥ W + Q − 1` rule with 8,192-token chunks means
  10,239 retained window tokens, more than this whole context. So at 8K the
  window saves nothing; it saves only beyond about 10K tokens, or with smaller
  prefill chunks (a TTFT/memory trade-off, not tuned).
- **In exchange,** capacity is now explicit and enforced. The Engine admits by
  slabs, and the bench preflight reports requirements before running. The
  Python path had no admission control.

### Interventions, bugs and blockers in this phase

| Kind | Item |
| --- | --- |
| Owner | Corrected two overstatements in the parity report (BF16 attribution, margin threshold) before this work began |
| Owner | Set scope and time box: assess first, then the smallest integration; keep tests; measure before claiming |
| Model author (expected) | Layout, slab packing, arena views, address mapping, capacity, binding: everything listed in part 1 |
| Model-side bug, caught on CPU | Arena dtype was hard-coded to BF16; it now follows the model (needed for the FP32 CPU test) |
| Contract friction | `CachePoolBinding` refuses a second pool, but the tests and bench build one Engine per case over one runner. The runner rebinds its arena per Engine; one live Engine at a time is documented. |
| Contract friction | Overriding `RunnerConfig` in tests must carry `cache_layout`, or the Engine and runner disagree |
| Test instrumentation | Batching assertion counted a local-path method; it now counts plan widths |
| Agent process | Bench outputs landed under `resin-source/` because of relative paths with the wrapper's `cd`. Files moved; wrapper fixed |
| Resin infra | **None needed.** No `bench.py`, Engine or binding changes |

### Still not exercised

- Prefix caching, which is now possible: `Engine(prefix_cache=True)`.
  Checkpoints align to 64-token blocks.
- Memory-pressure retraction and re-prefill.
- Overlap scheduling.
- CUDA graphs.
- Paged attention kernels (FlashInfer `window_left`).
- Images.

Each is a separate experiment. None was attempted in this time box.
