#!/usr/bin/env bash
# Benchmark the Muse Glimmer torch runner on one Modal H100 with Resin's own
# scripts/bench.py. Workload from the experiment brief:
#   prefill: 4,096 and 8,192-token prompts, batch 1, time to first token
#   decode:  8,192-token context, concurrency 1/2/4/8, ~256 new tokens/request
#            (5 warmup + 250 measured decode steps)
# Prompts are synthetic token controls (no chat corpus). Writes JSONL under
# results/bench/. MUSE_MODEL=muse-glimmer-30b-local-kv selects request-owned KV.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/results/bench/$(date +%Y%m%dT%H%M%S).jsonl}"
mkdir -p "$(dirname "$OUT")"
cd "$ROOT/resin-source"
# Resin locks modal 1.5.3; the workspace default image builder is too new for it.
export MODAL_IMAGE_BUILDER_VERSION="${MODAL_IMAGE_BUILDER_VERSION:-2025.06}"
.venv/bin/python scripts/bench.py --backend modal --model "${MUSE_MODEL:-muse-glimmer-30b}" \
  --prompt-source synthetic \
  --phases prefill,decode \
  --prefill-context-lengths 4096,8192 --prefill-batch-sizes 1 \
  --decode-context-lengths 8192 --decode-batch-sizes 1,2,4,8 \
  --warmup-steps 5 --measure-steps 250 \
  --prefill-graph disabled --decode-graph disabled \
  --output "$OUT" "${@:2}"
echo "$OUT"
