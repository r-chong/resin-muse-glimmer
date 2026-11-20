#!/usr/bin/env bash
# Recreate the patched Resin tree from the pinned upstream revision, to check
# that the patch series applies cleanly. (resin-source/ in this repo is already
# the vendored, patched tree.)
#   scripts/apply_resin_patches.sh <new-target-dir>
# RESIN_REPO overrides the upstream URL (a local clone of ekzhang/resin works).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:?usage: $0 <new-target-dir>}"
REPO="${RESIN_REPO:-https://github.com/ekzhang/resin.git}"
BASE=1fe19de12380cd0f5cf9c05bdd06213272217df7
if [ -e "$TARGET" ]; then
  echo "refusing to overwrite existing $TARGET" >&2
  exit 1
fi
git clone --quiet "$REPO" "$TARGET"
git -C "$TARGET" checkout --quiet -B muse-glimmer-torch "$BASE"
# Resin infrastructure fixes first, then the model-specific integration.
git -C "$TARGET" -c user.name=patches -c user.email=patches@localhost \
  am --quiet --committer-date-is-author-date \
  "$ROOT"/resin-patches/infra/*.patch "$ROOT"/resin-patches/model/*.patch
git -C "$TARGET" log --oneline "$BASE"..HEAD
