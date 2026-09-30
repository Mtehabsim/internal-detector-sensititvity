#!/usr/bin/env bash
# Prepare the MTK release this artifact audits: a pinned clone and the working copy the harness imports from.
#
#   bash scripts/setup_release.sh            # clone if needed; reuse an existing working copy that matches the release
#   bash scripts/setup_release.sh --reset    # replace the working copy (deletes the feature caches kept inside it)
#
# Running it again is safe: a working copy whose release files are all byte-identical to the pinned clone is kept
# with everything the runs added to it (feature caches, model links); one that differs stops the script.
#
# Optional environment:
#   MTK_RELEASE_REPO     where to clone github.com/Rookie143/mtk      (default: third_party/mtk)
#   MTK_RELEASE_SCRATCH  working copy and caches                      (default: scratch)
#   MTK_MODEL_DIR        model weights, GPU stages only (llama2, llama3, mistral_7b, vicuna-7b-v1_5)
#   GRADSAFE_DIR         where to clone github.com/xyq7/GradSafe, GradSafe stages only (default: third_party/GradSafe)
#
# If you set MTK_RELEASE_REPO, also set MTK_RELEASE_CHECKOUT="$MTK_RELEASE_REPO/llm" for the Python scripts.
set -euo pipefail
RESET=0
for arg in "$@"; do
  case "$arg" in
    --reset) RESET=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MTK_COMMIT=c5e2f18d913bd18b09b9357159543487beb2ce75
GS_COMMIT=2a8b6edd213ed8931b300f7270887227a8e880be
REPO=${MTK_RELEASE_REPO:-$ROOT/third_party/mtk}
SCRATCH=${MTK_RELEASE_SCRATCH:-$ROOT/scratch}

# 1. the authors' release, pinned (the harness refuses to run on any other commit or a modified checkout)
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/Rookie143/mtk.git "$REPO"
fi
git -C "$REPO" checkout --quiet "$MTK_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$MTK_COMMIT"

# 2. the working copy the harness imports (mtkaudit.release.verify_copy() checks it file by file against the clone)
COPY="$SCRATCH/release/llm"
copy_matches_release() {   # every release file present and byte-identical; files the runs added are allowed
  git -C "$REPO" ls-files -z llm | while IFS= read -r -d '' f; do
    cmp -s "$REPO/$f" "$SCRATCH/release/$f" || { echo "  differs from the release: ${f#llm/}" >&2; exit 1; }
  done
}
if [ -d "$COPY" ] && [ "$RESET" = 0 ]; then
  if copy_matches_release; then
    echo "reusing the working copy $COPY (kept with its caches; --reset replaces it)"
  else
    echo "the working copy $COPY does not match the pinned release; rerun with --reset to replace it" \
         "(this deletes the feature caches inside it)" >&2
    exit 1
  fi
else
  if [ -d "$COPY" ]; then
    echo "--reset: replacing $COPY"
    rm -rf "$COPY"
  fi
  mkdir -p "$SCRATCH/release"
  cp -a "$REPO/llm" "$COPY"
fi
# The one data substitution: Mistral's runner lists datasets/mistral_test/ijp_0.json, which the release ships as
# ijp_1.json.
if [ ! -e "$SCRATCH/release/llm/datasets/mistral_test/ijp_0.json" ]; then
  ln -s ijp_1.json "$SCRATCH/release/llm/datasets/mistral_test/ijp_0.json"
fi

# 3. model weights, for the GPU stages only
if [ -n "${MTK_MODEL_DIR:-}" ]; then
  mkdir -p "$SCRATCH/release/llm/model"
  for m in llama2 llama3 mistral_7b vicuna-7b-v1_5; do
    ln -sfn "$MTK_MODEL_DIR/$m" "$SCRATCH/release/llm/model/$m"
  done
fi

# 4. GradSafe's official code, for the GradSafe stages only
if [ -n "${WITH_GRADSAFE:-}" ] || [ -n "${GRADSAFE_DIR:-}" ]; then
  GS=${GRADSAFE_DIR:-$ROOT/third_party/GradSafe}
  [ -d "$GS/.git" ] || git clone https://github.com/xyq7/GradSafe.git "$GS"
  git -C "$GS" checkout --quiet "$GS_COMMIT"
fi

echo "release $MTK_COMMIT ready: clone $REPO, working copy $SCRATCH/release/llm"
