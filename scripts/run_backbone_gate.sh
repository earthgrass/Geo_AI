#!/usr/bin/env bash
#
# Backbone Gate — GPU environment (generate only; execute manually on the GPU host)
#
# Order: I0 Persistence (validation) -> I1 PlainConvLSTM (train) ->
#        I2 ResConvLSTM (validate-only REUSE, NEVER train) -> B1 TrajGRU (train)
#
# Constraints:
#   - TEST stays SEALED: no test option exists anywhere in this script.
#   - Any task failure stops the run (set -euo pipefail).
#   - Independent output dirs + timestamped logs + start/end timestamps.
#   - Unified seed=42 / batch=4 / 20 epochs / AMP=auto are baked into the
#     frozen YAMLs (formal CLI overrides are prohibited by the spec).
#   - E2/I2 is resolved through the artifact verifier; if no valid official
#     checkpoint exists, the gate BLOCKS instead of retraining a replacement.
#
# Usage (from the repo root):
#   bash scripts/run_backbone_gate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

E2_MANIFEST="results/E2_resconvlstm_seed42/manifest.json"

OUT_BASE="outputs/backbone_gate"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/backbone_gate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Backbone Gate started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  seed=42 batch=4 epochs=20 AMP=auto (frozen YAMLs)"
echo "  output base=$OUT_BASE"
echo "  log=$LOG"
echo "  TEST STATUS: SEALED (no --allow-test-eval, no test path)"
echo "============================================================"

run_exp() {
  local name="$1" config="$2" out="$3"
  echo ""
  echo ">>> [$name] start $(date '+%H:%M:%S')"
  python scripts/run_experiment.py --config "$config" --out "$out"
  echo "<<< [$name] done $(date '+%H:%M:%S')"
}

# ---- I0 = E0 Persistence: non-trainable, validation-only ----
run_exp "I0_persistence" "configs/experiments/E0_persistence.yaml" \
        "$OUT_BASE/I0_persistence"

# ---- I1 = E1 PlainConvLSTM: train ----
run_exp "I1_plain_convlstm" "configs/experiments/E1_plain_convlstm.yaml" \
        "$OUT_BASE/I1_plain_convlstm"

# ---- I2 = E2 ResConvLSTM: REUSE ONLY. Never train a replacement. ----
# Build an ordered candidate list, verify each with the artifact verifier,
# and use the first valid one. If none is valid -> BLOCKED.
E2_RESOLVED=""
candidates=()
if [ -n "${E2_CHECKPOINT_PATH:-}" ]; then
  candidates+=("$E2_CHECKPOINT_PATH")
fi
if [ -f "$E2_MANIFEST" ]; then
  mp="$(python -c "import json;print(json.load(open('$E2_MANIFEST')).get('checkpoint_local_path',''))" 2>/dev/null || true)"
  if [ -n "$mp" ] && [ -f "$mp" ]; then
    candidates+=("$mp")
  fi
fi
candidates+=(
  "saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth"
  "outputs/E2_resconvlstm_20ep/models/ResConvLSTM_best.pth"
  "outputs/experiments/E2_resconvlstm/models/ResConvLSTM_best.pth"
)
for ck in "${candidates[@]}"; do
  [ -f "$ck" ] || continue
  echo ""
  echo ">>> [I2] verifying candidate: $ck"
  if python scripts/verify_experiment_artifact.py \
      --experiment I2 --checkpoint "$ck" --manifest "$E2_MANIFEST" > /dev/null 2>&1; then
    E2_RESOLVED="$ck"
    echo ">>> [I2] valid official checkpoint: $ck"
    break
  fi
done
if [ -z "$E2_RESOLVED" ]; then
  echo "BLOCKED_MISSING_OR_INCOMPATIBLE_E2" >&2
  echo "No valid E2/I2 checkpoint found in any candidate location." >&2
  echo "Restore/stage the official artifact (saved_models/E2_resconvlstm_seed42/)." >&2
  echo "Do NOT train a replacement." >&2
  exit 1
fi
echo ""
echo ">>> [I2] validate-only v2 re-evaluation (never train)"
python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml \
  --mode validate-only --checkpoint "$E2_RESOLVED" \
  --out "$OUT_BASE/I2_resconvlstm"

# ---- B1 TrajGRU: train ----
run_exp "B1_trajgru" "configs/experiments/B1_trajgru.yaml" \
        "$OUT_BASE/B1_trajgru"

echo ""
echo "============================================================"
echo "Backbone Gate finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "TEST STATUS: SEALED (final test NOT run)"
echo "============================================================"
