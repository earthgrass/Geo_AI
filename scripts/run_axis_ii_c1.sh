#!/usr/bin/env bash
#
# Axis II Stage C1 — Loss / Inductive-Bias Ablation (GPU host; generate only)
#
# Order (docs/MINIMAX_IMPLEMENTATION_SPEC.md §11):
#   P0 = reuse the exact I5/E5 artifact (NEVER train P0 separately)
#   P1: train, then validation v2
#   P2/E6: train, then validation v2
#   P3: train, then validation v2
#
# All rows use all-12 canonical channels and the same ResConvLSTM backbone.
# P4/P5 are BLOCKED_BY_ENVIRONMENTAL_WIND_DATA and have no runnable path here;
# ERA5 is never treated as an available runtime input.
#
# Constraints: set -euo pipefail; independent output dirs; timestamped logs;
#   seed 42 / batch 4 / 20 epochs / AMP auto (frozen YAMLs); TEST SEALED.
#
# Usage: bash scripts/run_axis_ii_c1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_BASE="outputs/axis_ii_c1"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/axis_ii_c1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Axis II Stage C1 (Loss / Inductive-Bias Ablation) started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  output base=$OUT_BASE"
echo "  log=$LOG"
echo "  TEST STATUS: SEALED (no --allow-test-eval, no test path)"
echo "  P4/P5: BLOCKED_BY_ENVIRONMENTAL_WIND_DATA (no runnable config)"
echo "============================================================"

# ---- Preflight: config matrix validates + CUDA required ----
python - <<'PY'
import sys, yaml
sys.path.insert(0, '.')
from src.experiments.registry import validate_formal_config
cfgs = [
    "configs/experiments/E5_terrain_geometry.yaml",  # P0 == I5
    "configs/experiments/P1_resconvlstm_smooth.yaml",
    "configs/experiments/E6_terrain_extreme.yaml",   # P2
    "configs/experiments/P3_resconvlstm_smooth_extreme.yaml",
]
for f in cfgs:
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    validate_formal_config(cfg, f)
print("[preflight] Axis-II C1 config matrix OK")
PY
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Axis II C1 requires CUDA for formal trainable rows (P1/P2/P3)." >&2
  exit 1
fi
echo "[preflight] CUDA available"

# ---- P0 == I5: resolve the exact I5 checkpoint, verify, reuse (no training) ----
I5_CKPT="${I5_CHECKPOINT_PATH:-}"
if [ -z "$I5_CKPT" ] || [ ! -f "$I5_CKPT" ]; then
  AXIS_I_MANIFEST="outputs/axis_i/I5_terrain_geometry/manifest.json"
  if [ -f "$AXIS_I_MANIFEST" ]; then
    I5_CKPT="$(python -c "import json;print(json.load(open('$AXIS_I_MANIFEST'))['checkpoint_path'])" 2>/dev/null || true)"
  fi
fi
if [ -z "$I5_CKPT" ] || [ ! -f "$I5_CKPT" ]; then
  echo "BLOCKED_P0_MISSING_OR_INCOMPATIBLE_I5" >&2
  echo "P0 must resolve to the exact I5/E5 checkpoint. Set I5_CHECKPOINT_PATH or " >&2
  echo "run scripts/run_axis_i.sh first. Never train P0 separately." >&2
  exit 1
fi
echo ""
echo ">>> [P0=I5] verifying artifact identity"
I5_MANIFEST="outputs/axis_i/I5_terrain_geometry/manifest.json"
if [ -f "$I5_MANIFEST" ]; then
  python scripts/verify_experiment_artifact.py \
    --experiment I5 --checkpoint "$I5_CKPT" --manifest "$I5_MANIFEST"
  python scripts/verify_experiment_artifact.py \
    --experiment P0 --checkpoint "$I5_CKPT" --manifest "$I5_MANIFEST"
else
  python scripts/verify_experiment_artifact.py \
    --experiment I5 --checkpoint "$I5_CKPT"
  python scripts/verify_experiment_artifact.py \
    --experiment P0 --checkpoint "$I5_CKPT"
fi
echo ">>> [P0=I5] registered without copying or training: $I5_CKPT"

run_exp() {
  local name="$1" config="$2" out="$3"
  echo ""
  echo ">>> [$name] start $(date '+%H:%M:%S')"
  python scripts/run_experiment.py --config "$config" --out "$out"
  echo "<<< [$name] done $(date '+%H:%M:%S')"
}

# ---- P1 = smoothness ----
run_exp "P1_resconvlstm_smooth" "configs/experiments/P1_resconvlstm_smooth.yaml" \
        "$OUT_BASE/P1_resconvlstm_smooth"

# ---- P2 = E6 terrain extreme ----
run_exp "P2_terrain_extreme" "configs/experiments/E6_terrain_extreme.yaml" \
        "$OUT_BASE/P2_terrain_extreme"

# ---- P3 = smoothness + extreme ----
run_exp "P3_resconvlstm_smooth_extreme" "configs/experiments/P3_resconvlstm_smooth_extreme.yaml" \
        "$OUT_BASE/P3_resconvlstm_smooth_extreme"

echo ""
echo "============================================================"
echo "Axis II Stage C1 finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "TEST STATUS: SEALED (final test NOT run)"
echo "============================================================"
