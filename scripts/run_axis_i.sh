#!/usr/bin/env bash
#
# Axis I — Input Information Ablation (GPU host; generate only)
#
# Order (docs/MINIMAX_IMPLEMENTATION_SPEC.md §11):
#   I2/E2: validate-only official checkpoint (NEVER train)
#   I3/E3: train, then validation v2
#   I4/E4: train, then validation v2
#   I5/E5=P0: train once, then validation v2
#
# I0/I1 are handled by the backbone gate. I2 here is validation-only reuse,
# NOT a repeated training run. After I5, the artifact is verified as BOTH I5
# and P0 (one checkpoint, one run).
#
# Constraints: set -euo pipefail; independent output dirs; timestamped logs;
#   start/end timestamps; frozen YAML values (seed 42 / batch 4 / 20 epochs /
#   AMP auto); TEST SEALED (no --allow-test-eval, no test path).
#
# Usage: bash scripts/run_axis_i.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

E2_MANIFEST="results/E2_resconvlstm_seed42/manifest.json"

OUT_BASE="outputs/axis_i"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/axis_i_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Axis I (Input Information Ablation) started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  output base=$OUT_BASE"
echo "  log=$LOG"
echo "  TEST STATUS: SEALED (no --allow-test-eval, no test path)"
echo "============================================================"

# ---- Preflight: config matrix validates + CUDA required for trainable rows ----
python - <<'PY'
import sys, yaml
sys.path.insert(0, '.')
from src.experiments.registry import validate_formal_config
cfgs = [
    "configs/experiments/E2_resconvlstm.yaml",
    "configs/experiments/E3_resconvlstm_cma.yaml",
    "configs/experiments/E4_static_terrain.yaml",
    "configs/experiments/E5_terrain_geometry.yaml",
]
for f in cfgs:
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    validate_formal_config(cfg, f)
print("[preflight] Axis-I config matrix OK")
PY
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Axis I requires CUDA for formal trainable rows (I3/I4/I5)." >&2
  exit 1
fi
echo "[preflight] CUDA available"

run_exp() {
  local name="$1" config="$2" out="$3"
  echo ""
  echo ">>> [$name] start $(date '+%H:%M:%S')"
  python scripts/run_experiment.py --config "$config" --out "$out"
  echo "<<< [$name] done $(date '+%H:%M:%S')"
}

# ---- I2 = E2 ResConvLSTM: validate-only reuse (never train) ----
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
  if python scripts/verify_experiment_artifact.py \
      --experiment I2 --checkpoint "$ck" --manifest "$E2_MANIFEST" > /dev/null 2>&1; then
    E2_RESOLVED="$ck"
    break
  fi
done
if [ -z "$E2_RESOLVED" ]; then
  echo "BLOCKED_MISSING_OR_INCOMPATIBLE_E2" >&2
  echo "Run scripts/run_backbone_gate.sh first, or restore the official checkpoint." >&2
  exit 1
fi
# I2 is validate-only REUSE (never train). No run_exp() call here — run_exp
# would invoke the default training mode, which is forbidden for E2/I2.
echo ""
echo ">>> [I2] validate-only v2 re-evaluation (never train)"
python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml \
  --mode validate-only --checkpoint "$E2_RESOLVED" \
  --out "$OUT_BASE/I2_resconvlstm"

# ---- I3 = E3 ResConvLSTM + CMA: train ----
run_exp "I3_resconvlstm_cma" "configs/experiments/E3_resconvlstm_cma.yaml" \
        "$OUT_BASE/I3_resconvlstm_cma"

# ---- I4 = E4 ResConvLSTM + static terrain: train ----
run_exp "I4_static_terrain" "configs/experiments/E4_static_terrain.yaml" \
        "$OUT_BASE/I4_static_terrain"

# ---- I5 = E5 ResConvLSTM + terrain geometry: train ONCE, then register as P0 ----
run_exp "I5_terrain_geometry" "configs/experiments/E5_terrain_geometry.yaml" \
        "$OUT_BASE/I5_terrain_geometry"

# Verify I5 == P0 (same artifact identity) and record the checkpoint path.
I5_MANIFEST="$OUT_BASE/I5_terrain_geometry/manifest.json"
I5_CKPT="$(python -c "import json;print(json.load(open('$I5_MANIFEST'))['checkpoint_path'])" 2>/dev/null || true)"
if [ -z "$I5_CKPT" ] || [ ! -f "$I5_CKPT" ]; then
  echo "BLOCKED_I5_CHECKPOINT_MISSING: $I5_CKPT" >&2
  exit 1
fi
echo ""
echo ">>> [I5/P0] verifying artifact identity"
python scripts/verify_experiment_artifact.py \
  --experiment I5 --checkpoint "$I5_CKPT" --manifest "$I5_MANIFEST"
python scripts/verify_experiment_artifact.py \
  --experiment P0 --checkpoint "$I5_CKPT" --manifest "$I5_MANIFEST"
echo ">>> [I5/P0] I5 == P0 single artifact confirmed: $I5_CKPT"

echo ""
echo "============================================================"
echo "Axis I finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "TEST STATUS: SEALED (final test NOT run)"
echo "============================================================"
