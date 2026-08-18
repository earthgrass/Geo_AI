#!/usr/bin/env bash
#
# Backbone Gate — GPU 环境一键脚本（只生成，需手动在云端 GPU 环境执行，勿自动运行）
#
# 顺序: E0 Persistence(仅 validation) -> E1 PlainConvLSTM -> E2 ResConvLSTM(已完成则跳过) -> B1 TrajGRU
#
# 约束:
#   - TEST 保持 SEALED（本脚本绝不包含 --allow-test-eval，禁止添加）
#   - 任一任务失败立即停止（set -euo pipefail）
#   - 每个实验独立 output 目录
#   - 统一 seed=42 / batch_size=4 / AMP=auto / max_epochs=20 / 同一 train-val split
#   - 输出开始/结束时间 + 完整日志（tee 到 outputs/backbone_gate/logs/）
#
# 用法（在项目根目录执行）:
#   bash scripts/run_backbone_gate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS=20
BATCH_SIZE=4
AMP=auto          # 云端有 CUDA -> auto 会开 AMP（等价于用户要求的 AMP=auto）
H5="ConvLSTM_Dataset_128.h5"

OUT_BASE="outputs/backbone_gate"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/backbone_gate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Backbone Gate started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  epochs=$EPOCHS batch=$BATCH_SIZE amp=$AMP"
echo "  output base=$OUT_BASE"
echo "  log=$LOG"
echo "  TEST STATUS: SEALED (no --allow-test-eval)"
echo "============================================================"

# 独立 output 目录 + 失败即停（set -e 已覆盖）
run_exp() {
  local name="$1"
  local config="$2"
  local out_dir="$OUT_BASE/$name"
  echo ""
  echo ">>> [$name] start $(date '+%H:%M:%S')"
  python scripts/run_experiment.py \
    --config "$config" \
    --h5 "$H5" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --amp "$AMP" \
    --out "$out_dir"
  echo "<<< [$name] done $(date '+%H:%M:%S')"
}

# E0: Persistence — 不训练，仅 validation（run_experiment.py 自动识别 Persistence 分支）
run_exp "E0_persistence" "configs/experiments/E0_persistence.yaml"

# E1: PlainConvLSTM
run_exp "E1_plain_convlstm" "configs/experiments/E1_plain_convlstm.yaml"

# E2: ResConvLSTM — 若正式结果已归档（checkpoint + manifest 同时存在）则跳过，不重复训练
E2_CKPT="saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth"
E2_MANIFEST="results/E2_resconvlstm_seed42/manifest.json"
if [ -f "$E2_CKPT" ] && [ -f "$E2_MANIFEST" ]; then
  echo ""
  echo ">>> [E2_resconvlstm] SKIP — official result already present (checkpoint + manifest)"
else
  run_exp "E2_resconvlstm" "configs/experiments/E2_resconvlstm.yaml"
fi

# B1: TrajGRU
run_exp "B1_trajgru" "configs/experiments/B1_trajgru.yaml"

echo ""
echo "============================================================"
echo "Backbone Gate finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "TEST STATUS: SEALED (final test NOT run)"
echo "============================================================"
