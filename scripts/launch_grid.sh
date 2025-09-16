#!/usr/bin/env bash
set -euo pipefail

# Launch 8 parallel single-GPU training jobs with different configs.
# Usage: bash scripts/launch_grid.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
RUNS_DIR="$ROOT_DIR/runs"
mkdir -p "$LOG_DIR" "$RUNS_DIR" "$ROOT_DIR/hf_cache"

source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

# Offline + local HF cache
export HF_HOME="$ROOT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$ROOT_DIR/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

declare -a CONFIGS=(
  configs/exp0_local_base_stable.yaml
  configs/exp1_bf16_contrastive.yaml
  configs/exp2_unfreeze_early.yaml
  configs/exp3_deeper_hidden384.yaml
  configs/exp4_focal_loss.yaml
  configs/exp5_contrastive_temp_sweep.yaml
  configs/exp6_no_waveform.yaml
  configs/exp7_metadata_heads.yaml
)

PIDS=()
TS=$(date +%Y%m%d_%H%M%S)
for idx in $(seq 0 7); do
  CFG=${CONFIGS[$idx]}
  GPU=$idx
  LOG="$LOG_DIR/grid_${TS}_g${GPU}_$(basename ${CFG%.yaml}).log"
  echo "[GPU $GPU] Launching $CFG | LOG=$LOG"
  CUDA_VISIBLE_DEVICES=$GPU nohup python -u "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/$CFG" \
    --output "$RUNS_DIR" \
    > "$LOG" 2>&1 < /dev/null &
  PIDS+=("$!")
done

echo "Launched PIDs: ${PIDS[*]}"
echo "Logs:"
ls -1t "$LOG_DIR"/grid_${TS}_g*.log

