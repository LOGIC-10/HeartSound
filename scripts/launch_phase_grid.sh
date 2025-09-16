#!/usr/bin/env bash
set -euo pipefail

# Launch 7 parallel single-GPU jobs (GPU 1-7) for phase-aware experiments.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
RUNS_DIR="$ROOT_DIR/runs"
mkdir -p "$LOG_DIR" "$RUNS_DIR" "$ROOT_DIR/hf_cache"

source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

export HF_HOME="$ROOT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$ROOT_DIR/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

declare -a GPUS=(1 2 3 4 5 6 7)
declare -a CONFIGS=(
  configs/expB1_phase_gate_w05_k15.yaml
  configs/expB2_phase_gate_w08_k31.yaml
  configs/expB3_phase_gate_contrastive.yaml
  configs/expB4_phase_gate_focal.yaml
  configs/expB5_phase_gate_unfreeze1.yaml
  configs/expB6_phase_gate_deeper.yaml
  configs/expB7_phase_gate_mel_only.yaml
)

PIDS=()
TS=$(date +%Y%m%d_%H%M%S)
for i in $(seq 0 6); do
  GPU=${GPUS[$i]}
  CFG=${CONFIGS[$i]}
  LOG="$LOG_DIR/phase_${TS}_g${GPU}_$(basename ${CFG%.yaml}).log"
  echo "[GPU $GPU] Launching $CFG | LOG=$LOG"
  CUDA_VISIBLE_DEVICES=$GPU nohup python -u "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/$CFG" \
    --output "$RUNS_DIR" \
    > "$LOG" 2>&1 < /dev/null &
  PIDS+=("$!")
done

echo "Launched PIDs: ${PIDS[*]}"
ls -1t "$LOG_DIR"/phase_${TS}_g*.log

