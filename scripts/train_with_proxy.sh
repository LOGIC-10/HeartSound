#!/usr/bin/env bash
set -euo pipefail

# Simple launcher that injects proxy + HF cache envs and starts training.
# Usage:
#   bash scripts/train_with_proxy.sh \
#     --config configs/heart_mambaformer_small.yaml \
#     --gpus 0 \
#     --output runs

PROXY_URL="http://100.64.117.161:3128"
CONFIG="configs/heart_mambaformer_small.yaml"
GPUS="0"
OUTPUT_DIR="runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --gpus)
      GPUS="$2"; shift 2 ;;
    --output)
      OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/hf_cache"

# 1) Configure Git proxy (optional but handy for code/model repos)
git config --global http.proxy  "$PROXY_URL" || true
git config --global https.proxy "$PROXY_URL" || true
git config --global http.https://github.com.proxy "$PROXY_URL" || true

# 2) Export proxy envs for Python/requests/huggingface_hub
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"

# 3) Hugging Face cache inside repo (avoid /root/.cache space issues)
export HF_HOME="$ROOT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$ROOT_DIR/hf_cache/hub"   # kept for compatibility
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_READ_TIMEOUT=120

# 4) CUDA devices
export CUDA_VISIBLE_DEVICES="$GPUS"

ts=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT_DIR/logs/train_${ts}.log"

echo "Launching training with proxy on GPUs=$GPUS" | tee -a "$LOG"
echo "Config: $CONFIG | Output: $OUTPUT_DIR" | tee -a "$LOG"

nohup python -u "$ROOT_DIR/train.py" \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR" \
  > "$LOG" 2>&1 < /dev/null &

echo "PID $! | log: $LOG"

