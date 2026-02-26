#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Virtual env not found: $VENV_DIR" >&2
  echo "Run: bash install_env.sh" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

if [[ "$NODE_IP" != "$HEAD_NODE_IP" ]]; then
  echo "ERROR: run on head node only." >&2
  echo "NODE_IP=$NODE_IP, HEAD_NODE_IP=$HEAD_NODE_IP" >&2
  exit 1
fi

if [[ "$ENABLE_1M_CONTEXT" == "1" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

EXTRA_MEDIA_ARGS=()
if [[ -n "${ALLOWED_MEDIA_DOMAINS}" ]]; then
  EXTRA_MEDIA_ARGS+=(--allowed-media-domains ${ALLOWED_MEDIA_DOMAINS})
  export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
fi

KV_ARGS=()
if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
  KV_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  if [[ "$CALC_KV_SCALES" == "1" ]]; then
    KV_ARGS+=(--calculate-kv-scales)
  fi
fi

MM_ARGS=()
if [[ "$ENABLE_MULTIMODAL" == "1" ]]; then
  MM_ARGS+=(
    --mm-encoder-tp-mode data
    --mm-processor-cache-type shm
    --limit-mm-per-prompt "$LIMIT_MM_JSON"
  )
else
  MM_ARGS+=(--limit-mm-per-prompt "{}")
fi

mkdir -p "$APP_DIR/logs"

echo "[vLLM] Starting service"
echo "[vLLM] MODEL_ID=$MODEL_ID"
echo "[vLLM] TP=$TP_SIZE PP=$PP_SIZE MAX_MODEL_LEN=$MAX_MODEL_LEN"

nohup vllm serve "$MODEL_ID" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --reasoning-parser qwen3 \
  --max-model-len "$MAX_MODEL_LEN" \
  --hf-overrides "$HF_OVERRIDES_JSON" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --swap-space "$SWAP_SPACE_GB" \
  --enable-prefix-caching \
  "${MM_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  "${EXTRA_MEDIA_ARGS[@]}" \
  > "$APP_DIR/logs/vllm_server.log" 2>&1 &

echo "[vLLM] PID=$!"
echo "[vLLM] Log: $APP_DIR/logs/vllm_server.log"
echo "[vLLM] Follow: tail -f $APP_DIR/logs/vllm_server.log"
