#!/usr/bin/env bash
set -euo pipefail

# ===== Auto Detect Helpers =====
detect_default_node_ip() {
  local detected=""
  if command -v ip >/dev/null 2>&1; then
    detected="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
  fi
  if [[ -z "$detected" ]] && command -v hostname >/dev/null 2>&1; then
    detected="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  if [[ -z "$detected" ]] && command -v ipconfig >/dev/null 2>&1; then
    detected="$(ipconfig getifaddr en0 2>/dev/null || true)"
  fi
  if [[ -z "$detected" ]]; then
    detected="127.0.0.1"
  fi
  printf '%s' "$detected"
}

detect_default_nic_name() {
  local detected=""
  if command -v ip >/dev/null 2>&1; then
    detected="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
  fi
  if [[ -z "$detected" ]] && command -v route >/dev/null 2>&1; then
    detected="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
  fi
  if [[ -z "$detected" ]]; then
    detected="eth0"
  fi
  printf '%s' "$detected"
}

# ===== Cluster Basics =====
export NODE_IP="${NODE_IP:-$(detect_default_node_ip)}"
export NIC_NAME="${NIC_NAME:-$(detect_default_nic_name)}"
export HEAD_NODE_IP="${HEAD_NODE_IP:-$NODE_IP}"
export NODE_ROLE="${NODE_ROLE:-worker}"  # head | worker
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# ===== Ray =====
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/ray_tmp}"

# ===== Model / vLLM =====
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-mm-1m}"
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8000}"

# 2 nodes x 8 GPUs -> TP=8, PP=2 (common reference setup)
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"

# ===== 1M Context via YaRN =====
# Open-source Qwen3.5 native context is lower; this enables long-context extension mode.
export ENABLE_1M_CONTEXT="${ENABLE_1M_CONTEXT:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1010000}"

# ===== Multimodal =====
export ENABLE_MULTIMODAL="${ENABLE_MULTIMODAL:-1}"
LIMIT_MM_JSON=$(cat <<'JSON'
{"image": 4, "video": 0}
JSON
)
export LIMIT_MM_JSON

# Qwen3.5 YaRN RoPE overrides
HF_OVERRIDES_JSON=$(cat <<'JSON'
{
  "text_config": {
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144
    }
  }
}
JSON
)
export HF_OVERRIDES_JSON

# ===== Runtime Tuning =====
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export SWAP_SPACE_GB="${SWAP_SPACE_GB:-32}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"  # fp8 | auto
export CALC_KV_SCALES="${CALC_KV_SCALES:-1}"

# ===== Paths =====
export APP_DIR="${APP_DIR:-$HOME/qwen35_vllm_ray}"
export VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# ===== Network / NCCL =====
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$NIC_NAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NIC_NAME}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-$NODE_IP}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# Optional media allowlist for SSRF risk reduction.
# Example: "upload.wikimedia.org github.com"
export ALLOWED_MEDIA_DOMAINS="${ALLOWED_MEDIA_DOMAINS:-}"

# ===== Validation helper =====
if [[ "$TP_SIZE" -lt 1 || "$PP_SIZE" -lt 1 ]]; then
  echo "Invalid TP_SIZE/PP_SIZE" >&2
  exit 1
fi
