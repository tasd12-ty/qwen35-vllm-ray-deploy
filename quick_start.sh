#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

usage() {
  cat <<'USAGE'
One-command bootstrap for Ray + vLLM Qwen3.5.

Usage:
  bash quick_start.sh --cluster-ips "10.0.0.1,10.0.0.2" --install --start-vllm

Options:
  --cluster-ips <csv>     Comma-separated node IPs. First IP is treated as head.
  --head-ip <ip>          Explicit head IP. Overrides first value from --cluster-ips.
  --role <auto|head|worker>
                          Defaults to auto.
  --expected-nodes <n>    Expected alive nodes before head starts vLLM.
  --install               Run install_env.sh before starting Ray.
  --start-vllm            Start vLLM on head node after cluster is ready (default on).
  --no-start-vllm         Skip vLLM start.
  --wait-seconds <sec>    Max wait for cluster readiness on head. Default 600.
  --no-wait               Do not wait for worker nodes.
  -h, --help              Show this help.
USAGE
}

contains_word() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

detect_route_ip_and_nic() {
  local target="$1"
  local route_line=""
  local detected_ip=""
  local detected_nic=""

  if command -v ip >/dev/null 2>&1; then
    route_line="$(ip route get "$target" 2>/dev/null | head -n1 || true)"
    detected_ip="$(awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' <<<"$route_line")"
    detected_nic="$(awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}' <<<"$route_line")"
  fi

  printf '%s;%s' "$detected_ip" "$detected_nic"
}

ROLE="auto"
CLUSTER_IPS=""
HEAD_IP_ARG=""
RUN_INSTALL=0
RUN_START_VLLM=1
WAIT_FOR_CLUSTER=1
WAIT_SECONDS=600
EXPECTED_NODES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster-ips)
      CLUSTER_IPS="${2:-}"
      shift 2
      ;;
    --head-ip)
      HEAD_IP_ARG="${2:-}"
      shift 2
      ;;
    --role)
      ROLE="${2:-auto}"
      shift 2
      ;;
    --expected-nodes)
      EXPECTED_NODES="${2:-}"
      shift 2
      ;;
    --install)
      RUN_INSTALL=1
      shift
      ;;
    --start-vllm)
      RUN_START_VLLM=1
      shift
      ;;
    --no-start-vllm)
      RUN_START_VLLM=0
      shift
      ;;
    --wait-seconds)
      WAIT_SECONDS="${2:-600}"
      shift 2
      ;;
    --no-wait)
      WAIT_FOR_CLUSTER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$ROLE" != "auto" && "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "--role must be auto/head/worker" >&2
  exit 1
fi

CLUSTER_LIST=()
if [[ -n "$CLUSTER_IPS" ]]; then
  IFS=',' read -r -a RAW_CLUSTER_LIST <<< "$CLUSTER_IPS"
  for ip in "${RAW_CLUSTER_LIST[@]}"; do
    ip="${ip//[[:space:]]/}"
    if [[ -n "$ip" ]]; then
      CLUSTER_LIST+=("$ip")
    fi
  done
fi

if [[ -n "$HEAD_IP_ARG" ]]; then
  HEAD_NODE_IP="$HEAD_IP_ARG"
elif [[ ${#CLUSTER_LIST[@]} -gt 0 ]]; then
  HEAD_NODE_IP="${CLUSTER_LIST[0]}"
fi

if [[ ${#CLUSTER_LIST[@]} -gt 0 && -z "$EXPECTED_NODES" ]]; then
  EXPECTED_NODES="${#CLUSTER_LIST[@]}"
fi
if [[ -z "$EXPECTED_NODES" ]]; then
  EXPECTED_NODES=2
fi

route_info="$(detect_route_ip_and_nic "$HEAD_NODE_IP")"
route_ip="${route_info%%;*}"
route_nic="${route_info##*;}"

if [[ -n "$route_ip" ]]; then
  NODE_IP="$route_ip"
fi
if [[ -n "$route_nic" && "$route_nic" != "lo" ]]; then
  NIC_NAME="$route_nic"
fi

if [[ ${#CLUSTER_LIST[@]} -gt 0 ]]; then
  if ! contains_word "$NODE_IP" "${CLUSTER_LIST[@]}"; then
    if command -v hostname >/dev/null 2>&1; then
      local_ips="$(hostname -I 2>/dev/null || true)"
      for cluster_ip in "${CLUSTER_LIST[@]}"; do
        if grep -qw "$cluster_ip" <<<"$local_ips"; then
          NODE_IP="$cluster_ip"
          break
        fi
      done
    fi
  fi
fi

if [[ "$ROLE" == "auto" ]]; then
  if [[ "$NODE_IP" == "$HEAD_NODE_IP" ]]; then
    ROLE="head"
  else
    ROLE="worker"
  fi
fi

export NODE_IP
export HEAD_NODE_IP
export NIC_NAME
export NODE_ROLE="$ROLE"

echo "[quick-start] node_ip=$NODE_IP"
echo "[quick-start] head_ip=$HEAD_NODE_IP"
echo "[quick-start] nic=$NIC_NAME"
echo "[quick-start] role=$NODE_ROLE"
echo "[quick-start] expected_nodes=$EXPECTED_NODES"

if [[ "$RUN_INSTALL" == "1" ]]; then
  echo "[quick-start] installing runtime ..."
  bash "$SCRIPT_DIR/install_env.sh"
fi

if [[ "$NODE_ROLE" == "head" ]]; then
  bash "$SCRIPT_DIR/ray_cluster.sh" head
else
  bash "$SCRIPT_DIR/ray_cluster.sh" worker
fi

if [[ "$NODE_ROLE" == "head" && "$RUN_START_VLLM" == "1" ]]; then
  if [[ "$WAIT_FOR_CLUSTER" == "1" && "$EXPECTED_NODES" -gt 1 ]]; then
    echo "[quick-start] waiting for workers ..."
    source "$VENV_DIR/bin/activate"
    python - "$EXPECTED_NODES" "$WAIT_SECONDS" <<'PY'
import sys
import time

expected = int(sys.argv[1])
timeout = int(sys.argv[2])
deadline = time.time() + timeout

try:
    import ray
except Exception as exc:
    print(f"[quick-start] ray import failed: {exc}")
    sys.exit(1)

while time.time() < deadline:
    try:
        ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
        alive = sum(1 for n in ray.nodes() if n.get("Alive"))
        print(f"[quick-start] alive nodes: {alive}/{expected}")
        if alive >= expected:
            sys.exit(0)
    except Exception as exc:
        print(f"[quick-start] wait retry: {exc}")
    time.sleep(5)

print(f"[quick-start] timeout: cluster not ready in {timeout}s")
sys.exit(1)
PY
  fi

  bash "$SCRIPT_DIR/start_vllm_qwen35.sh"
fi

echo "[quick-start] done"
