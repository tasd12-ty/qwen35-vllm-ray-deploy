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

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  echo "Usage: $0 {head|worker|stop|status}"
  exit 1
fi

mkdir -p "$RAY_TMPDIR"

case "$cmd" in
  head)
    echo "[Ray] Starting HEAD at ${NODE_IP}:${RAY_PORT}"
    ray stop -f || true
    ray start \
      --head \
      --node-ip-address="$NODE_IP" \
      --port="$RAY_PORT" \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="$RAY_DASHBOARD_PORT" \
      --temp-dir="$RAY_TMPDIR" \
      --num-gpus="$GPUS_PER_NODE"
    echo "[Ray] HEAD started"
    ray status || true
    ;;

  worker)
    echo "[Ray] Starting WORKER ${NODE_IP} -> ${HEAD_NODE_IP}:${RAY_PORT}"
    ray stop -f || true
    ray start \
      --address="${HEAD_NODE_IP}:${RAY_PORT}" \
      --node-ip-address="$NODE_IP" \
      --temp-dir="$RAY_TMPDIR" \
      --num-gpus="$GPUS_PER_NODE"
    echo "[Ray] WORKER joined"
    ray status || true
    ;;

  stop)
    ray stop -f || true
    ;;

  status)
    ray status || true
    python - <<'PY' || true
import ray
try:
    ray.init(address="auto", ignore_reinit_error=True)
    print("Connected to Ray cluster.")
    for n in ray.nodes():
        print({
            "NodeID": n.get("NodeID"),
            "Alive": n.get("Alive"),
            "NodeManagerAddress": n.get("NodeManagerAddress"),
            "Resources": n.get("Resources", {}),
        })
except Exception as e:
    print("Ray status error:", e)
PY
    ;;

  *)
    echo "Unknown cmd: $cmd"
    exit 1
    ;;
esac
