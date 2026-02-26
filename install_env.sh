#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$APP_DIR" "$RAY_TMPDIR" "$HF_HOME"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel

# Ray + vLLM nightly (recommended for latest Qwen support)
pip install -U "ray[default]"
pip install -U "vllm" \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu121

# client deps
pip install -U openai pillow requests

echo "=== Version check ==="
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import ray
    print("Ray:", ray.__version__)
except Exception as e:
    print("Ray import failed:", e)
try:
    import vllm
    print("vLLM:", getattr(vllm, "__version__", "unknown"))
except Exception as e:
    print("vLLM import failed:", e)
PY

echo "=== GPU check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found (skip)"
fi
