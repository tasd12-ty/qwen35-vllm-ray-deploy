#!/bin/bash
###############################################################################
# GLM-5 (744B FP8) vLLM Multi-Node Environment Setup
#
# 硬件: 3 nodes x 8 x NVIDIA L20Z (80GB), Ubuntu Noble, CUDA 13.0
# 模型: GLM-5 (GlmMoeDsaForCausalLM) at /root/GLM
# 配置: TP=8, PP=3, Ray distributed executor
#
# 功能:
#   - 安装 uv 包管理器
#   - 创建 Python 3.12 虚拟环境
#   - 安装 vLLM nightly (cu130) + transformers (git main)
#   - 配置 NCCL 网络环境 (TCP Socket, 禁用未配置的 InfiniBand)
#   - 统一 NCCL 版本, 防止跨节点 segfault
#   - 修补 vLLM PP + Sparse Attention (DSA) 兼容性 bug
#   - 启动 Ray 集群
#
# 用法:
#   头节点:   bash setup_glm5.sh head   <head_ip>
#   工作节点: bash setup_glm5.sh worker <head_ip>
#
# 示例:
#   节点1 (10.2.1.11): bash setup_glm5.sh head   10.2.1.11
#   节点2 (10.2.1.22): bash setup_glm5.sh worker 10.2.1.11
#   节点3 (10.2.5.21): bash setup_glm5.sh worker 10.2.1.11
###############################################################################

set -euo pipefail

# ======================== 配置区 (可按需修改) ========================

VENV_DIR="/root/vllm-env"           # 虚拟环境路径
PYTHON_VERSION="3.12"               # Python 版本
CUDA_VARIANT="cu130"                # CUDA 变体 (匹配 CUDA 13.0)
NETWORK_IFACE="intranet_bond"       # 节点间通信网卡名
RAY_PORT="6379"                     # Ray head 端口
MODEL_PATH="/root/GLM"              # 模型路径

# ======================== 参数解析 ========================

ROLE="${1:-}"
HEAD_IP="${2:-}"

if [[ -z "$ROLE" || -z "$HEAD_IP" ]]; then
    echo "用法: $0 <head|worker> <head_node_ip>"
    echo ""
    echo "  head   - 头节点 (启动 Ray head, 最后启动 vLLM serve)"
    echo "  worker - 工作节点 (加入 Ray 集群)"
    echo ""
    echo "示例:"
    echo "  bash $0 head   10.2.1.11"
    echo "  bash $0 worker 10.2.1.11"
    exit 1
fi

if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
    echo "错误: 角色必须是 'head' 或 'worker', 当前: $ROLE"
    exit 1
fi

TOTAL_STEPS=8
step=0
step() {
    step=$((step + 1))
    echo ""
    echo "============================================================"
    echo " [Step ${step}/${TOTAL_STEPS}] $1"
    echo "============================================================"
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GLM-5 vLLM Full Setup                                     ║"
echo "║  Role: $(printf '%-8s' "$ROLE") │ Head: $(printf '%-20s' "$HEAD_IP")       ║"
echo "║  CUDA: ${CUDA_VARIANT}    │ Python: ${PYTHON_VERSION}                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ======================== Step 1: 安装 uv ========================

step "安装 uv 包管理器"

if command -v uv &>/dev/null; then
    UV_VER=$(uv --version 2>/dev/null || echo "unknown")
    echo "  uv 已安装: $UV_VER"
    echo "  升级到最新版..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || true
else
    echo "  安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 确保 uv 在 PATH 中
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 验证
if ! command -v uv &>/dev/null; then
    echo "  ✗ uv 安装失败, 请手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✓ uv $(uv --version)"

# ======================== Step 2: 创建虚拟环境 ========================

step "创建 Python ${PYTHON_VERSION} 虚拟环境"

if [[ -d "$VENV_DIR" ]]; then
    echo "  发现已有环境: $VENV_DIR"
    read -p "  是否删除重建? (y/N): " -r REBUILD
    if [[ "$REBUILD" =~ ^[Yy]$ ]]; then
        echo "  删除旧环境..."
        rm -rf "$VENV_DIR"
    else
        echo "  保留现有环境, 跳过创建"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "  创建虚拟环境: $VENV_DIR (Python ${PYTHON_VERSION})"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed
    echo "  ✓ 虚拟环境已创建"
fi

# 激活
source "${VENV_DIR}/bin/activate"
echo "  ✓ 已激活: $(python --version), $(which python)"

# ======================== Step 3: 安装 vLLM nightly + 依赖 ========================

step "安装 vLLM nightly (${CUDA_VARIANT}) + transformers + ray"

echo "  [3a] 安装 vLLM nightly wheel (${CUDA_VARIANT})..."
echo "       源: https://wheels.vllm.ai/nightly/${CUDA_VARIANT}"

uv pip install -U vllm \
    --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_VARIANT}" \
    --prerelease=allow \
    2>&1 | tail -5

VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "安装失败")
echo "  ✓ vLLM: ${VLLM_VER}"

echo ""
echo "  [3b] 安装 transformers (git main, 支持 GLM-5/Qwen3.5 等最新模型)..."
uv pip install -U "transformers @ git+https://github.com/huggingface/transformers.git" \
    2>&1 | tail -3

TF_VER=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "安装失败")
echo "  ✓ transformers: ${TF_VER}"

echo ""
echo "  [3c] 安装/升级 ray 和其他依赖..."
uv pip install -U ray[default] \
    2>&1 | tail -3

RAY_VER=$(python -c "import ray; print(ray.__version__)" 2>/dev/null || echo "安装失败")
echo "  ✓ ray: ${RAY_VER}"

echo ""
echo "  [3d] 安装补充依赖 (accelerate, sentencepiece, protobuf)..."
uv pip install -U accelerate sentencepiece protobuf \
    2>&1 | tail -3
echo "  ✓ 补充依赖已安装"

# ======================== Step 4: NCCL 版本统一 ========================

step "统一 NCCL 版本"

# 定位 vLLM 使用的 NCCL 库
NCCL_LIB_CANDIDATES=(
    "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib/libnccl.so.2"
    "${VENV_DIR}/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
)

NCCL_SO_PATH=""
for candidate in "${NCCL_LIB_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        NCCL_SO_PATH="$candidate"
        break
    fi
done

if [[ -z "$NCCL_SO_PATH" ]]; then
    echo "  ⚠ 未找到 NCCL 库, 安装 nvidia-nccl-cu12..."
    uv pip install -U nvidia-nccl-cu12 2>&1 | tail -3
    for candidate in "${NCCL_LIB_CANDIDATES[@]}"; do
        if [[ -f "$candidate" ]]; then
            NCCL_SO_PATH="$candidate"
            break
        fi
    done
fi

if [[ -n "$NCCL_SO_PATH" ]]; then
    NCCL_VER=$(python3 -c "
import ctypes
lib = ctypes.CDLL('${NCCL_SO_PATH}')
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
major = v.value // 10000
minor = (v.value % 10000) // 100
patch = v.value % 100
print(f'{major}.{minor}.{patch}')
" 2>/dev/null || echo "unknown")
    echo "  ✓ NCCL 版本: ${NCCL_VER}"
    echo "  ✓ 库路径: ${NCCL_SO_PATH}"
else
    echo "  ⚠ NCCL 库未找到, 跨节点通信可能出错"
    NCCL_SO_PATH=""
fi

# ======================== Step 5: 环境变量 ========================

step "配置环境变量 (NCCL / Gloo / Ray)"

# 导出到当前 session
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME="${NETWORK_IFACE}"
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME="${NETWORK_IFACE}"
export TP_SOCKET_IFNAME="${NETWORK_IFACE}"
export HF_HUB_OFFLINE=1

if [[ -n "$NCCL_SO_PATH" ]]; then
    export VLLM_NCCL_SO_PATH="${NCCL_SO_PATH}"
fi

# 写入 ~/.bashrc (幂等: 先删旧块再写新块)
MARKER="# >>> GLM-5 vLLM env >>>"
MARKER_END="# <<< GLM-5 vLLM env <<<"

if grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    sed -i "/$MARKER/,/$MARKER_END/d" ~/.bashrc
fi

cat >> ~/.bashrc << BASHRC_EOF
${MARKER}
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
source ${VENV_DIR}/bin/activate

# NCCL: 强制 TCP Socket (InfiniBand 未配置时必须)
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=${NETWORK_IFACE}
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

# Gloo / TP 网络接口
export GLOO_SOCKET_IFNAME=${NETWORK_IFACE}
export TP_SOCKET_IFNAME=${NETWORK_IFACE}

# vLLM NCCL 库路径 (确保所有节点用同一版本, 避免 segfault)
export VLLM_NCCL_SO_PATH=${NCCL_SO_PATH}

# HuggingFace 离线 (模型已在本地)
export HF_HUB_OFFLINE=1
${MARKER_END}
BASHRC_EOF

echo "  ✓ 环境变量已写入 ~/.bashrc"
echo ""
echo "  关键变量:"
echo "    NCCL_IB_DISABLE    = 1       (禁用 InfiniBand)"
echo "    NCCL_NET           = Socket  (强制 TCP, 覆盖 gIB)"
echo "    NCCL_SOCKET_IFNAME = ${NETWORK_IFACE}"
echo "    GLOO_SOCKET_IFNAME = ${NETWORK_IFACE}"
echo "    VLLM_NCCL_SO_PATH  = ${NCCL_SO_PATH}"

# ======================== Step 6: vLLM 源码补丁 ========================

step "修补 vLLM 源码 (PP + Sparse Attention 兼容性)"

# 定位 vllm config 文件
VLLM_PKG_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")

if [[ -z "$VLLM_PKG_DIR" ]]; then
    echo "  ⚠ 无法定位 vllm 包目录, 跳过补丁"
else
    VLLM_CONFIG="${VLLM_PKG_DIR}/config/vllm.py"

    if [[ -f "$VLLM_CONFIG" ]]; then
        # 备份原文件 (仅首次)
        if [[ ! -f "${VLLM_CONFIG}.bak.orig" ]]; then
            cp "$VLLM_CONFIG" "${VLLM_CONFIG}.bak.orig"
            echo "  ✓ 已备份: ${VLLM_CONFIG}.bak.orig"
        fi

        # 检查是否已打过补丁
        if grep -q "layer_name in forward_context and isinstance(forward_context\[layer_name\]" "$VLLM_CONFIG"; then
            echo "  ✓ 补丁已存在, 跳过"
        else
            MATCH='if isinstance(forward_context\[layer_name\], layer_type)'
            REPLACE='if layer_name in forward_context and isinstance(forward_context[layer_name], layer_type)'

            if grep -q 'if isinstance(forward_context\[layer_name\], layer_type)' "$VLLM_CONFIG"; then
                sed -i "s/${MATCH}/${REPLACE}/" "$VLLM_CONFIG"

                if grep -q "layer_name in forward_context and isinstance" "$VLLM_CONFIG"; then
                    echo "  ✓ 补丁应用成功"
                    echo "    修复: KeyError 'model.layers.X.self_attn.indexer.k_cache'"
                    echo "    位置: ${VLLM_CONFIG}"
                else
                    echo "  ⚠ 补丁可能未正确应用, 请手动检查"
                fi
            else
                echo "  ℹ 未找到目标代码, 可能此版本已修复或代码结构已变化"
                echo "    如果启动时出现 KeyError 'self_attn.indexer.k_cache', 需手动打补丁"
            fi
        fi
    else
        echo "  ℹ 未找到 ${VLLM_CONFIG}"
        echo "    此版本 vLLM 的代码结构可能不同, 如遇 PP + DSA 报错再手动处理"
    fi
fi

# ======================== Step 7: Ray 集群 ========================

step "启动 Ray 集群 (${ROLE})"

# 先停止旧进程
echo "  停止旧 Ray / vLLM 进程..."
ray stop --force 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

if [[ "$ROLE" == "head" ]]; then
    ray start --head --port="${RAY_PORT}"
    echo ""
    echo "  ✓ Ray head 已启动: ${HEAD_IP}:${RAY_PORT}"
    echo "  → 现在请在其他节点执行:"
    echo "    bash setup_glm5.sh worker ${HEAD_IP}"
else
    echo "  连接 Ray head: ${HEAD_IP}:${RAY_PORT}..."
    CONNECTED=false
    for i in $(seq 1 30); do
        if ray start --address="${HEAD_IP}:${RAY_PORT}" 2>/dev/null; then
            CONNECTED=true
            break
        fi
        echo "    重试 ($i/30)..."
        sleep 5
    done

    if [[ "$CONNECTED" == "true" ]]; then
        echo "  ✓ 已加入 Ray 集群"
    else
        echo "  ✗ 无法连接 Ray head (${HEAD_IP}:${RAY_PORT})"
        echo "    请确认: 1) head 节点已启动  2) 防火墙放行  3) 网络连通"
        exit 1
    fi
fi

# ======================== Step 8: 验证 ========================

step "环境验证"

echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │ 软件版本                                             │"
echo "  ├─────────────────────────────────────────────────────┤"
printf "  │ %-12s : %-38s │\n" "uv" "$(uv --version 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "Python" "$(python --version 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "vLLM" "$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "PyTorch" "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "transformers" "$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "Ray" "$(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'N/A')"
printf "  │ %-12s : %-38s │\n" "NCCL" "${NCCL_VER:-N/A}"
printf "  │ %-12s : %-38s │\n" "CUDA" "$(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || echo 'N/A')"
echo "  └─────────────────────────────────────────────────────┘"

echo ""
echo "  环境变量:"
for var in NCCL_IB_DISABLE NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME VLLM_NCCL_SO_PATH; do
    val="${!var:-NOT SET}"
    printf "    %-22s = %s\n" "$var" "$val"
done

echo ""
echo "  GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | \
    while read -r line; do echo "    $line"; done || echo "    ⚠ nvidia-smi 不可用"

echo ""
echo "  Ray 集群:"
ray status 2>/dev/null | grep -E "GPU|node|Healthy" | head -5 || echo "    (ray status 获取中...)"

echo ""
echo "  vLLM 补丁:"
if [[ -n "${VLLM_PKG_DIR:-}" ]] && grep -q "layer_name in forward_context and isinstance" "${VLLM_PKG_DIR}/config/vllm.py" 2>/dev/null; then
    echo "    ✓ PP + Sparse Attention (DSA) 补丁已应用"
else
    echo "    ℹ 补丁未检测到 (可能此版本不需要或代码结构已变)"
fi

# ======================== 完成 ========================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
if [[ "$ROLE" == "head" ]]; then
    echo "║  ✓ 头节点 Setup 完成!                                      ║"
    echo "║                                                            ║"
    echo "║  等所有 worker 节点就绪后, 启动 vLLM 服务.                    ║"
    echo "║  部署指令请参考 README.md 中的 GLM-5 部署章节.                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
else
    echo "║  ✓ Worker 节点 Setup 完成! 已加入 Ray 集群.                 ║"
    echo "║  请确认头节点和其他 worker 也已完成 setup.                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi
