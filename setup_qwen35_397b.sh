#!/bin/bash
###############################################################################
# Qwen3.5-397B-A17B (FP8) vLLM Multi-Node Environment Setup
#
# 模型: Qwen/Qwen3.5-397B-A17B-FP8 (397B 总参数, 17B 激活, MoE)
# 架构: Gated DeltaNet + Sparse MoE, 60 层, 512 experts (10 routed + 1 shared)
# 原生上下文: 262,144 tokens, 可通过 YaRN 扩展至 ~1M tokens
#
# 硬件要求:
#   单节点: 8 x NVIDIA H200 (141GB) 或 H100/A100 (80GB) — FP8 必须
#   多节点: 2+ 节点 x 8 GPU, Ray 分布式执行
#   显存需求: FP8 权重 ~400GB, 8x80GB GPU 可勉强装下, 8x141GB 更宽裕
#
# 部署模式 (通过 --deploy-mode 选择):
#   tp    - 张量并行 (TP=8), 多节点加 PP, 经典稳定方案 (默认)
#   ep    - 专家并行 + 数据并行 (EP+DP), MoE 高吞吐推荐方案
#
# 功能:
#   - 安装 uv 包管理器 + Python 虚拟环境
#   - 安装 vLLM nightly + transformers (git main) + ray
#   - 统一 NCCL 版本
#   - 配置 NCCL / Gloo 网络环境
#   - 启动 Ray 集群 (head / worker)
#   - 输出各部署模式的 vLLM 启动命令
#
# 用法:
#   头节点:   bash setup_qwen35_397b.sh head   <head_ip> [选项]
#   工作节点: bash setup_qwen35_397b.sh worker <head_ip> [选项]
#
# 选项:
#   --deploy-mode tp|ep      部署模式 (默认: tp)
#   --cuda cu121|cu124|cu130  CUDA 变体 (默认: cu121)
#   --nic <name>             网卡名 (默认: 自动检测)
#   --model-path <path>      本地模型路径 (默认: 从 HF 下载 FP8 版)
#   --enable-1m-context      启用 1M 长上下文 YaRN 扩展
#   --multimodal             启用多模态 (视觉编码器)
#   --language-model-only    仅加载语言模型 (跳过视觉编码器, 省显存)
#   --venv-dir <path>        虚拟环境路径 (默认: /root/qwen397b-env)
#   --python <ver>           Python 版本 (默认: 3.10)
#
# 示例:
#   # 单节点 8xH200, EP 模式高吞吐
#   bash setup_qwen35_397b.sh head 10.0.0.1 --deploy-mode ep
#
#   # 2节点 TP 模式, 1M 长上下文
#   节点1: bash setup_qwen35_397b.sh head   10.0.0.1 --enable-1m-context
#   节点2: bash setup_qwen35_397b.sh worker 10.0.0.1 --enable-1m-context
#
#   # 本地模型 + 多模态
#   bash setup_qwen35_397b.sh head 10.0.0.1 --model-path /data/Qwen3.5-397B-A17B-FP8 --multimodal
###############################################################################

set -euo pipefail

# ======================== 默认配置 ========================

VENV_DIR="/root/qwen397b-env"
PYTHON_VERSION="3.10"
CUDA_VARIANT="cu121"
NETWORK_IFACE=""               # 空 = 自动检测
RAY_PORT="6379"
MODEL_PATH=""                  # 空 = 使用 HF 模型 ID
MODEL_ID="Qwen/Qwen3.5-397B-A17B-FP8"
SERVED_MODEL_NAME="qwen3.5-397b"
DEPLOY_MODE="tp"               # tp | ep
ENABLE_1M_CONTEXT=0
ENABLE_MULTIMODAL=0
LANGUAGE_MODEL_ONLY=0
VLLM_PORT=8000
GPUS_PER_NODE=8

# ======================== 参数解析 ========================

ROLE="${1:-}"
HEAD_IP="${2:-}"
shift 2 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy-mode)  DEPLOY_MODE="$2";       shift 2 ;;
        --cuda)         CUDA_VARIANT="$2";      shift 2 ;;
        --nic)          NETWORK_IFACE="$2";     shift 2 ;;
        --model-path)   MODEL_PATH="$2";        shift 2 ;;
        --venv-dir)     VENV_DIR="$2";          shift 2 ;;
        --python)       PYTHON_VERSION="$2";    shift 2 ;;
        --enable-1m-context)    ENABLE_1M_CONTEXT=1;    shift ;;
        --multimodal)           ENABLE_MULTIMODAL=1;    shift ;;
        --language-model-only)  LANGUAGE_MODEL_ONLY=1;  shift ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 验证必须参数
if [[ -z "$ROLE" || -z "$HEAD_IP" ]]; then
    echo "用法: $0 <head|worker> <head_node_ip> [选项]"
    echo ""
    echo "  head   - 头节点 (启动 Ray head, 最后启动 vLLM serve)"
    echo "  worker - 工作节点 (加入 Ray 集群)"
    echo ""
    echo "选项:"
    echo "  --deploy-mode tp|ep      部署模式 (默认: tp)"
    echo "  --cuda cu121|cu124|cu130  CUDA 变体 (默认: cu121)"
    echo "  --nic <name>             网卡名 (默认: 自动检测)"
    echo "  --model-path <path>      本地模型路径"
    echo "  --enable-1m-context      启用 1M 长上下文"
    echo "  --multimodal             启用多模态"
    echo "  --language-model-only    仅语言模型 (省显存)"
    echo "  --venv-dir <path>        虚拟环境路径"
    echo "  --python <ver>           Python 版本"
    echo ""
    echo "示例:"
    echo "  bash $0 head   10.0.0.1 --deploy-mode ep"
    echo "  bash $0 worker 10.0.0.1"
    exit 1
fi

if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
    echo "错误: 角色必须是 'head' 或 'worker', 当前: $ROLE"
    exit 1
fi

if [[ "$DEPLOY_MODE" != "tp" && "$DEPLOY_MODE" != "ep" ]]; then
    echo "错误: 部署模式必须是 'tp' 或 'ep', 当前: $DEPLOY_MODE"
    exit 1
fi

# 自动检测网卡
if [[ -z "$NETWORK_IFACE" ]]; then
    if command -v ip >/dev/null 2>&1; then
        NETWORK_IFACE="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
    fi
    if [[ -z "$NETWORK_IFACE" ]]; then
        NETWORK_IFACE="eth0"
    fi
fi

# 确定实际模型标识
EFFECTIVE_MODEL="${MODEL_PATH:-$MODEL_ID}"

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
echo "║  Qwen3.5-397B-A17B (FP8) vLLM Setup                       ║"
echo "║  Role: $(printf '%-8s' "$ROLE") │ Head: $(printf '%-20s' "$HEAD_IP")       ║"
echo "║  Mode: $(printf '%-8s' "$DEPLOY_MODE") │ CUDA: $(printf '%-6s' "$CUDA_VARIANT") │ Py: ${PYTHON_VERSION}        ║"
echo "║  Model: $(printf '%-49s' "$EFFECTIVE_MODEL")║"
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

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "  uv 安装失败, 请手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  uv $(uv --version)"

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
    echo "  虚拟环境已创建"
fi

source "${VENV_DIR}/bin/activate"
echo "  已激活: $(python --version), $(which python)"

# ======================== Step 3: 安装 vLLM nightly + 依赖 ========================

step "安装 vLLM nightly (${CUDA_VARIANT}) + transformers + ray"

echo "  [3a] 安装 vLLM nightly wheel (${CUDA_VARIANT})..."
echo "       源: https://wheels.vllm.ai/nightly/${CUDA_VARIANT}"

# vLLM >= 0.17.0 正式支持 Qwen3.5 的 Gated DeltaNet + MoE 架构
# 在此之前必须用 nightly 版本
uv pip install -U vllm \
    --extra-index-url "https://wheels.vllm.ai/nightly/${CUDA_VARIANT}" \
    --prerelease=allow \
    2>&1 | tail -5

VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "安装失败")
echo "  vLLM: ${VLLM_VER}"

echo ""
echo "  [3b] 安装 transformers (git main, 支持 Qwen3.5-397B 架构)..."
uv pip install -U "transformers @ git+https://github.com/huggingface/transformers.git" \
    2>&1 | tail -3

TF_VER=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "安装失败")
echo "  transformers: ${TF_VER}"

echo ""
echo "  [3c] 安装/升级 ray 和其他依赖..."
uv pip install -U "ray[default]" \
    2>&1 | tail -3

RAY_VER=$(python -c "import ray; print(ray.__version__)" 2>/dev/null || echo "安装失败")
echo "  ray: ${RAY_VER}"

echo ""
echo "  [3d] 安装补充依赖 (accelerate, sentencepiece, protobuf, openai)..."
uv pip install -U accelerate sentencepiece protobuf openai pillow requests \
    2>&1 | tail -3
echo "  补充依赖已安装"

# ======================== Step 4: NCCL 版本统一 ========================

step "统一 NCCL 版本"

NCCL_LIB_CANDIDATES=(
    "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib/libnccl.so.2"
)

NCCL_SO_PATH=""
for candidate in "${NCCL_LIB_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        NCCL_SO_PATH="$candidate"
        break
    fi
done

if [[ -z "$NCCL_SO_PATH" ]]; then
    echo "  未找到 NCCL 库, 安装 nvidia-nccl-cu12..."
    uv pip install -U nvidia-nccl-cu12 2>&1 | tail -3
    for candidate in "${NCCL_LIB_CANDIDATES[@]}"; do
        if [[ -f "$candidate" ]]; then
            NCCL_SO_PATH="$candidate"
            break
        fi
    done
fi

NCCL_VER="unknown"
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
    echo "  NCCL 版本: ${NCCL_VER}"
    echo "  库路径: ${NCCL_SO_PATH}"
else
    echo "  NCCL 库未找到, 跨节点通信可能出错"
    NCCL_SO_PATH=""
fi

# ======================== Step 5: 环境变量 ========================

step "配置环境变量 (NCCL / Gloo / Ray)"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NETWORK_IFACE}"
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME="${NETWORK_IFACE}"
export TP_SOCKET_IFNAME="${NETWORK_IFACE}"

if [[ -n "$NCCL_SO_PATH" ]]; then
    export VLLM_NCCL_SO_PATH="${NCCL_SO_PATH}"
fi

# 写入 ~/.bashrc (幂等: 先删旧块再写新块)
MARKER="# >>> Qwen3.5-397B vLLM env >>>"
MARKER_END="# <<< Qwen3.5-397B vLLM env <<<"

if grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    sed -i "/$MARKER/,/$MARKER_END/d" ~/.bashrc
fi

cat >> ~/.bashrc << BASHRC_EOF
${MARKER}
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
source ${VENV_DIR}/bin/activate

# NCCL 网络配置
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NETWORK_IFACE}
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Gloo / TP 网络接口
export GLOO_SOCKET_IFNAME=${NETWORK_IFACE}
export TP_SOCKET_IFNAME=${NETWORK_IFACE}

# vLLM NCCL 库路径
export VLLM_NCCL_SO_PATH=${NCCL_SO_PATH}
${MARKER_END}
BASHRC_EOF

echo "  环境变量已写入 ~/.bashrc"
echo ""
echo "  关键变量:"
echo "    NCCL_IB_DISABLE    = ${NCCL_IB_DISABLE:-0}"
echo "    NCCL_SOCKET_IFNAME = ${NETWORK_IFACE}"
echo "    GLOO_SOCKET_IFNAME = ${NETWORK_IFACE}"
echo "    VLLM_NCCL_SO_PATH  = ${NCCL_SO_PATH}"

# ======================== Step 6: Ray 集群 ========================

step "启动 Ray 集群 (${ROLE})"

echo "  停止旧 Ray / vLLM 进程..."
ray stop --force 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

if [[ "$ROLE" == "head" ]]; then
    ray start --head --port="${RAY_PORT}" --num-gpus="${GPUS_PER_NODE}"
    echo ""
    echo "  Ray head 已启动: ${HEAD_IP}:${RAY_PORT}"
    echo "  (如需多节点) 请在其他节点执行:"
    echo "    bash setup_qwen35_397b.sh worker ${HEAD_IP} [相同选项]"
else
    echo "  连接 Ray head: ${HEAD_IP}:${RAY_PORT}..."
    CONNECTED=false
    for i in $(seq 1 30); do
        if ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus="${GPUS_PER_NODE}" 2>/dev/null; then
            CONNECTED=true
            break
        fi
        echo "    重试 ($i/30)..."
        sleep 5
    done

    if [[ "$CONNECTED" == "true" ]]; then
        echo "  已加入 Ray 集群"
    else
        echo "  无法连接 Ray head (${HEAD_IP}:${RAY_PORT})"
        echo "    请确认: 1) head 节点已启动  2) 防火墙放行  3) 网络连通"
        exit 1
    fi
fi

# ======================== Step 7: 验证 ========================

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
for var in NCCL_IB_DISABLE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME VLLM_NCCL_SO_PATH; do
    val="${!var:-NOT SET}"
    printf "    %-22s = %s\n" "$var" "$val"
done

echo ""
echo "  GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | \
    while read -r line; do echo "    $line"; done || echo "    nvidia-smi 不可用"

echo ""
echo "  Ray 集群:"
ray status 2>/dev/null | grep -E "GPU|node|Healthy" | head -5 || echo "    (ray status 获取中...)"

# ======================== Step 8: 输出 vLLM 启动命令 ========================

step "vLLM 启动命令参考"

# 构建 YaRN 长上下文 JSON
YARN_HF_OVERRIDES='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'

if [[ "$ROLE" == "head" ]]; then

    echo ""
    echo "  所有节点就绪后, 在头节点执行以下命令启动 vLLM 服务."
    echo ""

    if [[ "$DEPLOY_MODE" == "ep" ]]; then
        # ==================== EP (Expert Parallel) 模式 ====================
        echo "  ┌──────────────────────────────────────────────────────────┐"
        echo "  │ 模式: Expert Parallel + Data Parallel (EP+DP)            │"
        echo "  │ 推荐: MoE 高吞吐场景, 高并发                              │"
        echo "  └──────────────────────────────────────────────────────────┘"

        echo ""
        echo "  === 文本模式 (最高吞吐, 跳过视觉编码器) ==="
        echo ""
        echo "    vllm serve ${EFFECTIVE_MODEL} \\"
        echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
        echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
        echo "      -dp ${GPUS_PER_NODE} \\"
        echo "      --enable-expert-parallel \\"
        echo "      --language-model-only \\"
        echo "      --reasoning-parser qwen3 \\"
        echo "      --enable-prefix-caching"

        echo ""
        echo "  === 多模态模式 (图文理解) ==="
        echo ""
        echo "    vllm serve ${EFFECTIVE_MODEL} \\"
        echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
        echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
        echo "      -dp ${GPUS_PER_NODE} \\"
        echo "      --enable-expert-parallel \\"
        echo "      --mm-encoder-tp-mode data \\"
        echo "      --mm-processor-cache-type shm \\"
        echo "      --reasoning-parser qwen3 \\"
        echo "      --enable-prefix-caching"

        if [[ "$ENABLE_1M_CONTEXT" == "1" ]]; then
            echo ""
            echo "  === EP + 1M 长上下文 (YaRN) ==="
            echo ""
            echo "    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${EFFECTIVE_MODEL} \\"
            echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
            echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
            echo "      -dp ${GPUS_PER_NODE} \\"
            echo "      --enable-expert-parallel \\"
            echo "      --language-model-only \\"
            echo "      --reasoning-parser qwen3 \\"
            echo "      --enable-prefix-caching \\"
            echo "      --hf-overrides '${YARN_HF_OVERRIDES}' \\"
            echo "      --max-model-len 1010000"
        fi

    else
        # ==================== TP (Tensor Parallel) 模式 ====================
        echo "  ┌──────────────────────────────────────────────────────────┐"
        echo "  │ 模式: Tensor Parallel (TP=8)                             │"
        echo "  │ 多节点时自动加 Pipeline Parallel                          │"
        echo "  └──────────────────────────────────────────────────────────┘"

        echo ""
        echo "  === 单节点 (8 GPU, TP=8) ==="
        echo ""
        echo "    vllm serve ${EFFECTIVE_MODEL} \\"
        echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
        echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
        echo "      --tensor-parallel-size 8 \\"
        echo "      --distributed-executor-backend ray \\"
        echo "      --reasoning-parser qwen3 \\"
        echo "      --max-model-len 262144 \\"
        echo "      --gpu-memory-utilization 0.93 \\"
        echo "      --enable-prefix-caching"

        echo ""
        echo "  === 多节点 (2 节点 x 8 GPU, TP=8 PP=2) ==="
        echo ""
        echo "    vllm serve ${EFFECTIVE_MODEL} \\"
        echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
        echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
        echo "      --tensor-parallel-size 8 \\"
        echo "      --pipeline-parallel-size 2 \\"
        echo "      --distributed-executor-backend ray \\"
        echo "      --reasoning-parser qwen3 \\"
        echo "      --max-model-len 262144 \\"
        echo "      --gpu-memory-utilization 0.93 \\"
        echo "      --max-num-seqs 4 \\"
        echo "      --swap-space 32 \\"
        echo "      --enable-prefix-caching"

        if [[ "$ENABLE_MULTIMODAL" == "1" ]]; then
            echo ""
            echo "  === 多模态 (TP 模式) ==="
            echo ""
            echo "    vllm serve ${EFFECTIVE_MODEL} \\"
            echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
            echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
            echo "      --tensor-parallel-size 8 \\"
            echo "      --distributed-executor-backend ray \\"
            echo "      --reasoning-parser qwen3 \\"
            echo "      --max-model-len 262144 \\"
            echo "      --gpu-memory-utilization 0.93 \\"
            echo "      --mm-encoder-tp-mode data \\"
            echo "      --mm-processor-cache-type shm \\"
            echo "      --enable-prefix-caching"
        fi

        if [[ "$ENABLE_1M_CONTEXT" == "1" ]]; then
            echo ""
            echo "  === 1M 长上下文 (TP + YaRN) ==="
            echo ""
            echo "    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${EFFECTIVE_MODEL} \\"
            echo "      --host 0.0.0.0 --port ${VLLM_PORT} \\"
            echo "      --served-model-name ${SERVED_MODEL_NAME} \\"
            echo "      --tensor-parallel-size 8 \\"
            echo "      --distributed-executor-backend ray \\"
            echo "      --reasoning-parser qwen3 \\"
            echo "      --gpu-memory-utilization 0.93 \\"
            echo "      --enable-prefix-caching \\"
            echo "      --hf-overrides '${YARN_HF_OVERRIDES}' \\"
            echo "      --max-model-len 1010000"
        fi
    fi

    echo ""
    echo "  === 低延迟模式 (MTP 投机解码, 任意模式可叠加) ==="
    echo ""
    echo "    # 在上述任意命令后追加:"
    echo "    --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}'"

    echo ""
    echo "  === 工具调用 (任意模式可叠加) ==="
    echo ""
    echo "    # 在上述任意命令后追加:"
    echo "    --enable-auto-tool-choice --tool-call-parser qwen3_coder"
fi

# ======================== 完成 ========================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
if [[ "$ROLE" == "head" ]]; then
    echo "║  Head 节点 Setup 完成!                                      ║"
    echo "║                                                            ║"
    if [[ "$DEPLOY_MODE" == "ep" ]]; then
        echo "║  EP 模式: 单节点即可启动, 无需 worker 节点.                   ║"
    else
        echo "║  TP 模式: 单节点直接启动, 或等 worker 就绪后多节点启动.         ║"
    fi
    echo "║  参考上方命令启动 vLLM 服务.                                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
else
    echo "║  Worker 节点 Setup 完成! 已加入 Ray 集群.                    ║"
    echo "║  请确认头节点和其他 worker 也已完成 setup.                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi

echo ""
echo "  测试服务 (启动后):"
echo "    curl http://${HEAD_IP}:${VLLM_PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 256}'"
echo ""
