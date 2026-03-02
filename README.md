# qwen35-vllm-ray-deploy

使用 **Ray + vLLM** 在多节点 GPU 集群上部署大模型，支持 **Qwen3.5** 和 **GLM-5** 等模型：

- OpenAI 兼容接口（`/v1/chat/completions`）
- 多机多卡分布式推理（TP + PP）
- 多模态输入（图文）

## 1. 项目结构

```text
qwen35-vllm-ray-deploy/
├── config.sh               # Qwen3.5-35B 统一参数配置
├── install_env.sh          # Qwen3.5-35B 每节点安装 Python/Ray/vLLM 依赖
├── quick_start.sh          # Qwen3.5-35B 自动探测 IP/NIC + 一键启动（推荐）
├── ray_cluster.sh          # Ray 集群管理：head/worker/stop/status
├── start_vllm_qwen35.sh    # 在 head 节点启动 vLLM (Qwen3.5-35B)
├── setup_qwen35_397b.sh    # Qwen3.5-397B 一体化环境配置脚本（含 Ray 集群）
├── setup_glm5.sh           # GLM-5 一体化环境配置脚本（含 Ray 集群）
├── test_mm_client.py       # 多模态请求样例
└── README.md
```

## 2. 前置要求

- Linux x86_64
- NVIDIA 驱动 + CUDA 环境可用（`nvidia-smi` 正常）
- 两台机器网络互通（建议内网万兆或更高）
- Python 3.10+
- 每台 8 张 GPU（总计 16 张）

## 3. 关键默认配置

在 `config.sh` 中默认设置：

- `MODEL_ID=Qwen/Qwen3.5-35B-A3B`
- `TP_SIZE=8`
- `PP_SIZE=2`
- `MAX_MODEL_LEN=1010000`
- `ENABLE_1M_CONTEXT=1`
- `ENABLE_MULTIMODAL=1`
- `KV_CACHE_DTYPE=fp8`
- `MAX_NUM_SEQS=2`

> 注意：开源模型的超长上下文来自 YaRN 扩展，吞吐与并发会受显存和 KV cache 明显影响。

## 4. 一键启动（推荐，无需手工改配置）

在两台机器都执行同一条命令（把 IP 改成你的）：

```bash
cd qwen35-vllm-ray-deploy
bash quick_start.sh --cluster-ips "10.0.0.1,10.0.0.2" --install --start-vllm
```

行为说明：

- 自动探测本机 `NODE_IP` 和 `NIC_NAME`
- 根据 `--cluster-ips` 自动判断角色（第一个 IP 为 head，其他为 worker）
- 自动启动 Ray（head/worker）
- 在 head 上自动等待集群就绪后启动 vLLM

常用参数：

- `--wait-seconds 900`：调大等待 worker 的超时时间
- `--no-start-vllm`：只启动 Ray，不拉起 vLLM
- `--head-ip 10.0.0.1`：显式指定 head IP（覆盖自动推断）
- `--role head|worker`：手动指定角色（覆盖自动判断）
- `NODE_IP=... NIC_NAME=...`：当自动探测失败时可直接在命令前覆盖

## 5. 分步启动（可选，手动模式）

### 5.1 每个节点都执行一次安装

```bash
cd qwen35-vllm-ray-deploy
bash install_env.sh
```

### 5.2 在节点1启动 Ray Head

```bash
NODE_ROLE=head NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 NIC_NAME=eth0 bash ray_cluster.sh head
```

### 5.3 在节点2启动 Ray Worker

```bash
NODE_ROLE=worker NODE_IP=10.0.0.2 HEAD_NODE_IP=10.0.0.1 NIC_NAME=eth0 bash ray_cluster.sh worker
```

### 5.4 回到节点1，检查集群并启动 vLLM

```bash
NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 bash ray_cluster.sh status
NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 bash start_vllm_qwen35.sh
tail -f ~/qwen35_vllm_ray/logs/vllm_server.log
```

## 6. 调用测试

### 6.1 多模态 Python 示例

编辑 `test_mm_client.py` 的 `HEAD_IP`，然后执行：

```bash
source ~/qwen35_vllm_ray/.venv/bin/activate
python test_mm_client.py
```

### 6.2 文本请求 cURL 示例

```bash
curl http://10.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-mm-1m",
    "messages": [{"role": "user", "content": "介绍一下你自己"}],
    "max_tokens": 256
  }'
```

## 7. 常见调优建议

- OOM 或频繁重试：
  - 降低 `MAX_NUM_SEQS`（先降到 1）
  - 降低 `GPU_MEMORY_UTILIZATION`（如 0.9）
  - 将 `KV_CACHE_DTYPE` 从 `fp8` 改为 `auto` 做对比
- 首 token 慢：
  - 降低最大输入长度
  - 减少每请求图片数（`LIMIT_MM_JSON`）
- 多机通信异常：
  - 检查 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
  - 检查节点间端口和防火墙规则

## 8. 安全建议

如果你允许模型直接抓远程图片 URL，建议在 `config.sh` 配置：

- `ALLOWED_MEDIA_DOMAINS="your-domain-a your-domain-b"`

脚本会自动加上 `--allowed-media-domains` 并禁用重定向，降低 SSRF 风险。

---

## Qwen3.5-397B-A17B 部署

### 模型概述

[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) 是通义千问 3.5 旗舰级 MoE 模型：

- **总参数**: 397B，**激活参数**: 17B（每 token 路由 10 + 1 shared experts）
- **架构**: Gated DeltaNet（线性注意力）+ Gated Attention（全注意力）混合架构，60 层，512 experts
- **原生上下文**: 262,144 tokens，可通过 YaRN 扩展至 ~1M tokens
- **多模态**: 支持图文理解（视觉编码器可选加载）
- **推荐**: 使用 [FP8 量化版](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) 提升推理效率

### 硬件要求

| 配置 | GPU | 显存 | 说明 |
|------|-----|------|------|
| **单节点 (推荐)** | 8 x H200 (141GB) | ~1128GB | FP8 权重 ~400GB，充足 KV cache 空间 |
| **单节点 (紧凑)** | 8 x H100/A100 (80GB) | ~640GB | FP8 勉强可用，需限制上下文长度 |
| **多节点** | 2+ 节点 x 8 GPU | ≥1280GB | TP=8 + PP=2，显存更充裕 |

### 部署模式

| 模式 | 命令 | 适用场景 |
|------|------|----------|
| **EP+DP** (专家并行) | `-dp 8 --enable-expert-parallel` | MoE 高吞吐、高并发，推荐方案 |
| **TP** (张量并行) | `--tensor-parallel-size 8` | 经典稳定方案，多节点加 PP |

### 环境配置

使用 `setup_qwen35_397b.sh` 一键配置环境并组建 Ray 集群。脚本包含：uv 安装、Python 虚拟环境、vLLM nightly、transformers (git main)、NCCL 配置、Ray 集群启动。

```bash
# 单节点 (8xH200, EP 模式高吞吐)
bash setup_qwen35_397b.sh head 10.0.0.1 --deploy-mode ep

# 多节点 TP 模式
# 节点1 (头节点)
bash setup_qwen35_397b.sh head 10.0.0.1

# 节点2 (工作节点)
bash setup_qwen35_397b.sh worker 10.0.0.1

# 启用 1M 长上下文 + 多模态
bash setup_qwen35_397b.sh head 10.0.0.1 --enable-1m-context --multimodal

# 使用本地已下载的模型
bash setup_qwen35_397b.sh head 10.0.0.1 --model-path /data/Qwen3.5-397B-A17B-FP8
```

### 配置项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--deploy-mode` | `tp` | 部署模式：`tp` (张量并行) 或 `ep` (专家并行) |
| `--cuda` | `cu121` | CUDA 变体：`cu121` / `cu124` / `cu130` |
| `--nic` | 自动检测 | 节点间通信网卡名 |
| `--model-path` | HF 下载 | 本地模型路径 |
| `--enable-1m-context` | 关闭 | 启用 YaRN 1M 长上下文扩展 |
| `--multimodal` | 关闭 | 启用多模态 (视觉编码器) |
| `--language-model-only` | 关闭 | 仅语言模型，省显存 |
| `--venv-dir` | `/root/qwen397b-env` | 虚拟环境路径 |
| `--python` | `3.10` | Python 版本 |

### 启动 vLLM 服务

脚本执行完成后会输出对应部署模式的 vLLM 启动命令。以下是常用示例：

**EP 模式 (高吞吐文本)**:

```bash
source /root/qwen397b-env/bin/activate

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3.5-397b \
    -dp 8 \
    --enable-expert-parallel \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
```

**TP 模式 (单节点)**:

```bash
source /root/qwen397b-env/bin/activate

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3.5-397b \
    --tensor-parallel-size 8 \
    --distributed-executor-backend ray \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.93 \
    --enable-prefix-caching
```

**TP 模式 (多节点, TP=8 PP=2)**:

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3.5-397b \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.93 \
    --max-num-seqs 4 \
    --swap-space 32 \
    --enable-prefix-caching
```

**低延迟 (MTP 投机解码, 可叠加任意模式)**:

```bash
# 在上述命令后追加:
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

**工具调用 (可叠加)**:

```bash
# 在上述命令后追加:
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### 调用示例

```bash
curl http://<HEAD_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b",
    "messages": [{"role": "user", "content": "介绍一下你自己"}],
    "max_tokens": 256
  }'
```

### 与 Qwen3.5-35B 的区别

| | Qwen3.5-35B-A3B | Qwen3.5-397B-A17B |
|--|--|--|
| 总参数 | 35B | 397B |
| 激活参数 | 3B | 17B |
| 最少 GPU | 2x8 GPU (TP=8 PP=2) | 8 GPU (FP8, TP=8) |
| 推荐量化 | 无 (模型较小) | FP8 (必须) |
| 专家并行 | 不支持 | 支持 (推荐) |
| MTP 投机 | 不支持 | 支持 |

---

## GLM-5 部署

### 硬件要求

- 3 节点 x 8 x NVIDIA L20Z (80GB)
- Ubuntu Noble, CUDA 13.0
- 节点间网络互通（TCP Socket）

### 环境配置

使用 `setup_glm5.sh` 在每个节点配置环境并组建 Ray 集群。脚本包含：uv 安装、Python 3.12 虚拟环境、vLLM nightly (cu130)、transformers (git main)、NCCL 配置、vLLM 源码补丁、Ray 集群启动。

```bash
# 节点1 (头节点, 例如 10.2.1.11)
bash setup_glm5.sh head 10.2.1.11

# 节点2 (工作节点, 例如 10.2.1.22)
bash setup_glm5.sh worker 10.2.1.11

# 节点3 (工作节点, 例如 10.2.5.21)
bash setup_glm5.sh worker 10.2.1.11
```

### 配置项

脚本顶部可修改以下配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VENV_DIR` | `/root/vllm-env` | 虚拟环境路径 |
| `PYTHON_VERSION` | `3.12` | Python 版本 |
| `CUDA_VARIANT` | `cu130` | CUDA 变体 |
| `NETWORK_IFACE` | `intranet_bond` | 节点间通信网卡名 |
| `RAY_PORT` | `6379` | Ray head 端口 |
| `MODEL_PATH` | `/root/GLM` | 模型路径 |

### 启动 vLLM 服务

所有节点 setup 完成后，在**头节点**执行：

```bash
source /root/vllm-env/bin/activate

vllm serve /root/GLM \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 --pipeline-parallel-size 3 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.92 --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --enable-auto-tool-choice --served-model-name glm-5 \
    --trust-remote-code --enable-prefix-caching --enable-chunked-prefill
```

### 参数说明

| 参数 | 值 | 说明 |
|------|------|------|
| `--tensor-parallel-size` | `8` | 每节点 8 GPU 做张量并行 |
| `--pipeline-parallel-size` | `3` | 3 节点做流水线并行 |
| `--distributed-executor-backend` | `ray` | 使用 Ray 做分布式调度 |
| `--gpu-memory-utilization` | `0.92` | GPU 显存使用率 |
| `--max-model-len` | `65536` | 最大序列长度 |
| `--max-num-batched-tokens` | `8192` | 每批最大 token 数 |
| `--tool-call-parser` | `glm47` | GLM 工具调用解析器 |
| `--reasoning-parser` | `glm45` | GLM 推理解析器 |
| `--enable-auto-tool-choice` | - | 启用自动工具选择 |
| `--served-model-name` | `glm-5` | 对外服务的模型名 |
| `--trust-remote-code` | - | 信任模型自定义代码 |
| `--enable-prefix-caching` | - | 启用前缀缓存 |
| `--enable-chunked-prefill` | - | 启用分块预填充 |

### 调用示例

```bash
curl http://<HEAD_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "介绍一下你自己"}],
    "max_tokens": 256
  }'
```

### vLLM 补丁说明

GLM-5 使用 Dynamic Sparse Attention (DSA)，模型层包含 `self_attn.indexer.k_cache` 等组件。Pipeline Parallel 下每个 rank 只持有部分层，`get_layers_from_vllm_config()` 直接访问 `forward_context[layer_name]` 会触发 `KeyError`。`setup_glm5.sh` 会自动打补丁添加 key 存在性检查。

---

## 9. 参考文档

- [Qwen3.5-35B-A3B (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-397B-A17B (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Qwen3.5-397B-A17B-FP8 (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- [QwenLM/Qwen3.5 (GitHub)](https://github.com/QwenLM/Qwen3.5)
- [vLLM Qwen3.5 Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Distributed Serving](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)
- [vLLM CLI Serve](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

## 10. 免责声明

本项目默认配置用于快速落地与验证。生产环境建议补充：

- systemd 托管
- 健康检查与自动重启
- 日志轮转
- 请求限流与鉴权
- 监控告警（GPU、吞吐、错误率、延迟）
