# qwen35-vllm-ray-deploy

使用 **Ray + vLLM** 在 **2 个节点（每节点 8 x 80GB GPU）** 上部署 **Qwen3.5**，默认启用：

- 1M 上下文（通过 YaRN 扩展参数）
- 多模态输入（图文）
- OpenAI 兼容接口（`/v1/chat/completions`）

## 1. 项目结构

```text
qwen35-vllm-ray-deploy/
├── config.sh               # 统一参数配置
├── install_env.sh          # 每个节点安装 Python/Ray/vLLM 依赖
├── ray_cluster.sh          # Ray 集群管理：head/worker/stop/status
├── start_vllm_qwen35.sh    # 在 head 节点启动 vLLM
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

## 4. 快速开始

### 4.1 每个节点都执行一次安装

```bash
cd qwen35-vllm-ray-deploy
bash install_env.sh
```

先编辑 `config.sh`，至少修改：

- `HEAD_NODE_IP`
- `NIC_NAME`

### 4.2 在节点1启动 Ray Head

```bash
NODE_ROLE=head NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 NIC_NAME=eth0 bash ray_cluster.sh head
```

### 4.3 在节点2启动 Ray Worker

```bash
NODE_ROLE=worker NODE_IP=10.0.0.2 HEAD_NODE_IP=10.0.0.1 NIC_NAME=eth0 bash ray_cluster.sh worker
```

### 4.4 回到节点1，检查集群并启动 vLLM

```bash
NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 bash ray_cluster.sh status
NODE_IP=10.0.0.1 HEAD_NODE_IP=10.0.0.1 bash start_vllm_qwen35.sh

tail -f ~/qwen35_vllm_ray/logs/vllm_server.log
```

## 5. 调用测试

### 5.1 多模态 Python 示例

编辑 `test_mm_client.py` 的 `HEAD_IP`，然后执行：

```bash
source ~/qwen35_vllm_ray/.venv/bin/activate
python test_mm_client.py
```

### 5.2 文本请求 cURL 示例

```bash
curl http://10.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-mm-1m",
    "messages": [{"role": "user", "content": "介绍一下你自己"}],
    "max_tokens": 256
  }'
```

## 6. 常见调优建议

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

## 7. 安全建议

如果你允许模型直接抓远程图片 URL，建议在 `config.sh` 配置：

- `ALLOWED_MEDIA_DOMAINS="your-domain-a your-domain-b"`

脚本会自动加上 `--allowed-media-domains` 并禁用重定向，降低 SSRF 风险。

## 8. 参考文档

- [Qwen3.5-35B-A3B (Hugging Face)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [vLLM Distributed Serving](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)
- [vLLM Qwen Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM CLI Serve](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

## 9. 免责声明

本项目默认配置用于快速落地与验证。生产环境建议补充：

- systemd 托管
- 健康检查与自动重启
- 日志轮转
- 请求限流与鉴权
- 监控告警（GPU、吞吐、错误率、延迟）
