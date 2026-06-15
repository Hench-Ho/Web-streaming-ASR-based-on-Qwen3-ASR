# Qwen3-ASR - 实时流式语音识别

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3asr">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.21337">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-ASR">Hugging Face Demo</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-ASR">ModelScope Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-speech-recognition">API</a>
</p>

本项目基于阿里巴巴通义千问团队开源的 **Qwen3-ASR** 系列模型，提供了一套完整的实时流式语音识别方案。在原有模型基础上，新增了基于 WebSocket 的 Web 端流式语音识别服务，集成了 **VAD 语音活动检测**、**说话人日志（Speaker Diarization）** 和 **流式 ASR 转写** 三大核心能力。

---

## ✨ 核心特性

### 🎤 实时流式语音识别（Web Streaming ASR）

基于 `examples/web_streaming_asr/server.py` 构建，提供完整的浏览器端实时语音识别体验：

| 功能模块 | 技术方案 | 说明 |
|---------|---------|------|
| **VAD 语音活动检测** | FunASR DynamicStreamingVAD + fsmn-vad | 实时检测语音起止，自动分段 |
| **流式 ASR 转写** | Qwen3-ASR vLLM 后端 | 支持 52 种语言和方言的实时转写 |
| **说话人识别** | cam_plus + ClusterBackend 增量聚类 | 多人对话场景下的说话人分离 |
| **文本后处理** | 幻觉检测 + 文本清洗 | 自动移除重复模式和时间戳等伪影 |
| **Web 前端** | WebSocket + Web Audio API | 浏览器麦克风采集，实时推送识别结果 |
| **HTTPS/WSS** | 自签名证书自动生成 | 远程访问时浏览器麦克风权限适配 |

### 🧠 基础 ASR 能力（继承自 Qwen3-ASR）

- **多语言支持**：52 种语言和方言（30 种语言 + 22 种中文方言）
- **双模型规格**：0.6B（轻量高速）和 1.7B（旗舰精度）
- **推理模式**：支持离线批量推理和流式推理
- **时间戳输出**：集成 Qwen3-ForcedAligner 可实现词/字级别时间戳
- **vLLM 后端**：高性能批量推理，128 并发下可达 2000 倍吞吐量

---

## 📦 环境配置

### 基础环境

推荐使用 Python 3.12 的全新隔离环境：

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

### 安装 qwen-asr 包

```bash
# 基础安装（Transformers 后端）
pip install -U qwen-asr

# vLLM 后端（推荐，支持流式推理和更高性能）
pip install -U qwen-asr[vllm]
```

### 安装 Web 流式服务额外依赖

```bash
pip install funasr torch fastapi uvicorn soundfile librosa
```

### 可选：FlashAttention 2

推荐安装以降低 GPU 显存占用并加速推理：

```bash
pip install -U flash-attn --no-build-isolation
```

---

## 🚀 Web 实时流式语音识别

### 启动服务

```bash
cd examples/web_streaming_asr
python server.py \
    --asr-model /path/to/Qwen3-ASR-0.6B \
    --host 0.0.0.0 \
    --port 8765
```

### 完整参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--asr-model` | str | **必填** | Qwen3-ASR 模型路径（本地或 HuggingFace 模型名） |
| `--device` | str | `cuda:0` | 推理设备（cuda:0 / cpu） |
| `--gpu-memory-utilization` | float | `0.9` | vLLM GPU 显存利用率 |
| `--max-new-tokens` | int | `256` | 最大生成 token 数 |
| `--streaming-step-ms` | int | `500` | ASR 流式步长（毫秒） |
| `--unfixed-chunk-num` | int | `2` | 不固定前缀的 chunk 数量 |
| `--unfixed-token-num` | int | `5` | 前缀回退 token 数 |
| `--chunk-size-sec` | float | `2.0` | ASR chunk 大小（秒） |
| `--min-speech-dur-ms` | int | `300` | 最短语音段（毫秒，过滤噪音） |
| `--host` | str | `0.0.0.0` | 服务监听地址 |
| `--port` | int | `8765` | 服务监听端口 |
| `--ssl` | flag | `False` | 启用 HTTPS/WSS（远程访问必须开启） |
| `--verbose` / `-v` | flag | `False` | 详细日志输出 |

### 使用示例

**本地开发（HTTP）：**
```bash
python server.py \
    --asr-model Qwen/Qwen3-ASR-0.6B \
    --host 127.0.0.1 \
    --port 8765
```
打开浏览器访问 `http://127.0.0.1:8765`，点击「连接服务器」→「开始录音」即可。

**远程部署（HTTPS，浏览器麦克风必需）：**
```bash
python server.py \
    --asr-model /path/to/Qwen3-ASR-1.7B \
    --host 0.0.0.0 \
    --port 8765 \
    --ssl \
    --gpu-memory-utilization 0.8
```
服务启动时会自动生成自签名 SSL 证书，浏览器可能提示安全警告，点击「高级」→「继续访问」即可。

**多 GPU 指定：**
```bash
python server.py \
    --asr-model Qwen/Qwen3-ASR-1.7B \
    --device cuda:1 \
    --host 0.0.0.0 \
    --port 8765
```

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      浏览器前端 (index.html)                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ 麦克风采集 │ -> │ 重采样 16kHz  │ -> │ Float32 PCM 发送  │  │
│  │ getUserMedia│    │ linear interp│    │ WebSocket binary │  │
│  └──────────┘    └──────────────┘    └───────┬──────────┘  │
└─────────────────────────────────────────────┼───────────────┘
                                              │ WebSocket
┌─────────────────────────────────────────────┼───────────────┐
│                   服务端 (server.py)         ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               RealtimeASRSession                      │   │
│  │  ┌─────────┐   ┌──────────────┐   ┌───────────────┐  │   │
│  │  │ VAD     │-> │ 语音段切分    │-> │ Qwen3-ASR     │  │   │
│  │  │Dynamic  │   │(min 300ms)   │   │ vLLM 流式推理  │  │   │
│  │  │Streaming│   └──────────────┘   └───────┬───────┘  │   │
│  │  │VAD      │                              │          │   │
│  │  └─────────┘                              ▼          │   │
│  │                              ┌───────────────────┐   │   │
│  │                              │ HybridSpeaker     │   │   │
│  │                              │ Tracker           │   │   │
│  │                              │ cam_plus +        │   │   │
│  │                              │ ClusterBackend    │   │   │
│  │                              └───────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│            ┌────────────────────────────┐                    │
│            │  文本清洗 + 幻觉检测        │                    │
│            │  _clean_asr_text()         │                    │
│            │  detect_and_fix_hallucination()                │
│            └────────────┬───────────────┘                    │
│                         │                                    │
│                         ▼                                    │
│             WebSocket → JSON 推送至前端                       │
│  {type:"partial"|"final", speaker_id, text, language, ...}   │
└──────────────────────────────────────────────────────────────┘
```

### WebSocket 消息协议

服务端推送的 JSON 消息类型：

| 类型 | 说明 | 示例字段 |
|------|------|---------|
| `partial` | 流式中间结果（文本持续更新） | `type`, `text`, `speaker_id`, `language`, `start_sec`, `end_sec`, `segment_id` |
| `final` | 语音段最终结果 | `type`, `text`, `speaker_id`, `language`, `start_sec`, `end_sec`, `segment_id` |
| `status` | 状态消息（连接状态、统计信息） | `type`, `text` |
| `error` | 错误消息 | `type`, `text`, `segment_id` |

客户端发送的控制命令：

| 命令 | 说明 |
|------|------|
| `{"type": "stop"}` | 停止录音，触发缓冲刷新和最终说话人重聚类 |
| `{"type": "reset"}` | 重置会话状态 |
| `{"type": "ping"}` | 心跳检测，服务端回复 `pong` |
| `{"type": "get_stats"}` | 获取会话统计（片段数、说话人数、总时长等） |

---

## 📖 Python 包使用

### 快速离线推理（Transformers 后端）

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None,  # 自动语言检测
)

print(results[0].language)
print(results[0].text)
```

### vLLM 后端批量推理

```python
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_new_tokens=4096,
    )

    results = model.transcribe(
        audio=[
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"],
    )

    for r in results:
        print(r.language, r.text)
```

### 流式推理

```python
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.9,
    )

    state = model.init_streaming_state(
        context="",
        language=None,
        chunk_size_sec=2.0,
    )

    # 逐步喂入音频块
    for audio_chunk in audio_chunks:
        model.streaming_transcribe(audio_chunk, state)
        if state.text:
            print(state.text)  # 实时输出中间结果

    # 完成流式转写
    model.finish_streaming_transcribe(state)
    print("最终结果:", state.text)
```

### 语音对齐（Forced Aligner）

```python
import torch
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.align(
    audio="path/to/audio.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)

print(results[0][0].text, results[0][0].start_time, results[0][0].end_time)
```

---

## 🌐 vLLM 服务部署

### 启动 vLLM 服务

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8000
```

### 客户端请求

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=300)
print(response.json()['choices'][0]['message']['content'])
```

---

## 🎯 Gradio Web Demo

```bash
# Transformers 后端
qwen-asr-demo \
    --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
    --backend transformers \
    --cuda-visible-devices 0 \
    --ip 0.0.0.0 --port 8000

# vLLM 后端
qwen-asr-demo \
    --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
    --backend vllm \
    --cuda-visible-devices 0 \
    --backend-kwargs '{"gpu_memory_utilization":0.7,"max_new_tokens":2048}' \
    --ip 0.0.0.0 --port 8000
```

---

## 🐳 Docker

```bash
docker run --gpus all --name qwen3-asr \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8000:80 \
    --mount type=bind,source=/path/to/workspace,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest
```

---

## 📥 模型下载

```bash
# ModelScope（国内推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B

# Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
```

---

## 📊 项目结构

```
Qwen3-ASR-main/
├── qwen_asr/                    # 核心 Python 包
│   ├── __init__.py              # 包入口，导出 Qwen3ASRModel
│   ├── inference/
│   │   ├── qwen3_asr.py         # ASR 推理主类
│   │   ├── qwen3_forced_aligner.py  # 强制对齐器
│   │   ├── evaluation.py        # 评测工具
│   │   └── utils.py             # 工具函数（parse_asr_output 等）
│   ├── core/
│   │   ├── transformers_backend/ # Transformers 后端实现
│   │   └── vllm_backend/        # vLLM 后端实现
│   └── cli/
│       ├── demo.py              # Gradio Demo 入口
│       ├── demo_streaming.py    # Flask 流式 Demo 入口
│       └── serve.py             # vLLM 服务入口
├── examples/
│   ├── web_streaming_asr/       # 🌟 Web 实时流式 ASR
│   │   ├── server.py            # FastAPI WebSocket 服务端
│   │   └── static/
│   │       └── index.html       # 浏览器前端页面
│   ├── example_qwen3_asr_transformers.py
│   ├── example_qwen3_asr_vllm.py
│   ├── example_qwen3_asr_vllm_streaming.py
│   ├── example_qwen3_forced_aligner.py
│   └── example_qwen3_asr_streaming_with_speaker.py
├── finetuning/                  # 微调脚本
│   └── qwen3_asr_sft.py
├── docker/                      # Docker 配置
│   └── Dockerfile-qwen3-asr-cu128
├── assets/                      # 资源文件
│   └── Qwen3_ASR.pdf
├── pyproject.toml               # 项目配置
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## 📝 引用

```BibTeX
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang,
          Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```

---

## 📄 许可证

本项目基于 Apache 2.0 许可证开源，详见 [LICENSE](LICENSE)。

原始项目：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen3-ASR&type=Date)](https://star-history.com/#QwenLM/Qwen3-ASR&Date)

<br>
