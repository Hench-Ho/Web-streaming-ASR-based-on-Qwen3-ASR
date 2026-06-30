# Qwen3-ASR - 实时流式语音识别

本项目基于阿里巴巴通义千问团队开源的 **Qwen3-ASR** 系列模型，提供了一套完整的实时流式语音识别方案。在原有模型基础上，新增了基于 WebSocket 的 Web 端流式语音识别服务，集成了 VAD 语音活动检测、说话人日志和流式 ASR 转写三大核心能力。

---


### 实时流式语音识别

基于 `examples/web_streaming_asr/server.py` 构建，提供完整的浏览器端实时语音识别体验：

| 功能模块 | 技术方案 | 说明 |
|---------|---------|------|
| **VAD 语音活动检测** | FunASR DynamicStreamingVAD + fsmn-vad | 实时检测语音起止，自动分段 |
| **流式 ASR 转写** | Qwen3-ASR vLLM 后端 | 支持 52 种语言和方言的实时转写 |
| **说话人识别** | cam_plus + ClusterBackend 增量聚类 | 多人对话场景下的说话人分离 |
| **文本后处理** | 幻觉检测 + 文本清洗 | 自动移除重复模式和时间戳等伪影 |
| **Web 前端** | WebSocket + Web Audio API | 浏览器麦克风采集，实时推送识别结果 |
| **HTTPS/WSS** | 自签名证书自动生成 | 远程访问时浏览器麦克风权限适配 |

---

## 环境配置

### 基础环境

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

### 安装 qwen-asr 包

```bash
# 基础安装（Transformers 后端）
pip install -U qwen-asr

# vLLM 后端（支持流式推理和更高性能）
pip install -U qwen-asr[vllm]
```

### 安装 Web 流式服务额外依赖

```bash
pip install funasr torch fastapi uvicorn soundfile librosa
```
---

## Web 实时流式语音识别

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


## 模型下载

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

## 引用

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

原始项目：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
---

