# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Web 实时流式语音识别服务端（重构版）

功能:
  - 接收浏览器麦克风采集的音频流 (WebSocket)
  - 实时 VAD 语音活动检测 (FunASR DynamicStreamingVAD + fsmn-vad)
  - 说话人识别 (cam_plus + ClusterBackend 增量聚类)
  - 流式 ASR 转写 (Qwen3-ASR vLLM 后端)
  - 实时推送识别结果到前端

启动方式:
  python server.py \
      --asr-model /path/to/Qwen3-ASR-0.6B \
      --host 0.0.0.0 --port 8765

依赖安装:
  pip install qwen-asr funasr torch fastapi uvicorn soundfile librosa
"""

import json
import logging
import os
import argparse
import re as re_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import regex
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# ============================================================================
# 文本清洗 & 幻觉检测工具 (来自 serve_realtime_ws.py)
# ============================================================================

def _clean_asr_text(text: str) -> str:
    """移除 vLLM 输出中的时间戳标签和其他伪影。"""
    text = re_module.sub(r'<[^>]*>', '', text)
    text = re_module.sub(r'\[.*?\]', '', text)
    text = re_module.sub(r'[Ｏ\[\]&＆|｜]', '', text)
    text = re_module.sub(r'/sil|endofbreak|FFFF', '', text)
    text = re_module.sub(r'\s+', ' ', text)
    return text.strip()


def detect_and_fix_hallucination(text: str, max_ngram_length: int = 12, max_occurrences: int = 3):
    """检测重复模式（幻觉）并截断只保留一次出现。"""
    if not text or len(text) < max_ngram_length * 2:
        return text, False

    cleaned = regex.sub(r'\p{P}+', '', text)

    # 检测重复单词
    word_pattern = rf'(?<!\S)(?!\d+$)(\w+)(?:\s+\1){{{max_occurrences - 1},}}(?!\S)'
    if regex.search(word_pattern, cleaned, regex.IGNORECASE):
        match = regex.search(word_pattern, cleaned, regex.IGNORECASE)
        repeated = match.group(1)
        pos = text.find(repeated)
        if pos >= 0:
            end_pos = text.find(repeated, pos + len(repeated))
            if end_pos >= 0:
                return text[:end_pos + len(repeated)], True
        return text[:len(text)//2], True

    # 检测重复字符序列
    for length in range(1, max_ngram_length):
        pattern = rf'(?<!\d)(\S{{{length}}})\1{{{max_occurrences - 1},}}(?!\d)'
        combined = rf'(?=.*\D){pattern}'
        match = regex.search(combined, cleaned)
        if match:
            repeated = match.group(1)
            pos = text.find(repeated)
            if pos >= 0:
                end_pos = text.find(repeated, pos + len(repeated))
                if end_pos >= 0:
                    return text[:end_pos + len(repeated)], True
            return text[:len(text)//2], True

    return text, False


# ============================================================================
# WebSocket 消息数据结构
# ============================================================================

@dataclass
class ASRMessage:
    """推送给前端的识别结果消息"""
    type: str  # "partial" | "final" | "status" | "error"
    speaker_id: int = -1
    text: str = ""
    language: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    timestamp: float = 0.0
    segment_id: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "language": self.language,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "timestamp": round(self.timestamp, 3),
            "segment_id": self.segment_id,
        }, ensure_ascii=False)


# ============================================================================
# 说话人管理器 — HybridSpeakerTracker (来自 serve_realtime_ws.py)
# ============================================================================

class HybridSpeakerTracker:
    """说话人日志：流式 ClusterBackend + 最终重聚类。

    与 serve_realtime_ws.py 中的实现一致：
      - 使用 cam_plus 提取说话人嵌入
      - 使用 sv_chunk 对长段落切片
      - 流式增量聚类，最终全量重聚类
    """

    def __init__(self, spk_model, device: str, threshold: float = 0.6):
        self.spk_model = spk_model
        self.device = device
        self.threshold = threshold
        self.speaker_centers: List[torch.Tensor] = []

        # 延迟导入 FunASR 说话人工具（确保在 FunASR 加载后可用）
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend

        self.sv_chunk = sv_chunk
        self.postprocess = postprocess
        self.distribute_spk = distribute_spk
        self.cluster_backend = ClusterBackend(merge_thr=0.78).to(device)

        # 累积所有嵌入和切片，用于最终重聚类
        self.all_chunks: List[Any] = []
        self.all_embeddings: List[torch.Tensor] = []

        # 显示用说话人 ID 映射
        self.display_map: Dict[int, int] = {}
        self.next_display_id: int = 0

    @torch.no_grad()
    def assign_streaming(self, audio_samples: np.ndarray, seg_start_s: float, seg_end_s: float, sentence: Dict[str, Any]):
        """流式分配说话人 ID。

        Args:
            audio_samples: 该语音段的音频样本 (float32, 16kHz)
            seg_start_s: 段落起始时间（秒）
            seg_end_s: 段落结束时间（秒）
            sentence: 句子字典，会就地添加 "spk" 字段
        """
        vad_seg = [[seg_start_s, seg_end_s, audio_samples]]
        chunks = self.sv_chunk(vad_seg)
        if not chunks:
            sentence["spk"] = self.next_display_id
            self.next_display_id += 1
            return

        self.all_chunks.extend(chunks)
        speech_list = [ch[2] for ch in chunks]

        # 提取说话人嵌入
        spk_res = self.spk_model.generate(input=speech_list, cache={}, is_final=True)
        embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
        self.all_embeddings.append(embs)

        # 增量聚类
        all_embs = torch.cat(self.all_embeddings, dim=0)
        labels = self.cluster_backend(all_embs.cpu(), oracle_num=None)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        all_sorted = sorted(self.all_chunks, key=lambda x: x[0])
        sv_output = self.postprocess(all_sorted, None, labels, all_embs.cpu())

        temp = [{"start": int(seg_start_s * 1000), "end": int(seg_end_s * 1000), "text": sentence.get("text", "")}]
        self.distribute_spk(temp, sv_output)
        raw_spk = temp[0].get("spk", 0)

        if raw_spk not in self.display_map:
            self.display_map[raw_spk] = self.next_display_id
            self.next_display_id += 1
        sentence["spk"] = self.display_map[raw_spk]

    @torch.no_grad()
    def finalize(self, sentences: List[Dict[str, Any]], min_split_s: float = 3.0) -> List[Dict[str, Any]]:
        """最终全量重聚类，修正流式聚类中的错误分配。

        Args:
            sentences: 所有句子列表，每个包含 text/start/end
            min_split_s: 最小切分时长（秒）

        Returns:
            修正说话人后的句子列表
        """
        if not self.all_embeddings or not sentences:
            return sentences

        all_embs = torch.cat(self.all_embeddings, dim=0)
        labels = self.cluster_backend(all_embs.cpu(), oracle_num=None)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        all_sorted = sorted(self.all_chunks, key=lambda x: x[0])
        sv_output = self.postprocess(all_sorted, None, labels, all_embs.cpu())

        for s in sentences:
            s.pop("spk", None)
        self.distribute_spk(sentences, sv_output)

        # 重新分配连续 ID
        id_map: Dict[int, int] = {}
        next_id = 0
        for s in sentences:
            raw = s.get("spk", 0)
            if raw not in id_map:
                id_map[raw] = next_id
                next_id += 1
            s["spk"] = id_map[raw]

        # 尝试切分含多个说话人的长句
        final_sentences: List[Dict[str, Any]] = []
        for s in sentences:
            sub = self._try_split(s, sv_output, id_map, min_split_s)
            final_sentences.extend(sub)

        return final_sentences

    def _try_split(self, sentence: Dict[str, Any], sv_output: List, id_map: Dict[int, int], min_split_s: float) -> List[Dict[str, Any]]:
        """如果句子时间范围内检测到多个说话人，尝试切分。"""
        sent_start = sentence["start"] / 1000.0
        sent_end = sentence["end"] / 1000.0
        text = sentence["text"]

        overlapping = []
        for sv_start, sv_end, sv_spk in sv_output:
            o_start = max(sent_start, sv_start)
            o_end = min(sent_end, sv_end)
            if o_end > o_start:
                mapped_spk = id_map.get(int(sv_spk), int(sv_spk))
                overlapping.append([o_start, o_end, mapped_spk])

        if len(overlapping) <= 1:
            return [sentence]

        # 合并相邻相同说话人的时间段
        filtered = [overlapping[0]]
        for i in range(1, len(overlapping)):
            cur = overlapping[i]
            prev = filtered[-1]
            if cur[2] == prev[2]:
                filtered[-1] = [prev[0], cur[1], prev[2]]
            elif (cur[1] - cur[0]) < min_split_s:
                filtered[-1] = [prev[0], cur[1], prev[2]]
            else:
                filtered.append(cur)

        # 合并过短段到前一段
        merged = [filtered[0]]
        for i in range(1, len(filtered)):
            if (merged[-1][1] - merged[-1][0]) < min_split_s:
                merged[-1] = [merged[-1][0], filtered[i][1], filtered[i][2]]
            else:
                merged.append(filtered[i])
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_split_s:
            merged[-2] = [merged[-2][0], merged[-1][1], merged[-2][2]]
            merged.pop()

        if len(merged) <= 1:
            return [sentence]

        # 按时间比例分配文本
        total_dur = sum(m[1] - m[0] for m in merged)
        sub_sentences = []
        char_pos = 0
        for i, (m_start, m_end, m_spk) in enumerate(merged):
            if i == len(merged) - 1:
                sub_text = text[char_pos:]
            else:
                n_chars = max(1, int(len(text) * (m_end - m_start) / total_dur))
                sub_text = text[char_pos:char_pos + n_chars]
                char_pos += n_chars
            if sub_text.strip():
                sub_sentences.append({
                    "text": sub_text.strip(),
                    "start": int(m_start * 1000),
                    "end": int(m_end * 1000),
                    "spk": m_spk,
                })

        return sub_sentences if sub_sentences else [sentence]

    def reset(self):
        """重置说话人跟踪器状态"""
        self.speaker_centers = []
        self.all_chunks = []
        self.all_embeddings = []
        self.display_map = {}
        self.next_display_id = 0


# ============================================================================
# 实时流式 ASR 会话 (基于 serve_realtime_ws.py 的 RealtimeASRSession，ASR 改为 Qwen3-ASR)
# ============================================================================

class RealtimeASRSession:
    """管理单个 WebSocket 连接的流式 ASR 会话。

    整合: DynamicStreamingVAD → 语音分段 → Qwen3-ASR 流式转写 → HybridSpeakerTracker
    """

    def __init__(
        self,
        asr_model,             # Qwen3ASRModel (vLLM 后端)
        vad,                   # DynamicStreamingVAD 实例
        spk_tracker,           # HybridSpeakerTracker 实例
        # ASR 流式参数
        streaming_step_ms: int = 500,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        max_new_tokens: int = 256,
        # VAD 参数
        min_speech_dur_ms: int = 300,
        # 回调
        on_result: Optional[callable] = None,
    ):
        self.asr_model = asr_model
        self.vad = vad
        self.spk_tracker = spk_tracker

        self.streaming_step_ms = streaming_step_ms
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.chunk_size_sec = chunk_size_sec
        self.max_new_tokens = max_new_tokens
        self.min_speech_dur_ms = min_speech_dur_ms
        self.on_result = on_result

        # 音频缓冲区（float32, 16kHz）
        self.audio_buffer = np.array([], dtype=np.float32)

        # VAD 已喂入的采样点数
        self.vad_fed_samples: int = 0

        # 已确认的句子
        self.locked_sentences: List[Dict[str, Any]] = []

        # 段落计数
        self._segment_counter: int = 0

        # 全局会话时间
        self._global_time: float = 0.0

        # 会话是否活跃
        self.is_active: bool = False

        logger.info("ASR 会话已创建 (VAD=DynamicStreamingVAD, SPK=cam_plus, ASR=Qwen3-ASR)")

    def add_audio(self, pcm_bytes: bytes):
        """添加音频数据（Float32 PCM 字节流，来自浏览器前端）。

        Args:
            pcm_bytes: Float32 格式的 PCM 字节数据（浏览器 Float32Array → ArrayBuffer）
        """
        # 前端发送的是 Float32Array.buffer，直接解析为 float32
        audio_float = np.frombuffer(pcm_bytes, dtype=np.float32).copy()
        if len(audio_float) == 0:
            return
        # 确保是 1D 单声道
        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=-1)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])

    def process_vad(self) -> List[Dict[str, Any]]:
        """将新音频喂入 DynamicStreamingVAD，返回新确认的语音段事件。

        Returns:
            事件列表: [{"type": "speech_segment", "start_ms": int, "end_ms": int}, ...]
        """
        events: List[Dict[str, Any]] = []

        # 取出 VAD 尚未处理的新音频
        new_audio = self.audio_buffer[self.vad_fed_samples:]
        if len(new_audio) == 0:
            return events

        # 喂入 DynamicStreamingVAD
        new_confirmed = self.vad.feed(torch.from_numpy(new_audio).float(), is_final=False)
        self.vad_fed_samples = len(self.audio_buffer)

        for seg in new_confirmed:
            start_ms, end_ms = int(seg[0]), int(seg[1])
            dur_ms = end_ms - start_ms

            # 过滤过短段落
            if dur_ms < self.min_speech_dur_ms:
                logger.debug("VAD: 跳过过短段落 %dms (阈值=%dms)", dur_ms, self.min_speech_dur_ms)
                continue

            events.append({
                "type": "speech_segment",
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

        return events

    async def process_segment(self, start_ms: int, end_ms: int) -> List[ASRMessage]:
        """处理一个已确认的语音段：提取音频 → ASR转写 → 说话人识别。

        Args:
            start_ms: 段落起始时间（毫秒，相对于会话开始）
            end_ms: 段落结束时间（毫秒）

        Returns:
            ASRMessage 列表（partial + final + status）
        """
        self._segment_counter += 1
        seg_id = self._segment_counter

        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        # 从音频缓冲区提取该段的音频
        start_sample = int(start_ms * SAMPLE_RATE / 1000)
        end_sample = min(int(end_ms * SAMPLE_RATE / 1000), len(self.audio_buffer))
        seg_audio = self.audio_buffer[start_sample:end_sample].copy()

        duration = len(seg_audio) / SAMPLE_RATE
        logger.info("📝 处理语音段 #%d (%d-%dms, 时长%.2fs, %d采样点)",
                     seg_id, start_ms, end_ms, duration, len(seg_audio))

        results: List[ASRMessage] = []

        # 发送状态消息
        status = ASRMessage(
            type="status",
            text=f"语音段 {start_sec:.1f}s-{end_sec:.1f}s",
            start_sec=start_sec,
            end_sec=end_sec,
            timestamp=start_sec,
            segment_id=seg_id,
        )
        results.append(status)

        # ---- A. 流式 ASR 转写 (Qwen3-ASR vLLM) ----
        final_text = ""
        final_lang = ""
        last_text = ""

        try:
            # 初始化流式状态
            state = self.asr_model.init_streaming_state(
                context="",
                language=None,
                unfixed_chunk_num=self.unfixed_chunk_num,
                unfixed_token_num=self.unfixed_token_num,
                chunk_size_sec=self.chunk_size_sec,
            )

            # 按步长逐步送入音频
            step_samples = int(round(self.streaming_step_ms / 1000.0 * SAMPLE_RATE))
            pos = 0

            while pos < len(seg_audio):
                chunk = seg_audio[pos: pos + step_samples]
                pos += len(chunk)

                # 流式转写
                self.asr_model.streaming_transcribe(chunk, state)

                # 文本有变化时发送中间结果
                if state.text and state.text != last_text:
                    last_text = state.text
                    partial_msg = ASRMessage(
                        type="partial",
                        speaker_id=-1,  # 说话人 ID 稍后确定
                        text=state.text,
                        language=state.language,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        segment_id=seg_id,
                    )
                    results.append(partial_msg)

            # 完成流式转写
            self.asr_model.finish_streaming_transcribe(state)
            final_text = state.text or ""
            final_lang = state.language or ""

            # 清洗文本
            final_text = _clean_asr_text(final_text)

            # 幻觉检测
            final_text, hallucinated = detect_and_fix_hallucination(final_text)
            if hallucinated:
                logger.warning("  检测到幻觉，已截断")

            logger.info("  ASR结果(seg #%d): %s [%s]", seg_id, final_text[:80], final_lang)

        except Exception as e:
            logger.error("  流式ASR失败: %s", e, exc_info=True)
            error_msg = ASRMessage(
                type="error",
                text=f"ASR失败: {str(e)}",
                speaker_id=-1,
                start_sec=start_sec,
                end_sec=end_sec,
                segment_id=seg_id,
            )
            results.append(error_msg)
            return results

        # ---- B. 说话人识别 ----
        speaker_id = -1
        if self.spk_tracker is not None and final_text.strip():
            try:
                sentence = {"text": final_text, "start": start_ms, "end": end_ms}
                self.spk_tracker.assign_streaming(seg_audio, start_sec, end_sec, sentence)
                speaker_id = sentence.get("spk", -1)
                logger.info("  说话人分配: spk=%d", speaker_id)

                # 更新之前该段的 partial 消息中的 speaker_id
                for msg in results:
                    if msg.type == "partial" and msg.speaker_id == -1:
                        msg.speaker_id = speaker_id
            except Exception as e:
                logger.warning("  说话人处理失败: %s", e)

        # ---- C. 发送最终结果 ----
        if final_text.strip():
            # 保存到已确认句子列表（用于最终重聚类）
            self.locked_sentences.append({
                "text": final_text,
                "start": start_ms,
                "end": end_ms,
                "spk": speaker_id,
            })

            final_msg = ASRMessage(
                type="final",
                speaker_id=speaker_id,
                text=final_text,
                language=final_lang,
                start_sec=start_sec,
                end_sec=end_sec,
                timestamp=end_sec,
                segment_id=seg_id,
            )
            results.append(final_msg)

        return results

    async def flush(self) -> List[ASRMessage]:
        """强制处理缓冲中剩余的音频（会话结束时调用）。

        对 VAD 中尚未完成的语音段进行收尾处理。
        """
        results: List[ASRMessage] = []

        # 将剩余未喂入 VAD 的音频以 is_final=True 喂入
        remaining = self.audio_buffer[self.vad_fed_samples:]
        if len(remaining) > 0:
            final_segs = self.vad.feed(torch.from_numpy(remaining).float(), is_final=True)
            self.vad_fed_samples = len(self.audio_buffer)

            for seg in final_segs:
                start_ms, end_ms = int(seg[0]), int(seg[1])
                dur_ms = end_ms - start_ms
                if dur_ms < self.min_speech_dur_ms:
                    continue
                seg_results = await self.process_segment(start_ms, end_ms)
                results.extend(seg_results)

        # 检查 VAD 中是否有未完成的语音段（current_speech_start 非空）
        if self.vad.current_speech_start is not None:
            end_ms = int(len(self.audio_buffer) * 1000 / SAMPLE_RATE)
            start_ms = int(self.vad.current_speech_start)
            dur_ms = end_ms - start_ms
            if dur_ms >= self.min_speech_dur_ms:
                seg_results = await self.process_segment(start_ms, end_ms)
                results.extend(seg_results)

        # 最终说话人重聚类
        if self.spk_tracker is not None and self.locked_sentences:
            try:
                self.locked_sentences = self.spk_tracker.finalize(self.locked_sentences)
                logger.info("最终说话人重聚类完成，共 %d 句", len(self.locked_sentences))
            except Exception as e:
                logger.warning("重聚类失败: %s", e)

        return results

    def reset(self):
        """重置会话状态"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.vad_fed_samples = 0
        self.vad.reset()
        if self.spk_tracker:
            self.spk_tracker.reset()
        self.locked_sentences = []
        self._segment_counter = 0
        self._global_time = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取运行时统计"""
        duration_sec = len(self.audio_buffer) / SAMPLE_RATE
        return {
            "total_segments": self._segment_counter,
            "total_sentences": len(self.locked_sentences),
            "num_speakers": (len(self.spk_tracker.display_map) if self.spk_tracker else 0),
            "total_duration_sec": round(duration_sec, 1),
            "vad_is_speaking": self.vad.is_speaking,
        }


# ============================================================================
# WebSocket 连接管理
# ============================================================================

class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: List[Any] = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("客户端已连接 (总数: %d)", len(self.active_connections))

    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("客户端已断开 (总数: %d)", len(self.active_connections))

    async def send_json(self, websocket, data: Dict[str, Any]):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning("发送消息失败: %s", e)
            self.disconnect(websocket)

    async def send_text(self, websocket, text: str):
        try:
            await websocket.send_text(text)
        except Exception as e:
            logger.warning("发送消息失败: %s", e)
            self.disconnect(websocket)


# ============================================================================
# 全局模型实例（延迟加载，多连接共享）
# ============================================================================

_asr_model = None         # Qwen3ASRModel 实例
_vad_model = None         # FunASR AutoModel (fsmn-vad)
_spk_model = None         # FunASR AutoModel (cam_plus)


def load_models(asr_model_path: str, device: str, gpu_memory_utilization: float, max_new_tokens: int):
    """加载全局模型（延迟加载，多个 WebSocket 连接共享同一套模型）。

    与 serve_realtime_ws.py 的 load_models() 模式一致：
      - VAD: FunASR AutoModel("fsmn-vad") + DynamicStreamingVAD
      - Speaker: FunASR AutoModel("iic/speech_cam_plus_sv_zh-cn_16k-common")
      - ASR: Qwen3ASRModel.LLM() (替代 AutoModelVLLM)

    Returns:
        (asr_model, vad_model, spk_model)
    """
    global _asr_model, _vad_model, _spk_model

    if _asr_model is None:
        # CUDA_VISIBLE_DEVICES 已在 main() 中设置，目标 GPU 被重映射为 cuda:0
        # FunASR 模型使用 cuda:0（对应物理目标 GPU），CPU 模式保持不变
        funasr_device = "cuda:0" if device.startswith("cuda") else device

        # ---- 1. Qwen3-ASR (vLLM) ----
        logger.info("=" * 50)
        logger.info("1/3 加载 Qwen3-ASR (vLLM): %s", asr_model_path)
        from qwen_asr import Qwen3ASRModel
        _asr_model = Qwen3ASRModel.LLM(
            model=asr_model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=max_new_tokens,
        )
        logger.info("  Qwen3-ASR 模型加载完成")

        # ---- 2. VAD (FunASR fsmn-vad) ----
        logger.info("2/3 加载 FunASR VAD: fsmn-vad")
        from funasr import AutoModel as FunASRAutoModel
        _vad_model = FunASRAutoModel(
            model="/home/hhc/ASR/Models/fsmn_vad",
            device=funasr_device,
            disable_update=True,
        )
        logger.info("  fsmn-vad 模型加载完成")

        # ---- 3. Speaker (FunASR cam_plus) ----
        logger.info("3/3 加载 Speaker: cam_plus")
        _spk_model = FunASRAutoModel(
            model="/home/hhc/ASR/Models/cam_plus",
            device=funasr_device,
            disable_update=True,
        )
        logger.info("  cam_plus 模型加载完成")

        logger.info("=" * 50)
        logger.info("所有模型加载完毕！")

    return _asr_model, _vad_model, _spk_model


# ============================================================================
# FastAPI 应用
# ============================================================================

def create_app(
    asr_model_path: str,
    device: str,
    gpu_memory_utilization: float,
    max_new_tokens: int,
    # ASR 流式参数
    streaming_step_ms: int,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    chunk_size_sec: float,
    # VAD 参数
    min_speech_dur_ms: int,
):
    """创建 FastAPI 应用。

    Args:
        asr_model_path: Qwen3-ASR 模型路径
        device: 推理设备
        gpu_memory_utilization: vLLM GPU 内存利用率
        max_new_tokens: 最大生成 token 数
        streaming_step_ms: ASR 流式步长（毫秒）
        unfixed_chunk_num: 不固定前缀的 chunk 数量
        unfixed_token_num: 回退 token 数
        chunk_size_sec: ASR chunk 大小（秒）
        min_speech_dur_ms: 最短语音段（毫秒）
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

    app = FastAPI(
        title="Qwen3-ASR 实时语音识别",
        description="流式语音识别 + 说话人日志 Web 服务 (DynamicStreamingVAD + cam_plus)",
        version="2.0.0",
    )

    manager = ConnectionManager()

    # 静态文件
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    async def index():
        """返回前端页面"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>请在 static/ 目录下创建 index.html</h1>")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)

        # 加载全局模型
        asr_model, vad_model_raw, spk_model_raw = load_models(
            asr_model_path=asr_model_path,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=max_new_tokens,
        )

        # 为每个连接创建独立的 VAD、说话人跟踪器和会话
        vad = DynamicStreamingVAD(vad_model_raw)
        spk_tracker = HybridSpeakerTracker(spk_model_raw, device)
        session = RealtimeASRSession(
            asr_model=asr_model,
            vad=vad,
            spk_tracker=spk_tracker,
            streaming_step_ms=streaming_step_ms,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size_sec,
            max_new_tokens=max_new_tokens,
            min_speech_dur_ms=min_speech_dur_ms,
        )
        session.is_active = True

        logger.info("会话已启动")

        try:
            while True:
                # 接收消息（二进制音频数据 或 JSON 控制命令）
                data = await websocket.receive()

                if "bytes" in data:
                    # 二进制 int16 PCM 音频数据
                    raw_bytes = data["bytes"]
                    if len(raw_bytes) == 0:
                        continue

                    # 添加音频到会话
                    session.add_audio(raw_bytes)

                    # 处理 VAD
                    vad_events = session.process_vad()

                    # 处理每个新确认的语音段
                    for event in vad_events:
                        if event["type"] == "speech_segment":
                            seg_results = await session.process_segment(
                                event["start_ms"], event["end_ms"]
                            )
                            for msg in seg_results:
                                await manager.send_text(websocket, msg.to_json())

                elif "text" in data:
                    # JSON 控制命令
                    try:
                        command = json.loads(data["text"])
                        cmd_type = command.get("type", "")

                        if cmd_type == "stop":
                            # 停止，刷新缓冲
                            logger.info("收到停止命令，刷新缓冲...")
                            flush_results = await session.flush()
                            for msg in flush_results:
                                await manager.send_text(websocket, msg.to_json())

                            # 发送统计
                            stats = session.get_stats()
                            stats_msg = ASRMessage(
                                type="status",
                                text=json.dumps(stats, ensure_ascii=False),
                            )
                            await manager.send_text(websocket, stats_msg.to_json())

                        elif cmd_type == "reset":
                            logger.info("收到重置命令")
                            session.reset()
                            session.is_active = True

                        elif cmd_type == "ping":
                            await manager.send_text(
                                websocket,
                                ASRMessage(type="status", text="pong").to_json(),
                            )

                        elif cmd_type == "get_stats":
                            stats = session.get_stats()
                            await manager.send_text(
                                websocket,
                                ASRMessage(
                                    type="status",
                                    text=json.dumps(stats, ensure_ascii=False),
                                ).to_json(),
                            )

                    except json.JSONDecodeError as e:
                        logger.warning("无效的JSON: %s", e)

        except WebSocketDisconnect:
            logger.info("WebSocket 连接断开")
        except Exception as e:
            logger.error("WebSocket 错误: %s", e, exc_info=True)
        finally:
            # 清理
            manager.disconnect(websocket)
            try:
                flush_results = await session.flush()
                for msg in flush_results:
                    await manager.send_text(websocket, msg.to_json())
            except Exception:
                pass
            session.reset()

    return app


# ============================================================================
# 自签名证书生成（Python 内置，无需安装 OpenSSL）
# ============================================================================

def _generate_self_signed_cert(certfile: str, keyfile: str):
    """使用 Python cryptography 库生成自签名证书，适配 HTTPS/WSS。"""
    from datetime import datetime, timedelta, timezone

    # 生成 RSA 私钥
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(keyfile, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # 生成自签名证书
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ASR Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(certfile, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Web 实时流式语音识别服务端 (DynamicStreamingVAD + cam_plus + Qwen3-ASR)"
    )
    # ASR (Qwen3-ASR vLLM)
    parser.add_argument("--asr-model", required=True, help="Qwen3-ASR 模型路径")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM GPU 内存利用率 (默认 0.4)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="最大生成 token 数")

    # ASR 流式参数
    parser.add_argument("--streaming-step-ms", type=int, default=500,
                        help="ASR 流式步长（毫秒）")
    parser.add_argument("--unfixed-chunk-num", type=int, default=2,
                        help="不固定前缀的 chunk 数量")
    parser.add_argument("--unfixed-token-num", type=int, default=5,
                        help="前缀回退 token 数")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0,
                        help="ASR chunk 大小（秒）")

    # VAD 参数
    parser.add_argument("--min-speech-dur-ms", type=int, default=300,
                        help="最短语音段（毫秒）")

    # 服务
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--ssl", action="store_true", default=False,
                        help="启用 HTTPS/WSS（远程访问必须开启，浏览器要求 HTTPS 才能用麦克风）")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()

    # ---- GPU 设备配置 ----
    # vLLM 通过 CUDA_VISIBLE_DEVICES 环境变量选择 GPU，必须在 vLLM 导入前设置
    # 从 --device 参数（如 "cuda:1"）中提取 GPU 编号
    gpu_id = "0"
    if args.device.startswith("cuda:"):
        gpu_id = args.device.split(":", 1)[1]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        logger.info("GPU 配置: CUDA_VISIBLE_DEVICES=%s (来自 --device %s)", gpu_id, args.device)

    # 日志配置
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 创建 FastAPI 应用
    app = create_app(
        asr_model_path=args.asr_model,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        streaming_step_ms=args.streaming_step_ms,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
        min_speech_dur_ms=args.min_speech_dur_ms,
    )

    # 启动
    import uvicorn

    # ---- HTTPS / SSL 配置 ----
    # 浏览器安全策略：非 localhost 的 HTTP 连接禁止使用麦克风 (getUserMedia)
    # 远程访问需要 HTTPS，自动生成自签名证书
    ssl_certfile = Path(__file__).parent / "cert.pem"
    ssl_keyfile = Path(__file__).parent / "key.pem"
    use_ssl = False

    if args.ssl:
        use_ssl = True
        if not ssl_certfile.exists() or not ssl_keyfile.exists():
            logger.info("生成自签名 SSL 证书（Python 内置方式）...")
            _generate_self_signed_cert(str(ssl_certfile), str(ssl_keyfile))
            logger.info("SSL 证书已生成: %s / %s", ssl_certfile, ssl_keyfile)
        else:
            logger.info("使用已有 SSL 证书: %s", ssl_certfile)

    protocol = "https" if use_ssl else "http"
    ws_protocol = "wss" if use_ssl else "ws"
    logger.info("=" * 50)
    logger.info("Web 界面: %s://%s:%d/", protocol, args.host, args.port)
    logger.info("WebSocket 地址: %s://%s:%d/ws", ws_protocol, args.host, args.port)
    logger.info("=" * 50)

    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
        ssl_certfile=str(ssl_certfile) if use_ssl else None,
        ssl_keyfile=str(ssl_keyfile) if use_ssl else None,
    )


if __name__ == "__main__":
    main()