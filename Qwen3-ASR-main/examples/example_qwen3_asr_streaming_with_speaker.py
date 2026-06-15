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
流式ASR + VAD + CAM++ 说话人日记化

结合三种技术的语音识别流水线:
  - 流式ASR (Qwen3-ASR vLLM后端): 逐块增量式语音转写
  - VAD (FSMN-VAD): 语音活动检测 → "何时在说话"
  - CAM++: 说话人嵌入提取 + 聚类 → "谁在说话"

与 qwen3_asr_with_speaker.py 的区别:
  - 本脚本使用流式 ASR (vLLM backend)，逐块增量转写，实时输出中间结果
  - qwen3_asr_with_speaker.py 使用非流式 ASR (Transformers backend)，一次性转写整段

流程
----
  输入音频
      │
      ▼
  FSMN-VAD ──→ [段1: 0-30s] [段2: 30-60s] ...
      │
      ├──→ 每段送入 CAM++ → 192维嵌入 → 谱聚类 → spk_0, spk_1, ...
      │
      └──→ 每段送入 Qwen3-ASR (流式) → 逐块增量转写 → 实时输出
      │
      ▼
  按时间段对齐融合 → [说话人0] 文本1 | [说话人1] 文本2 | ...

依赖
----
  pip install qwen-asr[vllm] funasr torch librosa soundfile

用法
----
  python example_qwen3_asr_streaming_with_speaker.py \
      --audio /path/to/meeting.wav \
      --asr-model /path/to/Qwen3-ASR-0.6B \
      --device cuda:0
"""

import io
import base64
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SpeakerSegment:
    """
    带说话人标签的语音转写片段。

    Attributes:
        text: 转写文本
        language: 语言
        speaker_id: 说话人编号 (0, 1, 2, ...)
        start_sec: 片段起始时间（秒）
        end_sec: 片段结束时间（秒）
    """
    text: str
    language: str
    speaker_id: int
    start_sec: float
    end_sec: float


# ============================================================================
# 核心类
# ============================================================================

class StreamingSpeakerDiarizer:
    """
    流式ASR + VAD + CAM++ 说话人日记化。

    三模块:
      self.asr_model  → Qwen3-ASR (vLLM后端, 流式转写)
      self.vad_model  → FSMN-VAD，检测语音活动段落
      self.spk_model  → CAM++，提取说话人嵌入 + 聚类

    特点:
      - ASR 部分使用流式推理，逐块增量输出转写文本
      - VAD + CAM++ 负责说话人分离
      - 支持实时查看中间转写结果
    """

    def __init__(
        self,
        asr_model_path: str,
        device: str = "cuda:0",
        # ASR 流式参数
        asr_dtype: Optional[torch.dtype] = None,
        gpu_memory_utilization: float = 0.4,
        max_new_tokens: int = 256,
        # 流式分块参数
        streaming_step_ms: int = 1000,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        # VAD 参数
        vad_model: str = "/home/hhc/ASR/Models/fsmn_vad",
        max_single_segment_sec: float = 30.0,
        # 说话人模型
        spk_model: str = "/home/hhc/ASR/Models/cam_plus",
        # 聚类参数
        speaker_merge_threshold: float = 0.78,
        min_cluster_samples: int = 20,
        # 过滤参数
        min_speech_duration_sec: float = 0.3,
    ):
        """
        Args:
            asr_model_path: Qwen3-ASR 模型路径。
            device: 推理设备。
            asr_dtype: ASR 数据类型 (如 torch.bfloat16)。
            gpu_memory_utilization: vLLM GPU 内存利用率 (0-1)。
            max_new_tokens: 最大生成 token 数。
            streaming_step_ms: 流式输入步长（毫秒），控制每次送入模型的音频大小。
            unfixed_chunk_num: 前N个chunk不使用历史文本作为前缀提示。
            unfixed_token_num: 回退最后K个token以减少边界抖动。
            chunk_size_sec: 内部chunk大小（秒），音频按此增量送入模型。
            vad_model: VAD 模型路径。
            max_single_segment_sec: VAD 段落最大长度（秒）。
            spk_model: CAM++ 模型路径。
            speaker_merge_threshold: 聚类合并余弦相似度阈值。
            min_cluster_samples: 最少嵌入数才进行聚类。
            min_speech_duration_sec: 最短语音段落（秒）。
        """
        self.device = device
        self.streaming_step_ms = streaming_step_ms
        self.max_single_segment_sec = max_single_segment_sec
        self.merge_threshold = speaker_merge_threshold
        self.min_cluster_samples = min_cluster_samples
        self.min_speech_dur = min_speech_duration_sec
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.chunk_size_sec = chunk_size_sec

        # ---- 1. Qwen3-ASR (vLLM 流式后端) ----
        logger.info("=" * 50)
        logger.info("1/3 加载 Qwen3-ASR (vLLM 流式后端): %s", asr_model_path)
        from qwen_asr import Qwen3ASRModel

        self.asr_model = Qwen3ASRModel.LLM(
            model=asr_model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=max_new_tokens,
        )

        # ---- 2. VAD ----
        logger.info("2/3 加载 VAD: %s", vad_model)
        from funasr import AutoModel as FunASRModel

        self.vad_model = FunASRModel(
            model=vad_model,
            device=device,
            max_single_segment_time=int(max_single_segment_sec * 1000),
        )

        # ---- 3. CAM++ ----
        logger.info("3/3 加载说话人模型: %s", spk_model)
        self.spk_model = FunASRModel(model=spk_model, device=device)

        # 聚类后端
        from funasr.models.campplus.cluster_backend import ClusterBackend
        self.cluster = ClusterBackend().to(device)

        # 运行时统计（每次 transcribe 调用后更新）
        self._stats: Dict[str, Any] = {}

        logger.info("初始化完成！（流式ASR模式）")

    # ==================================================================
    # 统计信息
    # ==================================================================

    @property
    def statistics(self) -> Dict[str, Any]:
        """返回最近一次 transcribe 调用的运行时统计信息。

        Returns:
            包含以下字段的字典:
              - total_audio_duration_sec: 音频总时长（秒）
              - vad_segment_count: VAD 检测的语音段落数
              - total_embeddings: 提取的说话人嵌入总数
              - detected_speaker_count: 识别到的说话人数
              - merged_segment_count: 合并后的段落数（= ASR 调用次数）
              - asr_call_count: ASR 模型调用次数
              - asr_call_reduction_pct: ASR 调用减少百分比
              - streaming_mode: 是否使用流式模式
        """
        return dict(self._stats)

    # ==================================================================
    # 主入口
    # ==================================================================

    def transcribe(
        self,
        audio: Any,
        context: str = "",
        language: Optional[str] = None,
        preset_speaker_num: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """
        流式语音识别 + 说话人日记化。

        Args:
            audio: 音频输入 (路径 / URL / (ndarray, sr) / ndarray)。
            context: 上下文提示词。
            language: 强制语言。
            preset_speaker_num: 预设说话人数。

        Returns:
            按时间排序的 SpeakerSegment 列表。
        """
        # ---- 归一化 ----
        wav = self._normalize(audio)
        total_sec = len(wav) / SAMPLE_RATE
        logger.info("音频时长: %.1f 秒", total_sec)

        # ---- 步骤1: VAD ----
        logger.info("步骤1/5: VAD 语音端点检测...")
        vad_segs = self._run_vad(wav)
        if not vad_segs:
            logger.warning("VAD 未检测到语音")
            return []
        logger.info("  检测到 %d 个语音段落", len(vad_segs))

        # ---- 步骤2: CAM++ 说话人嵌入提取 ----
        logger.info("步骤2/5: CAM++ 说话人嵌入提取...")
        embeddings = self._extract_embeddings(vad_segs)
        logger.info("  提取 %d 个嵌入向量", embeddings.shape[0])

        # ---- 步骤3: 聚类 ----
        logger.info("步骤3/5: 说话人聚类...")
        speaker_labels = self._cluster(embeddings, preset_speaker_num)
        num_spk = int(speaker_labels.max().item()) + 1 if speaker_labels.numel() > 0 else 0
        logger.info("  识别到 %d 位说话人", num_spk)

        # ---- 步骤4: 为每个 VAD 段分配说话人，合并同说话人相邻段 ----
        logger.info("步骤4/5: 合并同说话人相邻段落...")
        seg_speakers = self._get_segment_speakers(vad_segs, speaker_labels)
        merged_segs = self._merge_vad_by_speaker(vad_segs, seg_speakers)
        asr_call_count = len(merged_segs)
        logger.info("  VAD %d 段 → 合并为 %d 段（按说话人分组）",
                     len(vad_segs), asr_call_count)

        # ---- 步骤5: 流式ASR 对合并后的段落转录 ----
        logger.info("步骤5/5: Qwen3-ASR 流式转录 (%d 次调用)...", asr_call_count)
        segments = self._transcribe_segments_streaming(merged_segs, context, language)

        # 重新编号
        segments = self._renumber(segments)

        # ---- 收集运行时统计信息 ----
        final_speakers = len(set(s.speaker_id for s in segments))
        vad_count = len(vad_segs)
        self._stats = {
            "total_audio_duration_sec": round(total_sec, 1),
            "vad_segment_count": vad_count,
            "total_embeddings": int(embeddings.shape[0]) if embeddings.numel() > 0 else 0,
            "detected_speaker_count": num_spk,
            "merged_segment_count": asr_call_count,
            "asr_call_count": len(merged_segs),
            "asr_call_reduction_pct": round(
                (1 - asr_call_count / vad_count) * 100, 1
            ) if vad_count > 0 else 0.0,
            "output_segment_count": len(segments),
            "final_speaker_count": final_speakers,
            "streaming_mode": True,
            "streaming_step_ms": self.streaming_step_ms,
        }

        logger.info("输出 %d 个片段, %d 位说话人",
                     len(segments), final_speakers)
        logger.info("ASR 调用效率: %d次 → 减少 %.1f%% (原始VAD段: %d)",
                     asr_call_count, self._stats["asr_call_reduction_pct"], vad_count)
        return segments

    # ==================================================================
    # 步骤1: VAD
    # ==================================================================

    def _run_vad(
        self, wav: np.ndarray
    ) -> List[Tuple[float, float, np.ndarray]]:
        """FSMN-VAD → 语音段落列表 [(start_sec, end_sec, audio), ...]"""
        res = self.vad_model.generate(
            input=wav,
            cache={},
            max_single_segment_time=int(self.max_single_segment_sec * 1000),
        )
        if not res:
            return []

        timestamps = res[0].get("value", [])
        min_samples = int(self.min_speech_dur * SAMPLE_RATE)
        segments = []

        for seg in timestamps:
            s_ms, e_ms = float(seg[0]), float(seg[1])
            s_sec, e_sec = s_ms / 1000.0, e_ms / 1000.0
            if int((e_sec - s_sec) * SAMPLE_RATE) < min_samples:
                continue
            s0 = max(0, int(s_sec * SAMPLE_RATE))
            s1 = min(len(wav), int(e_sec * SAMPLE_RATE))
            segments.append((s_sec, e_sec, wav[s0:s1].copy()))

        return segments

    # ==================================================================
    # 步骤2: CAM++ 嵌入提取
    # ==================================================================

    def _extract_embeddings(
        self, vad_segs: List[Tuple[float, float, np.ndarray]]
    ) -> torch.Tensor:
        """
        对每个 VAD 段落提取说话人嵌入。

        策略: 对每段取一个整体嵌入（段内说话人通常一致）。
        长段落用 1.5s/0.75s 滑窗拆分为多个嵌入，提升聚类稳定性。
        """
        all_embs: List[torch.Tensor] = []
        window_s = int(1.5 * SAMPLE_RATE)   # 1.5s 窗口
        stride_s = int(0.75 * SAMPLE_RATE)  # 0.75s 步长

        for _, _, audio in vad_segs:
            if len(audio) < window_s:
                # 短段落: 直接取一个嵌入
                emb = self._get_embedding(audio)
                if emb is not None:
                    all_embs.append(emb)
            else:
                # 长段落: 滑窗取多个嵌入
                offset = 0
                while offset + window_s <= len(audio):
                    chunk = audio[offset:offset + window_s]
                    emb = self._get_embedding(chunk)
                    if emb is not None:
                        all_embs.append(emb)
                    offset += stride_s

        if not all_embs:
            return torch.empty((0, 192), device=self.device)
        return torch.cat(all_embs, dim=0)

    def _get_embedding(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        """对一段音频提取 CAM++ 192维嵌入。"""
        if len(audio) < int(0.3 * SAMPLE_RATE):
            return None
        try:
            res = self.spk_model.generate(input=audio, cache={})
        except Exception:
            return None
        if res and len(res) > 0 and "spk_embedding" in res[0]:
            emb = res[0]["spk_embedding"]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            emb = emb.to(self.device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return emb
        return None

    # ==================================================================
    # 步骤3: 聚类
    # ==================================================================

    def _cluster(
        self, embeddings: torch.Tensor, preset_num: Optional[int]
    ) -> torch.Tensor:
        """谱聚类 → 说话人标签"""
        N = embeddings.shape[0]
        if N < self.min_cluster_samples:
            logger.info("  嵌入数(%d)不足, 全部归为说话人0", N)
            return torch.zeros(N, dtype=torch.long)
        try:
            labels = self.cluster(embeddings.cpu(), oracle_num=preset_num)
        except Exception as e:
            logger.warning("聚类失败: %s, 降级为单说话人", e)
            return torch.zeros(N, dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        return labels

    # ==================================================================
    # 步骤4: 嵌入 → VAD段 → 说话人分配 → 合并
    # ==================================================================

    def _get_segment_speakers(
        self,
        vad_segs: List[Tuple[float, float, np.ndarray]],
        speaker_labels: torch.Tensor,
    ) -> List[int]:
        """
        为每个 VAD 段落计算主导说话人。

        每个 VAD 段可能对应多个嵌入（长段落会滑窗），
        对段内所有嵌入投票，取多数说话人。
        """
        seg_speakers: List[int] = []
        emb_idx = 0  # 在 speaker_labels 中的游标

        window_s = int(1.5 * SAMPLE_RATE)
        stride_s = int(0.75 * SAMPLE_RATE)

        for seg_tuple in vad_segs:
            audio = seg_tuple[2]
            # 计算该段对应几个嵌入
            if len(audio) < window_s:
                n_embs = 1
            else:
                n_embs = max(1, (len(audio) - window_s) // stride_s + 1)

            # 投票
            votes: Dict[int, int] = defaultdict(int)
            for _ in range(n_embs):
                if emb_idx < len(speaker_labels):
                    votes[int(speaker_labels[emb_idx].item())] += 1
                emb_idx += 1

            spk = max(votes, key=votes.get) if votes else 0
            seg_speakers.append(spk)

        return seg_speakers

    @staticmethod
    def _merge_vad_by_speaker(
        vad_segs: List[Tuple[float, float, np.ndarray]],
        seg_speakers: List[int],
        max_gap_sec: float = 3.0,
    ) -> List[Tuple[float, float, np.ndarray, int]]:
        """
        合并相邻且同说话人的 VAD 段落。

        合并条件:
          1. 说话人相同
          2. 段落间隔 < max_gap_sec（避免跨大段静音合并）

        Returns:
            [(start_sec, end_sec, merged_audio, speaker_id), ...]
        """
        if not vad_segs:
            return []

        merged: List[Tuple[float, float, np.ndarray, int]] = []
        i = 0

        while i < len(vad_segs):
            s_start, s_end, audio = vad_segs[i]
            spk = seg_speakers[i]
            merged_audio = audio.copy()
            merged_end = s_end

            # 向后搜索可合并的段
            j = i + 1
            while j < len(vad_segs):
                # 说话人不同 → 停止
                if seg_speakers[j] != spk:
                    break
                # 间隔太大 → 停止
                gap = vad_segs[j][0] - merged_end
                if gap > max_gap_sec:
                    break
                # 合并
                merged_audio = np.concatenate([merged_audio, vad_segs[j][2]])
                merged_end = vad_segs[j][1]
                j += 1

            merged.append((s_start, merged_end, merged_audio, spk))
            i = j

        return merged

    # ==================================================================
    # 步骤5: 流式 ASR 转录（核心改动 - 使用流式推理）
    # ==================================================================

    def _transcribe_segments_streaming(
        self,
        merged_segs: List[Tuple[float, float, np.ndarray, int]],
        context: str,
        language: Optional[str],
    ) -> List[SpeakerSegment]:
        """
        对合并后的段落使用流式 ASR 进行转录。

        与 _transcribe_segments (非流式) 的区别:
          - 使用 init_streaming_state() → streaming_transcribe() → finish_streaming_transcribe()
          - 音频分块逐步送入模型，实时输出中间结果
          - 每处理一个 chunk 就打印中间转写文本

        每个合并段 = 同一说话人的连续语音。
        段数 = 说话人切换次数 + 1，远少于原始 VAD 段数。
        """
        segments: List[SpeakerSegment] = []

        for seg_idx, (seg_start, seg_end, audio, spk) in enumerate(merged_segs):
            logger.info(
                "  [%d/%d] 流式转录 说话人%d (%.1fs-%.1fs, 时长%.1fs)...",
                seg_idx + 1, len(merged_segs), spk,
                seg_start, seg_end, len(audio) / SAMPLE_RATE,
            )

            try:
                # 初始化流式状态
                state = self.asr_model.init_streaming_state(
                    context=context,
                    language=language,
                    unfixed_chunk_num=self.unfixed_chunk_num,
                    unfixed_token_num=self.unfixed_token_num,
                    chunk_size_sec=self.chunk_size_sec,
                )

                # 按 streaming_step_ms 步长逐步送入音频
                step_samples = int(round(self.streaming_step_ms / 1000.0 * SAMPLE_RATE))
                pos = 0
                call_id = 0

                while pos < len(audio):
                    seg = audio[pos: pos + step_samples]
                    pos += len(seg)
                    call_id += 1

                    # 流式转写
                    self.asr_model.streaming_transcribe(seg, state)

                    # 实时打印中间结果
                    if call_id % 5 == 0 or pos >= len(audio):
                        logger.debug(
                            "    流式步骤 %d: language=%s text=%s",
                            call_id, state.language, state.text,
                        )

                # 完成流式转写
                self.asr_model.finish_streaming_transcribe(state)

                text = state.text
                lang = state.language

                logger.info(
                    "    最终结果: 说话人%d language=%s text=%s",
                    spk, lang, text,
                )

            except Exception as e:
                logger.warning(
                    "流式ASR 失败 (%.1f-%.1fs, spk=%d): %s",
                    seg_start, seg_end, spk, e,
                )
                continue

            if not text.strip():
                continue

            segments.append(SpeakerSegment(
                text=text,
                language=lang,
                speaker_id=spk,
                start_sec=seg_start,
                end_sec=seg_end,
            ))

        return segments

    # ==================================================================
    # 后处理
    # ==================================================================

    @staticmethod
    def _renumber(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """重新编号说话人为 0, 1, 2, ..."""
        mapping: Dict[int, int] = {}
        next_id = 0
        for seg in segments:
            if seg.speaker_id not in mapping:
                mapping[seg.speaker_id] = next_id
                next_id += 1
            seg.speaker_id = mapping[seg.speaker_id]
        return segments

    # ==================================================================
    # 音频归一化
    # ==================================================================

    @staticmethod
    def _normalize(audio_input: Any) -> np.ndarray:
        """统一转为 16kHz mono float32。"""
        try:
            import librosa
            import soundfile as sf
        except ImportError as e:
            raise ImportError(f"需要安装: {e}")

        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            raw, sr = audio_input
            wav = np.asarray(raw, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        elif isinstance(audio_input, np.ndarray):
            wav = audio_input.astype(np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)

        elif isinstance(audio_input, str):
            if audio_input.startswith(("http://", "https://")):
                from urllib.request import urlopen
                with urlopen(audio_input) as resp:
                    data, sr = sf.read(io.BytesIO(resp.read()))
            elif audio_input.startswith("data:audio"):
                _, b64 = audio_input.split(",", 1)
                data, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
            else:
                data, sr = sf.read(audio_input)

            wav = np.asarray(data, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            raise ValueError(f"不支持的音频类型: {type(audio_input)}")

        peak = float(np.abs(wav).max())
        if peak > 1.0:
            wav = wav / peak
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    # ==================================================================
    # 工具
    # ==================================================================

    @staticmethod
    def speaker_count(segments: List[SpeakerSegment]) -> int:
        return len(set(s.speaker_id for s in segments))

    @staticmethod
    def print_results(segments: List[SpeakerSegment]) -> None:
        print(f"\n{'='*50}")
        print(f"共 {StreamingSpeakerDiarizer.speaker_count(segments)} 位说话人, "
              f"{len(segments)} 个片段")
        print(f"{'='*50}\n")
        for seg in segments:
            lang = f"[{seg.language}]" if seg.language else ""
            print(f"说话人{seg.speaker_id} {lang} "
                  f"({seg.start_sec:.1f}s-{seg.end_sec:.1f}s): {seg.text}")
        print()


# ============================================================================
# 命令行
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="流式ASR + VAD + CAM++ 说话人日记化"
    )
    parser.add_argument(
        "--audio",
        default="/home/hhc/ASR/Qwen3-ASR-main/examples/output_16k.wav",
        help="音频文件路径",
    )
    parser.add_argument(
        "--asr-model",
        default="/home/hhc/ASR/Models/Qwen3-ASR-0.6B",
        help="Qwen3-ASR 模型路径",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="推理设备",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="强制语言",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="上下文提示词",
    )
    parser.add_argument(
        "--preset-speaker-num",
        type=int,
        default=None,
        help="预设说话人数",
    )
    parser.add_argument(
        "--vad-model",
        default="/home/hhc/ASR/Models/fsmn_vad",
        help="VAD 模型路径",
    )
    parser.add_argument(
        "--spk-model",
        default="/home/hhc/ASR/Models/cam_plus",
        help="CAM++ 说话人模型路径",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM GPU 内存利用率",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--streaming-step-ms",
        type=int,
        default=1000,
        help="流式输入步长（毫秒），控制每次送入模型的音频大小",
    )
    parser.add_argument(
        "--chunk-size-sec",
        type=float,
        default=2.0,
        help="内部chunk大小（秒）",
    )
    parser.add_argument(
        "--unfixed-chunk-num",
        type=int,
        default=2,
        help="前N个chunk不使用历史文本作为前缀",
    )
    parser.add_argument(
        "--unfixed-token-num",
        type=int,
        default=5,
        help="回退最后K个token以减少边界抖动",
    )
    parser.add_argument(
        "--max-single-segment-sec",
        type=float,
        default=30.0,
        help="VAD 段落最大长度（秒）",
    )
    parser.add_argument(
        "--speaker-merge-threshold",
        type=float,
        default=0.78,
        help="说话人聚类合并阈值",
    )
    parser.add_argument(
        "--min-speech-duration-sec",
        type=float,
        default=0.3,
        help="最短语音段落（秒）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 创建流式说话人日记化器
    diarizer = StreamingSpeakerDiarizer(
        asr_model_path=args.asr_model,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        streaming_step_ms=args.streaming_step_ms,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
        vad_model=args.vad_model,
        max_single_segment_sec=args.max_single_segment_sec,
        spk_model=args.spk_model,
        speaker_merge_threshold=args.speaker_merge_threshold,
        min_speech_duration_sec=args.min_speech_duration_sec,
    )

    # 执行转写
    results = diarizer.transcribe(
        audio=args.audio,
        context=args.context,
        language=args.language,
        preset_speaker_num=args.preset_speaker_num,
    )

    # 打印结果
    StreamingSpeakerDiarizer.print_results(results)

    # 打印统计信息
    stats = diarizer.statistics
    print(f"统计信息:")
    print(f"  音频总时长: {stats.get('total_audio_duration_sec', 0)}秒")
    print(f"  VAD段落数: {stats.get('vad_segment_count', 0)}")
    print(f"  嵌入向量数: {stats.get('total_embeddings', 0)}")
    print(f"  检测说话人数: {stats.get('detected_speaker_count', 0)}")
    print(f"  合并后段落数: {stats.get('merged_segment_count', 0)}")
    print(f"  ASR调用次数: {stats.get('asr_call_count', 0)}")
    print(f"  ASR调用减少: {stats.get('asr_call_reduction_pct', 0)}%")
    print(f"  最终说话人数: {stats.get('final_speaker_count', 0)}")
    print(f"  流式模式: {stats.get('streaming_mode', False)}")
    print(f"  流式步长: {stats.get('streaming_step_ms', 0)}ms")


if __name__ == "__main__":
    main()