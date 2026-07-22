"""
模块 1: ASR — 语音 → 文字
只管语音转录，不关心内容是什么。
"""

import threading
import numpy as np
import sounddevice as sd
import wave
from faster_whisper import WhisperModel


# 首次运行自动下载模型，之后缓存
_model = None

# 提示词：用领域词汇引导模型，减少听错
_INITIAL_PROMPT = (
    "Cat breeds: Persian, Siamese, Maine Coon, Bengal, Ragdoll. "
    "Zones: Zone A, Zone B, Zone C, start. "
    "Actions: find, rescue, feed, search, return, take care."
)

# ---- 手动停止录音状态 ----
_recording = False
_buffer: list = []
_stream: sd.InputStream | None = None
_samplerate = 44100


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _model


def _audio_callback(indata, frames, time, status):
    if _recording:
        _buffer.append(indata.copy())


# ---- 手动控制录音（Web） ----

def start_recording(samplerate: int = 44100):
    """开始录音（非阻塞），音频持续追加到缓冲区。"""
    global _recording, _buffer, _stream, _samplerate
    _buffer = []
    _samplerate = samplerate
    _recording = True
    _stream = sd.InputStream(
        samplerate=samplerate, channels=1, dtype="int16",
        callback=_audio_callback,
    )
    _stream.start()


def stop_recording(filename: str = "command.wav", timeout: float = 10.0) -> str | None:
    """停止录音，保存为 WAV，返回文件路径。timeout 秒内无数据则返回 None。"""
    global _recording, _stream
    _recording = False
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

    if not _buffer:
        return None

    audio = np.concatenate(_buffer)
    duration = len(audio) / _samplerate
    if duration < 0.3:
        return None  # 太短，当作误触

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_samplerate)
        wf.writeframes(audio.tobytes())

    return filename


def is_recording() -> bool:
    return _recording


# ---- 固定时长录音（CLI） ----

def record_audio(duration: float = 5.0, samplerate: int = 44100,
                 filename: str = "command.wav") -> str:
    """录音（阻塞），保存为 WAV，返回文件路径。faster-whisper 内部会自动重采样。"""
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    return filename


def speech_to_text(audio_path: str, language: str = "en") -> str:
    """将音频文件转录为文字。"""
    model = _get_model()
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=_INITIAL_PROMPT,
        vad_filter=True,
    )
    return " ".join(seg.text for seg in segments)
