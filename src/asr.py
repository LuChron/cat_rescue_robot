"""
模块 1: ASR — 语音 → 文字
只管语音转录，不关心内容是什么。
"""

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


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _model


def record_audio(duration: float = 5.0, samplerate: int = 44100,
                 filename: str = "command.wav") -> str:
    """录音，保存为 WAV，返回文件路径。faster-whisper 内部会自动重采样。"""
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
