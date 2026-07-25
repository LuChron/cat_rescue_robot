"""
模块 1: ASR — 语音 → 文字
只管语音转录，不关心内容是什么。
"""

import os
import re
import tempfile
import threading
import wave


def _default_device() -> str:
    """Use CUDA when CTranslate2 can actually see a GPU."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto").strip().lower()
if ASR_DEVICE == "auto":
    ASR_DEVICE = _default_device()

# The multilingual turbo weights are cached on the project machine. They use
# the large-v3 encoder and are more reliable for short Mandarin commands.
ASR_MODEL = os.environ.get(
    "ASR_MODEL",
    "large-v3-turbo" if ASR_DEVICE == "cuda" else "small",
)
ASR_PRIMARY_LANGUAGE = os.environ.get("ASR_LANGUAGE", "zh").strip().lower()
ASR_COMPUTE_TYPE = os.environ.get(
    "ASR_COMPUTE_TYPE", "int8_float16" if ASR_DEVICE == "cuda" else "int8"
)
ASR_CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", str(min(4, os.cpu_count() or 1))))
ASR_MIN_LOG_PROB = float(os.environ.get("ASR_MIN_LOG_PROB", "-0.65"))
ASR_NAV_MIN_LOG_PROB = float(os.environ.get("ASR_NAV_MIN_LOG_PROB", "-1.05"))
ASR_CONTROL_MIN_LOG_PROB = float(
    os.environ.get("ASR_CONTROL_MIN_LOG_PROB", "-1.50")
)

_model = None
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()

# 提示词按语言分开，避免中文短句被双语提示词带偏。
_ZH_PROMPT = (
    "机器人中文指令：找猫，找波斯猫，找布偶猫，找斯芬克斯猫，找新加坡猫，找兔狲，"
    "找狗, 找鸟, 去A区, 去B区, 去C区, 去茶水间, 喂食, 拍照, 陪玩, 安抚, "
    "向前走十厘米，向前走二十厘米，后退十厘米，向前走10 cm，后退20 cm，"
    "向前，后退，左转九十度，右转九十度，停止，旋转三百六十度，转一圈，"
    "在路口一停下，在路口二停下，继续，继续任务，"
    "抬起机械臂，放下机械臂，打开夹爪，归位。"
)
_EN_PROMPT = (
    "English robot commands: find cat, find Persian, find Ragdoll, go to zone C, go to point C, "
    "go to zone A and stop at junction one, go to point A and stop at junction one, "
    "go to point A, stop at junction one, find the cat and feed it, "
    "find the Singapura cat and interact with it, interact with it, play with the cat, "
    "feed, play, photo, move forward, move backward, turn left, turn right, rotate 360 degrees, "
    "stop at junction one, stop at junction two, continue, resume mission, "
    "stop, raise arm, lower arm, extend arm, retract arm, open gripper, arm home."
)
_ZH_HOTWORDS = (
    "向前 后退 厘米 米 十厘米 二十厘米 三十厘米 五十厘米 "
    "左转 右转 九十度 一百八十度 三百六十度 停止 路口一 路口二 继续 机械臂 夹爪"
)
_EN_HOTWORDS = (
    "forward backward centimeters meters turn left turn right "
    "90 degrees 180 degrees 360 degrees point A point B point C point D "
    "point E point F point G point H go to zone A and stop at junction one "
    "go to point A stop at junction one find the cat feed it "
    "find the Singapura cat interact with it play with the cat "
    "stop at junction one stop at junction two "
    "continue resume robot arm gripper"
)

# ---- 手动停止录音状态 ----
_recording = False
_buffer: list = []
_stream = None
_samplerate = 16000
_recording_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(
                ASR_MODEL,
                device=ASR_DEVICE,
                compute_type=ASR_COMPUTE_TYPE,
                cpu_threads=ASR_CPU_THREADS,
                num_workers=1,
            )
    return _model


def _audio_callback(indata, frames, time, status):
    del frames, time, status
    with _recording_lock:
        if _recording:
            _buffer.append(indata.copy())


# ---- 手动控制录音（Web） ----

def _input_samplerate(sd, requested: int | None) -> int:
    """优先使用显式采样率，否则使用声卡原生采样率。"""
    if requested:
        return int(requested)
    device = sd.query_devices(kind="input")
    return int(round(device["default_samplerate"]))


def start_recording(samplerate: int | None = None):
    """开始录音（非阻塞），音频持续追加到缓冲区。"""
    global _recording, _buffer, _stream, _samplerate
    import sounddevice as sd

    samplerate = _input_samplerate(sd, samplerate)
    with _recording_lock:
        if _recording:
            raise RuntimeError("already recording")
        _buffer = []
        _samplerate = samplerate
    stream = sd.InputStream(
        samplerate=samplerate, channels=1, dtype="int16",
        callback=_audio_callback,
    )
    with _recording_lock:
        _stream = stream
        _recording = True
    try:
        stream.start()
    except Exception:
        with _recording_lock:
            _recording = False
            _stream = None
        stream.close()
        raise


def stop_recording(filename: str | None = None) -> str | None:
    """停止录音，保存为 WAV；不会重复处理上一次录音。"""
    global _recording, _stream, _buffer
    with _recording_lock:
        if not _recording:
            return None
        _recording = False
        stream = _stream
        _stream = None
        chunks = _buffer
        _buffer = []

    if stream is not None:
        stream.stop()
        stream.close()

    if not chunks:
        return None

    import numpy as np

    audio = np.concatenate(chunks)
    duration = len(audio) / _samplerate
    if duration < 0.3:
        return None  # 太短，当作误触

    if filename is None:
        fd, filename = tempfile.mkstemp(prefix="catrescue_voice_", suffix=".wav")
        os.close(fd)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_samplerate)
        wf.writeframes(audio.tobytes())

    return filename


def is_recording() -> bool:
    with _recording_lock:
        return _recording


# ---- 固定时长录音（CLI） ----

def record_audio(duration: float = 5.0, samplerate: int | None = None,
                 filename: str = "command.wav") -> str:
    """录音（阻塞），保存为 WAV，返回文件路径。faster-whisper 内部会自动重采样。"""
    import sounddevice as sd

    samplerate = _input_samplerate(sd, samplerate)
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


def speech_to_text_result(audio_path: str, language: str | None = None) -> dict:
    """Transcribe audio and retain confidence metadata for safe selection."""
    with _transcribe_lock:
        kwargs = {
            "initial_prompt": (
                _ZH_PROMPT if language == "zh"
                else _EN_PROMPT if language == "en"
                else f"{_ZH_PROMPT} {_EN_PROMPT}"
            ),
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 250,
            },
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "hotwords": (
                _ZH_HOTWORDS if language == "zh"
                else _EN_HOTWORDS if language == "en"
                else f"{_ZH_HOTWORDS} {_EN_HOTWORDS}"
            ),
        }
        if language:
            kwargs["language"] = language
        segments, info = _get_model().transcribe(audio_path, **kwargs)
        segments = list(segments)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    weighted_log_prob = sum(
        segment.avg_logprob * max(segment.end - segment.start, 0.01)
        for segment in segments
    )
    duration = sum(max(segment.end - segment.start, 0.01) for segment in segments)
    return {
        "text": text,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "avg_logprob": weighted_log_prob / duration if duration else -10.0,
    }


def is_repetitive_transcript(text: str) -> bool:
    """Detect Whisper loops such as '继续继续继续...'."""
    compact = re.sub(r"[\s，。！？,.!?:：;；'\"“”]+", "", text.casefold())
    if len(compact) < 12:
        return False
    for unit_size in range(1, min(9, len(compact) // 4 + 1)):
        for start in range(0, len(compact) - unit_size * 4 + 1):
            unit = compact[start:start + unit_size]
            end = start + unit_size
            while compact[end:end + unit_size] == unit:
                end += unit_size
            repeated_length = end - start
            if repeated_length >= 12 and repeated_length / len(compact) >= 0.45:
                return True
    return False


def transcript_preview(text: str, limit: int = 48) -> str:
    """Keep pathological ASR output bounded in API and UI responses."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def speech_to_text(audio_path: str, language: str | None = None) -> str:
    """Backward-compatible text-only transcription API."""
    return speech_to_text_result(audio_path, language=language)["text"]


def normalize_command_transcript(text: str) -> str:
    """Repair a small set of high-confidence ASR errors in command context."""
    text = re.sub(r"\s+", " ", text).strip()
    # Whisper small can render spoken "十厘米" as "左失落". Restrict the
    # correction to a movement prefix so unrelated speech is never rewritten.
    text = re.sub(
        r"((?:向前|前进|往前|后退|往后)(?:走|移动)?)\s*(?:左失落|走失落|十里米)",
        r"\g<1>十厘米",
        text,
    )
    return text


def voice_command_min_log_prob(command: dict | None) -> float:
    """Return a command-specific ASR confidence threshold.

    A one-word continue/resume command naturally receives a lower Whisper
    average log probability than a full sentence. It is still safe to accept
    more liberally because the controller only applies it while a mission is
    already paused.
    """
    if command and command.get("control_action") == "continue":
        return ASR_CONTROL_MIN_LOG_PROB
    if (
        command
        and command.get("zone")
        and not command.get("breed")
        and not command.get("actions")
        and not command.get("distance_cm")
        and not command.get("turn_deg")
        and not command.get("manual_key")
    ):
        return ASR_NAV_MIN_LOG_PROB
    return ASR_MIN_LOG_PROB


def warmup_model():
    """后台预加载模型，避免第一次停止录音时才等待模型加载。"""
    _get_model()


def get_asr_config() -> dict:
    return {
        "model": ASR_MODEL,
        "device": ASR_DEVICE,
        "compute_type": ASR_COMPUTE_TYPE,
        "primary_language": ASR_PRIMARY_LANGUAGE or "auto",
        "min_log_prob": ASR_MIN_LOG_PROB,
        "navigation_min_log_prob": ASR_NAV_MIN_LOG_PROB,
        "control_min_log_prob": ASR_CONTROL_MIN_LOG_PROB,
        "capture_samplerate": "device default",
    }


def command_languages() -> list[str | None]:
    """短指令不适合完全依赖自动语言检测，按优先级返回候选语言。"""
    if ASR_PRIMARY_LANGUAGE in ("", "auto"):
        return [None, "zh", "en"]
    if ASR_PRIMARY_LANGUAGE == "en":
        return ["en", "zh"]
    return ["zh", "en"]


def is_chinese_or_english(text: str) -> bool:
    """只接受 ASCII 英文和中日韩统一表意文字；拒绝俄文、假名、韩文等输出。"""
    for char in text:
        if not char.isalpha():
            continue
        codepoint = ord(char)
        if codepoint < 128:
            continue
        if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
            continue
        return False
    return True
