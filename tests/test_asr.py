from src.asr import (
    ASR_MIN_LOG_PROB,
    is_chinese_or_english,
    is_repetitive_transcript,
    normalize_command_transcript,
    transcript_preview,
)


def test_accepts_chinese_and_english_transcripts():
    assert is_chinese_or_english("旋转360度")
    assert is_chinese_or_english("rotate 360 degrees")
    assert is_chinese_or_english("去 zone A 找 cat")


def test_rejects_other_language_scripts():
    assert not is_chinese_or_english("Всё здесь")
    assert not is_chinese_or_english("ロボット")
    assert not is_chinese_or_english("로봇")


def test_normalizes_observed_mandarin_distance_error_only_in_context():
    assert normalize_command_transcript("向前走左失落") == "向前走十厘米"
    assert normalize_command_transcript("我感到失落") == "我感到失落"


def test_confidence_threshold_is_conservative():
    assert -1.0 < ASR_MIN_LOG_PROB < -0.3


def test_rejects_repetitive_whisper_hallucination():
    text = "请在" + "继续" * 30
    assert is_repetitive_transcript(text)
    assert len(transcript_preview(text)) <= 51


def test_normal_commands_are_not_marked_repetitive():
    assert not is_repetitive_transcript("请向前走十厘米")
    assert not is_repetitive_transcript("turn right 360 degrees")
