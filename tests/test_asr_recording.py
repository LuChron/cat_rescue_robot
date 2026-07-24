from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from src import asr


class _FakeStream:
    def stop(self):
        pass

    def close(self):
        pass


def test_stop_recording_uses_a_unique_temporary_file():
    with asr._recording_lock:
        asr._recording = True
        asr._stream = _FakeStream()
        asr._samplerate = 16000
        asr._buffer = [np.zeros((8000, 1), dtype=np.int16)]

    audio_path = asr.stop_recording()
    try:
        assert audio_path is not None
        assert Path(audio_path).name.startswith("catrescue_voice_")
        assert Path(audio_path).is_file()
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)
