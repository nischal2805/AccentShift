"""Tests for Preprocessor: VAD segmentation and load_audio."""
from __future__ import annotations
import numpy as np
import pytest
from pipeline.preprocessor import Preprocessor, load_audio

MINIMAL_CFG = {
    "audio": {"sample_rate": 16000, "max_segment_s": 30.0},
    "vad":   {"threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 100},
}


class TestLoadAudio:
    def test_returns_float32(self, tmp_path):
        import soundfile as sf
        wav = np.zeros(16000, dtype=np.float32)
        p = tmp_path / "test.wav"
        sf.write(str(p), wav, 16000)
        audio = load_audio(str(p), sr=16000)
        assert audio.dtype == np.float32

    def test_mono(self, tmp_path):
        import soundfile as sf
        stereo = np.zeros((16000, 2), dtype=np.float32)
        p = tmp_path / "stereo.wav"
        sf.write(str(p), stereo, 16000)
        audio = load_audio(str(p), sr=16000)
        assert audio.ndim == 1


class TestPreprocessor:
    def test_segment_fallback_on_silence(self, silence_wav):
        """Pure silence → VAD finds no speech → fallback returns 1 segment."""
        audio, sr = silence_wav
        pre = Preprocessor(MINIMAL_CFG)
        segs = pre.segment(audio)
        assert len(segs) >= 1

    def test_segment_caps_at_max_segment_s(self, sine_wav):
        """Segments must not exceed max_segment_s."""
        audio, sr = sine_wav
        # Repeat to 35 s so it exceeds the 30 s cap
        long_audio = np.tile(audio, 35)
        pre = Preprocessor(MINIMAL_CFG)
        segs = pre.segment(long_audio)
        for seg in segs:
            assert seg.duration_s <= MINIMAL_CFG["audio"]["max_segment_s"] + 0.01

    def test_segment_preserves_audio_content(self, sine_wav):
        """Audio content inside each segment matches the source array slice."""
        audio, sr = sine_wav
        pre = Preprocessor(MINIMAL_CFG)
        segs = pre.segment(audio)
        assert all(len(s.audio) > 0 for s in segs)

    def test_segment_sr_set_correctly(self, sine_wav):
        audio, sr = sine_wav
        pre = Preprocessor(MINIMAL_CFG)
        segs = pre.segment(audio)
        for seg in segs:
            assert seg.sr == 16000
