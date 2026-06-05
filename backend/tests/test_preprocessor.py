"""Preprocessor tests.

Two layers:
  - stubbed VAD (deterministic, no silero download): 30s force-split + empty-VAD fallback
  - integration (real silero via fixtures): load_audio + segment caps/sr
"""
from __future__ import annotations
import numpy as np
import pytest
from pipeline.preprocessor import Preprocessor, load_audio


CFG = {"audio": {"sample_rate": 16000, "max_segment_s": 30.0},
       "vad": {"threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 100}}


def _stub(pre, timestamps):
    """Bypass real silero load; feed canned timestamps."""
    pre._model = object()
    pre._get_ts = lambda *a, **k: timestamps


# --- stubbed VAD: deterministic, no model download ---------------------------

def test_force_split_caps_segments_at_30s():
    sr = 16000
    pre = Preprocessor(CFG)
    _stub(pre, [{"start": 0, "end": 70 * sr}])     # 70s span
    audio = np.zeros(70 * sr, dtype=np.float32)
    segs = pre.segment(audio)
    assert len(segs) == 3                            # 30 + 30 + 10
    assert all(s.duration_s <= 30.0 + 1e-6 for s in segs)
    assert abs(segs[-1].duration_s - 10.0) < 1e-6


def test_empty_vad_fallback_single_segment():
    sr = 16000
    pre = Preprocessor(CFG)
    _stub(pre, [])
    audio = np.zeros(5 * sr, dtype=np.float32)
    segs = pre.segment(audio)
    assert len(segs) == 1
    assert segs[0].start_s == 0.0


def test_fallback_clip_capped_at_max():
    sr = 16000
    pre = Preprocessor(CFG)
    _stub(pre, [])
    audio = np.zeros(40 * sr, dtype=np.float32)      # longer than cap, no VAD hits
    segs = pre.segment(audio)
    assert len(segs) == 1
    assert segs[0].duration_s <= 30.0 + 1e-6


# --- load_audio -------------------------------------------------------------

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


# --- integration: real silero VAD (downloads model on first run) ------------

class TestPreprocessor:
    def test_segment_fallback_on_silence(self, silence_wav):
        """Pure silence → VAD finds no speech → fallback returns 1 segment."""
        audio, sr = silence_wav
        pre = Preprocessor(CFG)
        segs = pre.segment(audio)
        assert len(segs) >= 1

    def test_segment_caps_at_max_segment_s(self, sine_wav):
        """Segments must not exceed max_segment_s."""
        audio, sr = sine_wav
        long_audio = np.tile(audio, 35)              # 35s > 30s cap
        pre = Preprocessor(CFG)
        segs = pre.segment(long_audio)
        for seg in segs:
            assert seg.duration_s <= CFG["audio"]["max_segment_s"] + 0.01

    def test_segment_preserves_audio_content(self, sine_wav):
        audio, sr = sine_wav
        pre = Preprocessor(CFG)
        segs = pre.segment(audio)
        assert all(len(s.audio) > 0 for s in segs)

    def test_segment_sr_set_correctly(self, sine_wav):
        audio, sr = sine_wav
        pre = Preprocessor(CFG)
        segs = pre.segment(audio)
        for seg in segs:
            assert seg.sr == 16000
