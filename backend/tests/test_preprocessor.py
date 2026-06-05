"""Preprocessor.segment: 30s force-split + empty-VAD fallback (silero stubbed, no download)."""
import numpy as np
from pipeline.preprocessor import Preprocessor


CFG = {"audio": {"sample_rate": 16000, "max_segment_s": 30.0},
       "vad": {"threshold": 0.5, "min_speech_ms": 250, "min_silence_ms": 100}}


def _stub(pre, timestamps):
    """Bypass real silero load; feed canned timestamps."""
    pre._model = object()
    pre._get_ts = lambda *a, **k: timestamps


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
