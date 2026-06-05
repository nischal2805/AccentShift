"""Postprocessor: overlap-add length math + loudness guards."""
import numpy as np
from pipeline.postprocessor import Postprocessor


CFG = {"audio": {"sample_rate": 16000},
       "postprocess": {"crossfade_ms": 30, "target_lufs": -23.0}}


def test_assemble_empty():
    assert Postprocessor(CFG).assemble([]).size == 0


def test_assemble_single_passthrough():
    a = np.ones(100, dtype=np.float32)
    out = Postprocessor(CFG).assemble([a])
    assert np.array_equal(out, a)


def test_assemble_crossfade_length():
    sr = 16000
    xf = int(30 / 1000 * sr)            # 480 samples
    a = np.ones(sr, dtype=np.float32)
    b = np.ones(sr, dtype=np.float32)
    out = Postprocessor(CFG).assemble([a, b])
    assert len(out) == len(a) + len(b) - xf  # overlap-add eats one crossfade window


def test_loudness_skips_short_clip():
    short = np.ones(100, dtype=np.float32)
    out = Postprocessor(CFG).loudness_normalize(short)
    assert np.array_equal(out, short)   # < 1s -> returned unchanged


def test_loudness_caps_peak():
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    loud = (0.9 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    out = Postprocessor(CFG).loudness_normalize(loud)
    assert np.abs(out).max() <= 1.0 + 1e-6
