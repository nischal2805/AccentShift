"""EmotionCorrector: no-op above threshold, real correction below, no NaN/garbage out."""
import numpy as np
import pyworld
import pytest
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.types import EmotionVec, ProsodyFeatures


CFG = {"emotion_correction": {"threshold": 0.85,
                              "f0_scale_clip": [0.8, 1.2],
                              "energy_scale_clip": [0.7, 1.3]}}


def _prosody(wav, sr):
    x = wav.astype(np.float64)
    f0, t = pyworld.harvest(x, sr)
    f0 = pyworld.stonemask(x, f0, t, sr)
    energy = np.sqrt(np.mean(wav ** 2)) * np.ones(1, dtype=np.float64)
    return ProsodyFeatures(f0=f0, timeaxis=t, energy=energy, speaking_rate=2.0)


@pytest.fixture
def voiced_wav():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr


def test_noop_when_emotion_similar(voiced_wav):
    wav, sr = voiced_wav
    ec = EmotionCorrector(CFG)
    same = EmotionVec(valence=0.7, arousal=0.6, dominance=0.5)
    out = ec.correct(wav, sr, same, same, _prosody(wav, sr))
    assert out is wav  # identical centered vectors -> cos 1 >= threshold -> untouched


def test_corrects_when_emotion_drifts(voiced_wav):
    wav, sr = voiced_wav
    ec = EmotionCorrector(CFG)
    src = EmotionVec(valence=0.9, arousal=0.9, dominance=0.9)
    out_emo = EmotionVec(valence=0.1, arousal=0.1, dominance=0.1)  # centered cos = -1 < 0.85
    out = ec.correct(wav, sr, src, out_emo, _prosody(wav, sr))
    assert out is not wav
    assert out.dtype == np.float32
    assert out.size > 0
    assert np.isfinite(out).all()        # BUG2 fix: clean envelope -> no NaN/inf
    assert np.abs(out).max() <= 2.0      # no blow-up from clipped scaling
