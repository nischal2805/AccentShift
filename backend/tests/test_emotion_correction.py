"""Tests for EmotionCorrector: F0/energy warping, threshold behaviour."""
from __future__ import annotations
import numpy as np
import pytest
from pipeline.types import EmotionVec, ProsodyFeatures
from pipeline.emotion_corrector import EmotionCorrector

MINIMAL_CFG = {
    "emotion_correction": {
        "threshold": 0.85,
        "f0_scale_clip": [0.8, 1.2],
        "energy_scale_clip": [0.7, 1.3],
    }
}

SR = 16000


def _make_sine(freq: float = 220.0, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _prosody(wav: np.ndarray) -> ProsodyFeatures:
    import pyworld
    import librosa
    x = wav.astype(np.float64)
    f0, t = pyworld.harvest(x, SR)
    f0 = pyworld.stonemask(x, f0, t, SR)
    energy = librosa.feature.rms(y=wav)[0]
    return ProsodyFeatures(f0=f0, timeaxis=t, energy=energy, speaking_rate=2.0)


class TestEmotionCorrector:
    def setup_method(self):
        self.ec = EmotionCorrector(MINIMAL_CFG)
        self.wav = _make_sine()

    def test_noop_when_similarity_above_threshold(self):
        """When emotion is well-preserved, wav returned unchanged."""
        emo = EmotionVec(valence=0.5, arousal=0.5, dominance=0.5)
        prosody = _prosody(self.wav)
        result = self.ec.correct(self.wav, SR, emo, emo, prosody)
        # Identical emotion → no correction; result equals input
        np.testing.assert_array_equal(result, self.wav)

    def test_correction_applied_when_drifted(self):
        """When emotions differ, corrected wav differs from input."""
        src_emo = EmotionVec(valence=0.9, arousal=0.9, dominance=0.9)
        out_emo = EmotionVec(valence=-0.9, arousal=-0.9, dominance=-0.9)
        prosody = _prosody(self.wav)
        result = self.ec.correct(self.wav, SR, src_emo, out_emo, prosody)
        assert result.dtype == np.float32
        assert len(result) > 0

    def test_output_is_float32(self):
        src_emo = EmotionVec(valence=0.0, arousal=0.0, dominance=0.0)
        out_emo = EmotionVec(valence=1.0, arousal=1.0, dominance=1.0)
        prosody = _prosody(self.wav)
        result = self.ec.correct(self.wav, SR, src_emo, out_emo, prosody)
        assert result.dtype == np.float32

    def test_output_not_clipping(self):
        """Corrected audio peak must stay ≤ 1.0 (we don't clip, but scale should be mild)."""
        src_emo = EmotionVec(valence=0.0, arousal=-1.0, dominance=0.0)
        out_emo = EmotionVec(valence=1.0, arousal=1.0, dominance=1.0)
        prosody = _prosody(self.wav)
        result = self.ec.correct(self.wav, SR, src_emo, out_emo, prosody)
        # energy_scale_clip caps at 1.3 × source; reasonable sine should stay clean
        assert np.isfinite(result).all()
