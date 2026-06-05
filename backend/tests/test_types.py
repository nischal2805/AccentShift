"""EmotionVec array conversions — centered array is the fix for the dead emotion metric."""
import numpy as np
from pipeline.types import EmotionVec


def test_to_array_order_is_vad():
    e = EmotionVec(valence=0.1, arousal=0.2, dominance=0.3)
    assert np.allclose(e.to_array(), [0.1, 0.2, 0.3])


def test_centered_subtracts_neutral_midpoint():
    e = EmotionVec(valence=0.5, arousal=0.5, dominance=0.5)  # neutral
    assert np.allclose(e.to_centered_array(), [0.0, 0.0, 0.0])


def test_centered_restores_cosine_discrimination():
    # Raw all-positive vectors -> cosine ~1 for clearly different emotions (the bug).
    happy = EmotionVec(valence=0.9, arousal=0.8, dominance=0.7)
    sad = EmotionVec(valence=0.1, arousal=0.2, dominance=0.3)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    raw = cos(happy.to_array(), sad.to_array())
    centered = cos(happy.to_centered_array(), sad.to_centered_array())
    assert raw > 0.85            # raw cosine cannot tell them apart
    assert centered < raw        # centering separates them
