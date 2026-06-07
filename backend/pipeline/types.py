"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Segment:
    audio: np.ndarray          # float32, 16kHz mono
    start_s: float
    end_s: float
    sr: int = 16000

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sr


@dataclass
class EmotionVec:
    valence: float
    arousal: float
    dominance: float

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance], dtype=np.float64)

    def to_centered_array(self) -> np.ndarray:
        """Sigmoid-normalised [0,1] V/A/D centred at 0.5 → [-0.5, 0.5] range.

        Sigmoid is applied in feature_extractor before constructing EmotionVec, so
        values here are in [0,1] with neutral ≈ 0.5. Subtracting 0.5 gives a
        direction vector where positive = above neutral, negative = below neutral.
        Cosine / L2 on these vectors is meaningful for discrimination.
        """
        return self.to_array() - 0.5


@dataclass
class EmotionFrame:
    """Emotion vector at a specific time offset (seconds from segment start)."""
    t: float
    vec: EmotionVec


@dataclass
class EmotionTrajectory:
    """Temporal sequence of emotion frames across an audio segment."""
    frames: list  # list[EmotionFrame]
    sr: int

    def at(self, t: float) -> EmotionVec:
        """Nearest-frame lookup."""
        if not self.frames:
            return EmotionVec(0.5, 0.5, 0.5)
        best = min(self.frames, key=lambda f: abs(f.t - t))
        return best.vec

    def to_array(self) -> np.ndarray:
        """Shape (T, 3) — V/A/D per frame."""
        return np.array([[f.vec.valence, f.vec.arousal, f.vec.dominance]
                         for f in self.frames], dtype=np.float32)


@dataclass
class ProsodyFeatures:
    f0: np.ndarray             # float64
    timeaxis: np.ndarray
    energy: np.ndarray
    speaking_rate: float       # words per second


@dataclass
class Candidate:
    name: str                  # "seed_vc" | "vevo"
    wav: np.ndarray
    sr: int


@dataclass
class SegmentResult:
    wav: np.ndarray
    sr: int
    emotion_source: EmotionVec
    emotion_output: EmotionVec
    transcript: str
    wer: float
    mos: float
    chosen_backend: str
    score: float
    start_s: float
    end_s: float
