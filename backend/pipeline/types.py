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
        """V/A/D centered on the neutral midpoint (audeering MSP-dim outputs ~[0,1]).

        Cosine similarity on the raw all-positive vectors sits ~0.95 for any pair, so it
        cannot discriminate emotions. Subtracting the 0.5 neutral point restores direction
        so cosine actually measures emotional difference.
        """
        return self.to_array() - 0.5


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
