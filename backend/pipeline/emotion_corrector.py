"""Emotion correction: warp output F0/energy toward source when emotion drifts.

No-op when emotion cosine similarity >= threshold. Otherwise rescale F0 (clipped) and
energy (clipped) toward the source so the emotional "shape" survives accent conversion.
"""
from __future__ import annotations
import logging
import librosa
import numpy as np
import pyworld
from .types import EmotionVec, ProsodyFeatures

log = logging.getLogger("emotion_corrector")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


class EmotionCorrector:
    def __init__(self, cfg: dict):
        self.threshold = cfg["emotion_correction"]["threshold"]
        self.f0_clip = tuple(cfg["emotion_correction"]["f0_scale_clip"])
        self.energy_clip = tuple(cfg["emotion_correction"]["energy_scale_clip"])

    def correct(self, wav: np.ndarray, sr: int, source_emotion: EmotionVec,
                output_emotion: EmotionVec, source_prosody: ProsodyFeatures) -> np.ndarray:
        sim = _cosine(source_emotion.to_array(), output_emotion.to_array())
        if sim >= self.threshold:
            return wav  # no correction needed
        log.info("emotion drift (cos=%.3f < %.2f) — correcting", sim, self.threshold)
        x = wav.astype(np.float64)  # pyworld REQUIRES float64
        f0, t = pyworld.harvest(x, sr)
        f0 = pyworld.stonemask(x, f0, t, sr)
        voiced = f0 > 0
        src_f0 = source_prosody.f0
        src_voiced = src_f0[src_f0 > 0]
        if voiced.sum() > 0 and src_voiced.size > 0:
            scale = src_voiced.mean() / f0[voiced].mean()
            f0[voiced] *= np.clip(scale, self.f0_clip[0], self.f0_clip[1])
        sp = pyworld.cheaptrick(x, f0, t, sr)
        ap = pyworld.d4c(x, f0, t, sr)
        corrected = pyworld.synthesize(f0, sp, ap, sr).astype(np.float32)
        # energy warp toward source RMS
        out_rms = librosa.feature.rms(y=corrected)[0].mean()
        src_rms = source_prosody.energy.mean()
        if out_rms > 0:
            escale = np.clip(src_rms / out_rms, self.energy_clip[0], self.energy_clip[1])
            corrected = (corrected * escale).astype(np.float32)
        return corrected
