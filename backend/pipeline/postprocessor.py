"""Assemble segments (overlap-add crossfade) and loudness-normalise to target LUFS."""
from __future__ import annotations
import logging
import numpy as np
import pyloudnorm as pyln

log = logging.getLogger("postprocessor")


class Postprocessor:
    def __init__(self, cfg: dict):
        self.sr = cfg["audio"]["sample_rate"]
        self.crossfade_ms = cfg["postprocess"]["crossfade_ms"]
        self.target_lufs = cfg["postprocess"]["target_lufs"]

    def assemble(self, wavs: list[np.ndarray]) -> np.ndarray:
        if not wavs:
            return np.zeros(0, dtype=np.float32)
        if len(wavs) == 1:
            return wavs[0].astype(np.float32)
        xf = int(self.crossfade_ms / 1000 * self.sr)
        out = wavs[0].astype(np.float32)
        for nxt in wavs[1:]:
            nxt = nxt.astype(np.float32)
            n = min(xf, len(out), len(nxt))
            if n > 0:
                fade = np.linspace(1, 0, n, dtype=np.float32)
                out = out.copy()
                out[-n:] = out[-n:] * fade + nxt[:n] * (1 - fade)
                out = np.concatenate([out, nxt[n:]])
            else:
                out = np.concatenate([out, nxt])
        return out

    def loudness_normalize(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.sr:  # too short for reliable measurement
            return audio
        meter = pyln.Meter(self.sr)
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        normed = pyln.normalize.loudness(audio.astype(np.float64), loudness, self.target_lufs)
        peak = np.abs(normed).max()
        if peak > 1.0:
            normed = normed / peak
        return normed.astype(np.float32)
