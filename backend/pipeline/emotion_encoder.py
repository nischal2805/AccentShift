"""Temporal emotion encoder — chunked SER mode only.

Runs the loaded audeering wav2vec2 SER on overlapping 500ms windows (100ms hop)
to build a per-frame V/A/D trajectory. Sigmoid is applied to raw logits so each
value is in [0,1] with neutral ≈ 0.5.

The emformer mode has been removed: it used LibriSpeech ASR features projected
through a zero-initialised linear layer — ASR features ≠ emotion features, the
projection had no trained weights, so outputs were noise.

The output EmotionTrajectory drives temporal F0/energy warping in EmotionCorrector.
"""
from __future__ import annotations
import logging
import numpy as np
import torch
from .types import EmotionVec, EmotionFrame, EmotionTrajectory

log = logging.getLogger("emotion_encoder")

CHUNK_MS          = 500   # analysis window (ms)
HOP_MS            = 100   # hop between windows (ms)
MIN_CHUNK_SAMPLES = 400   # skip chunks shorter than this (silence/edge)


class EmotionEncoder:
    """Temporal emotion encoder. Init once, call encode() per segment."""

    def __init__(self, mm, mode: str = "chunked_ser"):
        self.mm = mm
        if mode != "chunked_ser":
            log.warning("EmotionEncoder mode=%r not supported; using chunked_ser", mode)

    def encode(self, audio: np.ndarray, sr: int) -> EmotionTrajectory:
        """Return per-frame emotion trajectory for `audio`."""
        return self._encode_chunked_ser(audio, sr)

    @torch.no_grad()
    def _encode_chunked_ser(self, audio: np.ndarray, sr: int) -> EmotionTrajectory:
        chunk_n = int(CHUNK_MS / 1000 * sr)
        hop_n   = int(HOP_MS   / 1000 * sr)
        ext     = self.mm.ser_extractor
        model   = self.mm.ser
        device  = self.mm.ser_device

        frames: list[EmotionFrame] = []
        pos = 0
        while pos + chunk_n <= len(audio):
            chunk = audio[pos : pos + chunk_n]
            if len(chunk) < MIN_CHUNK_SAMPLES:
                pos += hop_n
                continue
            inp = ext(chunk, sampling_rate=sr, return_tensors="pt")
            iv  = {k: v.to(device) for k, v in inp.items()}
            out = model(**iv)
            logits = out.logits if hasattr(out, "logits") else out[1]
            vals   = torch.sigmoid(logits.squeeze().float()).cpu().numpy().reshape(-1)
            vec    = EmotionVec(
                valence   = float(vals[2]),
                arousal   = float(vals[0]),
                dominance = float(vals[1]),
            )
            t = (pos + chunk_n // 2) / sr
            frames.append(EmotionFrame(t=t, vec=vec))
            pos += hop_n

        # Fallback: whole clip if too short for even one window
        if not frames:
            inp = ext(audio, sampling_rate=sr, return_tensors="pt")
            iv  = {k: v.to(device) for k, v in inp.items()}
            out = model(**iv)
            logits = out.logits if hasattr(out, "logits") else out[1]
            vals   = torch.sigmoid(logits.squeeze().float()).cpu().numpy().reshape(-1)
            frames.append(EmotionFrame(
                t   = len(audio) / sr / 2,
                vec = EmotionVec(float(vals[2]), float(vals[0]), float(vals[1])),
            ))

        log.debug("encode: %d emotion frames (chunk=%dms hop=%dms)",
                  len(frames), CHUNK_MS, HOP_MS)
        return EmotionTrajectory(frames=frames, sr=sr)
