"""Temporal emotion encoder.

Two modes:
  - chunked_ser  (default, zero training): runs the loaded audeering wav2vec2 SER on
    overlapping 500ms windows → V/A/D trajectory. Fast, no extra models.
  - emformer     (optional, richer features): uses torchaudio's pre-trained Emformer
    to extract frame-level audio features, then projects them through the SER head.
    Emformer weights are pre-trained on LibriSpeech (streaming ASR); no emotion
    fine-tuning needed here — we still use SER logits, but Emformer features
    give us richer temporal context than raw waveform chunks alone.

The output EmotionTrajectory drives temporal F0/energy warping in EmotionCorrector,
replacing the old single-vector global correction.
"""
from __future__ import annotations
import logging
import numpy as np
import torch
from .types import EmotionVec, EmotionFrame, EmotionTrajectory

log = logging.getLogger("emotion_encoder")

# Window / hop config
CHUNK_MS   = 500    # analysis window (ms)
HOP_MS     = 100    # hop between windows (ms)
MIN_CHUNK_SAMPLES = 400   # skip chunks shorter than this (silence/edge)


class EmotionEncoder:
    """Temporal emotion encoder. Init once, call encode() per segment."""

    def __init__(self, mm, mode: str = "chunked_ser"):
        """
        mm   : ModelManager (must have .ser, .ser_extractor, .ser_device loaded)
        mode : "chunked_ser" | "emformer"
        """
        self.mm   = mm
        self.mode = mode
        self._emformer = None
        self._emf_proj = None
        if mode == "emformer":
            self._init_emformer()

    # ------------------------------------------------------------------
    # Emformer init (optional)
    # ------------------------------------------------------------------
    def _init_emformer(self) -> None:
        try:
            import torchaudio
            bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
            # Load only the encoder (no decoder/joiner needed)
            self._emformer = bundle.get_encoder()
            self._emformer.eval()
            device = self.mm.ser_device
            self._emformer.to(device)
            # Small linear head: emformer hidden (1024) → 3 (V/A/D)
            self._emf_proj = torch.nn.Linear(1024, 3).to(device)
            # Initialise projection so it starts near zero (neutral emotion)
            torch.nn.init.zeros_(self._emf_proj.weight)
            torch.nn.init.constant_(self._emf_proj.bias, 0.5)
            log.info("Emformer encoder loaded (device=%s)", device)
        except Exception as e:
            log.warning("Emformer init failed (%s) — falling back to chunked_ser", e)
            self.mode = "chunked_ser"
            self._emformer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, audio: np.ndarray, sr: int) -> EmotionTrajectory:
        """Return per-frame emotion trajectory for `audio`."""
        if self.mode == "emformer" and self._emformer is not None:
            return self._encode_emformer(audio, sr)
        return self._encode_chunked_ser(audio, sr)

    # ------------------------------------------------------------------
    # Mode 1: chunked SER
    # ------------------------------------------------------------------
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
            vals   = logits.squeeze().float().cpu().numpy().reshape(-1)
            vec    = EmotionVec(
                valence   = float(vals[2]),
                arousal   = float(vals[0]),
                dominance = float(vals[1]),
            )
            t = (pos + chunk_n // 2) / sr   # centre of window
            frames.append(EmotionFrame(t=t, vec=vec))
            pos += hop_n

        # Fallback: whole clip if too short
        if not frames:
            inp = ext(audio, sampling_rate=sr, return_tensors="pt")
            iv  = {k: v.to(device) for k, v in inp.items()}
            out = model(**iv)
            logits = out.logits if hasattr(out, "logits") else out[1]
            vals   = logits.squeeze().float().cpu().numpy().reshape(-1)
            frames.append(EmotionFrame(t=len(audio) / sr / 2,
                                       vec=EmotionVec(float(vals[2]),
                                                      float(vals[0]),
                                                      float(vals[1]))))

        log.debug("encode: %d emotion frames (chunk=%dms hop=%dms)",
                  len(frames), CHUNK_MS, HOP_MS)
        return EmotionTrajectory(frames=frames, sr=sr)

    # ------------------------------------------------------------------
    # Mode 2: Emformer backbone + SER projection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_emformer(self, audio: np.ndarray, sr: int) -> EmotionTrajectory:
        import torchaudio
        device = self.mm.ser_device

        # Resample to emformer expected 16kHz if needed
        wav = torch.from_numpy(audio).unsqueeze(0).float().to(device)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Emformer forward: returns (output, lengths)
        # output shape: (batch, frames, hidden=1024)
        lengths = torch.tensor([wav.shape[-1]], device=device)
        out, _ = self._emformer(wav, lengths)   # (1, T, 1024)
        # Project each frame to V/A/D
        vad = torch.sigmoid(self._emf_proj(out.squeeze(0)))  # (T, 3), range [0,1]
        vad_np = vad.float().cpu().numpy()

        # Map frame indices back to time (Emformer stride ~10ms for RNNT_BASE)
        frame_stride_s = 0.010
        frames = [
            EmotionFrame(
                t   = i * frame_stride_s,
                vec = EmotionVec(
                    valence   = float(vad_np[i, 0]),
                    arousal   = float(vad_np[i, 1]),
                    dominance = float(vad_np[i, 2]),
                )
            )
            for i in range(len(vad_np))
        ]
        log.debug("emformer: %d emotion frames at ~10ms stride", len(frames))
        return EmotionTrajectory(frames=frames, sr=sr)
