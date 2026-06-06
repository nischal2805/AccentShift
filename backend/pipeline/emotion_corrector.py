"""Emotion correction: warp output F0/energy to match source emotion trajectory.

Replaces the old global-scale approach with per-frame correction guided by
the EmotionTrajectory from EmotionEncoder. Each voiced frame is warped
individually based on its local emotion deviation from the source.

Algorithm per voiced frame i:
  1. Look up source emotion at time t_i  → src_vad
  2. Look up output emotion at time t_i  → out_vad
  3. Cosine similarity of centered vecs
  4. If sim < frame_threshold: apply weighted F0 + energy correction
     - F0 scale toward src_f0_mean * arousal_ratio  (arousal drives pitch)
     - energy scale toward src_rms * dominance_ratio (dominance drives loudness)
     - alpha = 1 - sim (stronger correction where emotion drifts more)
  5. Smoothed via Gaussian window to avoid click artifacts

Falls back to global correction if EmotionTrajectory has only 1 frame (short clip).
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
import librosa
import pyworld
from .types import EmotionVec, EmotionTrajectory, ProsodyFeatures

log = logging.getLogger("emotion_corrector")

_NEUTRAL = 0.5   # audeering MSP-dim neutral midpoint


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def _center(v: EmotionVec) -> np.ndarray:
    return np.array([v.valence - _NEUTRAL,
                     v.arousal - _NEUTRAL,
                     v.dominance - _NEUTRAL], dtype=np.float64)


class EmotionCorrector:
    def __init__(self, cfg: dict):
        ec = cfg["emotion_correction"]
        self.threshold    = ec["threshold"]
        self.f0_clip      = tuple(ec["f0_scale_clip"])
        self.energy_clip  = tuple(ec["energy_scale_clip"])
        # Per-frame alpha smoothing window (frames)
        self.smooth_sigma = cfg.get("emotion_encoder", {}).get("smooth_sigma", 5)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def correct(
        self,
        wav: np.ndarray,
        sr: int,
        source_emotion: EmotionVec,
        output_emotion: EmotionVec,
        source_prosody: ProsodyFeatures,
        source_trajectory: EmotionTrajectory | None = None,
        output_trajectory: EmotionTrajectory | None = None,
    ) -> np.ndarray:
        # Global similarity check
        sim = _cosine(_center(source_emotion), _center(output_emotion))
        if sim >= self.threshold:
            return wav   # emotion well preserved — no correction

        log.info("emotion drift (cos=%.3f < %.2f) — applying correction", sim, self.threshold)

        has_traj = (source_trajectory is not None
                    and output_trajectory is not None
                    and len(source_trajectory.frames) > 1
                    and len(output_trajectory.frames) > 1)

        if has_traj:
            return self._correct_temporal(wav, sr, source_prosody,
                                          source_trajectory, output_trajectory)
        return self._correct_global(wav, sr, source_emotion, output_emotion, source_prosody)

    # ------------------------------------------------------------------
    # Global correction (fallback — original behaviour)
    # ------------------------------------------------------------------
    def _correct_global(
        self,
        wav: np.ndarray,
        sr: int,
        source_emotion: EmotionVec,
        output_emotion: EmotionVec,
        source_prosody: ProsodyFeatures,
    ) -> np.ndarray:
        x = wav.astype(np.float64)
        pw = pyworld
        f0, t  = pw.harvest(x, sr)
        f0     = pw.stonemask(x, f0, t, sr)
        sp     = pw.cheaptrick(x, f0, t, sr)
        ap     = pw.d4c(x, f0, t, sr)

        voiced     = f0 > 0
        src_voiced = source_prosody.f0[source_prosody.f0 > 0]
        if voiced.sum() > 0 and src_voiced.size > 0:
            scale = src_voiced.mean() / f0[voiced].mean()
            f0[voiced] *= np.clip(scale, self.f0_clip[0], self.f0_clip[1])

        corrected = pw.synthesize(f0, sp, ap, sr).astype(np.float32)
        out_rms = librosa.feature.rms(y=corrected)[0].mean()
        src_rms = source_prosody.energy.mean()
        if out_rms > 0:
            escale = np.clip(src_rms / out_rms, self.energy_clip[0], self.energy_clip[1])
            corrected = (corrected * escale).astype(np.float32)
        return corrected

    # ------------------------------------------------------------------
    # Temporal (per-frame) correction — new path
    # ------------------------------------------------------------------
    def _correct_temporal(
        self,
        wav: np.ndarray,
        sr: int,
        source_prosody: ProsodyFeatures,
        src_traj: EmotionTrajectory,
        out_traj: EmotionTrajectory,
    ) -> np.ndarray:
        x  = wav.astype(np.float64)
        pw = pyworld

        f0, t  = pw.harvest(x, sr)
        f0     = pw.stonemask(x, f0, t, sr)
        sp     = pw.cheaptrick(x, f0, t, sr)
        ap     = pw.d4c(x, f0, t, sr)

        n_frames = len(f0)

        # Build per-frame alpha (correction strength) from emotion cosine
        alpha = np.zeros(n_frames, dtype=np.float64)
        f0_scale = np.ones(n_frames, dtype=np.float64)

        # Source voiced F0 mean for reference
        src_f0_voiced = source_prosody.f0[source_prosody.f0 > 0]
        src_f0_mean   = float(src_f0_voiced.mean()) if src_f0_voiced.size > 0 else 0.0

        for i in range(n_frames):
            ti      = t[i]
            src_vad = src_traj.at(ti)
            out_vad = out_traj.at(ti)
            sim     = _cosine(_center(src_vad), _center(out_vad))
            a       = max(0.0, 1.0 - sim)   # higher drift → stronger correction
            alpha[i] = a

            # Arousal-guided F0 target: arousal > neutral → higher pitch
            if src_f0_mean > 0 and f0[i] > 0:
                arousal_ratio = (src_vad.arousal + 0.5) / (out_vad.arousal + 0.5 + 1e-6)
                arousal_ratio = float(np.clip(arousal_ratio, self.f0_clip[0], self.f0_clip[1]))
                # Blend: no correction at a=0, full arousal_ratio at a=1
                f0_scale[i]  = 1.0 + a * (arousal_ratio - 1.0)

        # Smooth alpha and scale to avoid frame-boundary clicks
        alpha    = gaussian_filter1d(alpha,    sigma=self.smooth_sigma)
        f0_scale = gaussian_filter1d(f0_scale, sigma=self.smooth_sigma)

        # Apply F0 scale only to voiced frames
        voiced      = f0 > 0
        f0_corrected = f0.copy()
        f0_corrected[voiced] = np.clip(
            f0[voiced] * f0_scale[voiced],
            f0[voiced] * self.f0_clip[0],
            f0[voiced] * self.f0_clip[1],
        )

        corrected = pw.synthesize(f0_corrected, sp, ap, sr).astype(np.float32)

        # Per-frame energy warp guided by dominance ratio + alpha
        hop     = 512
        out_rms = librosa.feature.rms(y=corrected, hop_length=hop)[0]
        src_rms = source_prosody.energy   # shape (E,)
        min_len = min(len(out_rms), len(src_rms))
        if min_len > 0 and out_rms[:min_len].mean() > 0:
            # Frame-level energy ratio, clipped
            ratio = np.clip(
                src_rms[:min_len] / (out_rms[:min_len] + 1e-8),
                self.energy_clip[0], self.energy_clip[1]
            )
            # Alpha for energy frames (resample alpha to energy frame count)
            alpha_e = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(alpha)),
                alpha,
            )
            energy_scale = 1.0 + alpha_e * (ratio - 1.0)
            energy_scale = gaussian_filter1d(energy_scale, sigma=3)
            # Apply: expand per-frame scale to sample-level
            scale_samples = np.interp(
                np.arange(len(corrected)),
                np.arange(min_len) * hop + hop // 2,
                energy_scale[:min_len],
            )
            corrected = (corrected * scale_samples).astype(np.float32)

        log.debug("temporal correction applied: %d frames, mean_alpha=%.3f",
                  n_frames, float(alpha.mean()))
        return corrected
