"""Emotion correction: warp output F0/energy to match source emotion trajectory.

Three F0 transfer modes (config: emotion_correction.f0_transfer):

  log_norm   (default) — log-domain contour copy normalized to output speaker
             register. Preserves the *shape* of the source F0 contour (emotional
             dynamics) while keeping the output speaker's mean pitch level.
             Proven approach from CWT-based F0 transfer literature: log F0 is
             perceptually uniform (semitone scale), so shifting by the mean
             difference maps source dynamics onto target register cleanly.

  cwt        — Mexican-hat CWT decomposition of log-F0 at 10 scales (4 ms →
             ~400 ms at 200 Hz WORLD frame rate). Each scale represents a
             different prosodic level: micro-prosody → sentence intonation.
             Source CWT coefficients replace output coefficients, then inverse
             CWT reconstructs log-F0. Stronger temporal structure preservation
             at cost of some wavelet ringing artefacts.

  mean_scale — original single-ratio mean scaling (legacy fallback).

Energy correction is always per-frame via dominance × alpha weighting from
the EmotionTrajectory. F0 correction is only applied when global cosine
similarity falls below threshold (same gate as before).
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
import librosa
import pyworld
from .types import EmotionVec, EmotionTrajectory, ProsodyFeatures

log = logging.getLogger("emotion_corrector")

def _ricker(points: int, a: float) -> np.ndarray:
    """Mexican-hat (Ricker) wavelet — avoids scipy version dependency."""
    A   = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    vec = np.arange(points) - (points - 1) / 2.0
    xsq = vec ** 2
    wsq = a ** 2
    return A * (1.0 - xsq / wsq) * np.exp(-xsq / (2.0 * wsq))


def _cwt(signal: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Manual CWT via convolution — scipy.signal.cwt removed in 1.12."""
    out = np.zeros((len(widths), len(signal)), dtype=np.float64)
    for i, w in enumerate(widths):
        n = min(int(10 * w) | 1, len(signal))   # odd kernel length
        wav = _ricker(n, w)
        wav /= np.sum(np.abs(wav)) + 1e-12
        out[i] = np.convolve(signal, wav, mode="same")
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def _center(v: EmotionVec) -> np.ndarray:
    return v.to_centered_array().astype(np.float64)


def _interp_f0(f0: np.ndarray) -> np.ndarray:
    """Linear interpolation across unvoiced (F0=0) frames → continuous contour."""
    f0c = f0.copy().astype(np.float64)
    voiced = f0c > 0
    if voiced.sum() < 2:
        f0c[~voiced] = 1.0
        return f0c
    idx = np.arange(len(f0c))
    f0c[~voiced] = np.interp(idx[~voiced], idx[voiced], f0c[voiced])
    return f0c


# ---------------------------------------------------------------------------
# F0 transfer strategies
# ---------------------------------------------------------------------------

def _resample_f0(src: np.ndarray, target_len: int) -> np.ndarray:
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(src)),
        src,
    )


def _f0_log_norm(src_f0: np.ndarray, out_f0: np.ndarray,
                 f0_clip: tuple) -> np.ndarray:
    """Log-normalised F0 contour transfer.

    Maps source emotional F0 dynamics onto the output speaker's register by
    shifting in log space. Voiced/unvoiced mask from *output* is preserved.
    """
    out_voiced = out_f0 > 0
    src_rs = _resample_f0(src_f0, len(out_f0))
    src_v  = src_rs > 0

    if out_voiced.sum() < 2 or src_v.sum() < 2:
        return out_f0.copy()

    out_log_mean = float(np.log(out_f0[out_voiced]).mean())
    src_log_mean = float(np.log(src_rs[src_v]).mean())

    # Continuous log-F0 source contour (interpolate over unvoiced regions)
    src_cont     = np.log(np.maximum(_interp_f0(src_rs), 1.0))
    # Shift to output speaker register
    transferred  = np.exp(src_cont - src_log_mean + out_log_mean)
    # Clip to plausible Hz range and restore unvoiced
    lo = out_f0[out_voiced].min() * f0_clip[0]
    hi = out_f0[out_voiced].max() * f0_clip[1]
    return np.where(out_voiced, np.clip(transferred, lo, hi), 0.0)


def _f0_cwt(src_f0: np.ndarray, out_f0: np.ndarray,
            f0_clip: tuple) -> np.ndarray:
    """CWT multi-scale F0 contour transfer (Mexican-hat, 10 scales).

    Decomposes source and output log-F0 into prosodic scales from micro
    (~4 ms) to sentence (~400 ms at 200 Hz WORLD rate), transfers source
    coefficients, then reconstructs. Normalised to output speaker mean.
    """
    out_voiced = out_f0 > 0
    src_rs = _resample_f0(src_f0, len(out_f0))
    src_v  = src_rs > 0

    if out_voiced.sum() < 2 or src_v.sum() < 2:
        return out_f0.copy()

    out_log_mean = float(np.log(out_f0[out_voiced]).mean())

    src_log = np.log(np.maximum(_interp_f0(src_rs), 1.0))

    # 10 octave-spaced scales; WORLD default frame rate ≈5 ms
    widths = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=np.float64)
    # CWT returns (n_scales, n_frames) complex (ricker is real)
    coefs = _cwt(src_log, widths)          # (10, N)
    # Approximate reconstruction via Morlet admissibility ∑_j C_j
    transferred_log = coefs.real.sum(axis=0)
    # Re-centre to output speaker register
    voiced_mean = float(transferred_log[out_voiced].mean()) if out_voiced.any() else 0.0
    transferred_log = transferred_log - voiced_mean + out_log_mean

    transferred = np.exp(transferred_log)
    lo = out_f0[out_voiced].min() * f0_clip[0]
    hi = out_f0[out_voiced].max() * f0_clip[1]
    return np.where(out_voiced, np.clip(transferred, lo, hi), 0.0)


def _f0_mean_scale(src_f0: np.ndarray, out_f0: np.ndarray,
                   f0_clip: tuple) -> np.ndarray:
    """Legacy mean-ratio scaling (fallback for very short clips)."""
    out_voiced = out_f0 > 0
    src_rs = _resample_f0(src_f0, len(out_f0))
    src_v  = src_rs > 0
    if out_voiced.sum() < 1 or src_v.sum() < 1:
        return out_f0.copy()
    scale = np.clip(
        src_rs[src_v].mean() / out_f0[out_voiced].mean(),
        f0_clip[0], f0_clip[1],
    )
    result = out_f0.copy()
    result[out_voiced] *= scale
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmotionCorrector:
    def __init__(self, cfg: dict):
        ec = cfg["emotion_correction"]
        self.threshold         = ec["threshold"]
        self.always_correct_f0 = ec.get("always_correct_f0", False)
        self.f0_corr_threshold = ec.get("f0_corr_threshold", 0.3)
        self.f0_clip           = tuple(ec["f0_scale_clip"])
        self.energy_clip       = tuple(ec["energy_scale_clip"])
        self.smooth_sigma      = cfg.get("emotion_encoder", {}).get("smooth_sigma", 5)
        self.f0_mode           = ec.get("f0_transfer", "log_norm")

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
        f0_corr: float = 1.0,
    ) -> np.ndarray:
        # Gate: only run PyWorld re-synthesis when prosody is badly degraded.
        # f0_corr < threshold means Seed-VC mangled the prosody — correct it.
        # always_correct_f0 overrides for testing/debugging only.
        should_correct = self.always_correct_f0 or (f0_corr < self.f0_corr_threshold)

        if should_correct:
            log.info("F0 correction: f0_corr=%.3f, threshold=%.2f, mode=%s",
                     f0_corr, self.f0_corr_threshold, self.f0_mode)
            has_traj = (source_trajectory is not None
                        and output_trajectory is not None
                        and len(source_trajectory.frames) > 1
                        and len(output_trajectory.frames) > 1)
            if has_traj:
                return self._correct_temporal(wav, sr, source_prosody,
                                              source_trajectory, output_trajectory)
            return self._correct_global(wav, sr, source_emotion, output_emotion, source_prosody)

        log.info("F0 correction skipped: f0_corr=%.3f >= %.2f (Seed-VC preserved prosody)",
                 f0_corr, self.f0_corr_threshold)
        return wav

    # ------------------------------------------------------------------
    def _pick_f0(self, src_f0: np.ndarray, out_f0: np.ndarray) -> np.ndarray:
        """Route to configured F0 transfer strategy."""
        if self.f0_mode == "cwt":
            return _f0_cwt(src_f0, out_f0, self.f0_clip)
        if self.f0_mode == "mean_scale":
            return _f0_mean_scale(src_f0, out_f0, self.f0_clip)
        return _f0_log_norm(src_f0, out_f0, self.f0_clip)   # default: log_norm

    # ------------------------------------------------------------------
    def _correct_global(self, wav, sr, _src_emo, _out_emo, source_prosody):
        x = wav.astype(np.float64)
        pw = pyworld
        f0, t = pw.harvest(x, sr)
        f0    = pw.stonemask(x, f0, t, sr)
        sp    = pw.cheaptrick(x, f0, t, sr)
        ap    = pw.d4c(x, f0, t, sr)

        src_f0 = source_prosody.f0
        f0_new = self._pick_f0(src_f0, f0)
        corrected = pw.synthesize(f0_new, sp, ap, sr).astype(np.float32)

        out_rms = librosa.feature.rms(y=corrected)[0].mean()
        src_rms = source_prosody.energy.mean()
        if out_rms > 0:
            escale = np.clip(src_rms / out_rms, self.energy_clip[0], self.energy_clip[1])
            corrected = (corrected * escale).astype(np.float32)
        return corrected

    # ------------------------------------------------------------------
    def _correct_temporal(self, wav, sr, source_prosody, src_traj, out_traj):
        x  = wav.astype(np.float64)
        pw = pyworld

        f0, t  = pw.harvest(x, sr)
        f0     = pw.stonemask(x, f0, t, sr)
        sp     = pw.cheaptrick(x, f0, t, sr)
        ap     = pw.d4c(x, f0, t, sr)

        # ---------- F0 transfer (log-norm / cwt / mean_scale) ----------
        f0_new = self._pick_f0(source_prosody.f0, f0)

        # ---------- Per-frame alpha from emotion trajectory drift -------
        n_frames = len(f0)
        alpha    = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            ti      = t[i]
            src_vad = src_traj.at(ti)
            out_vad = out_traj.at(ti)
            sim     = _cosine(_center(src_vad), _center(out_vad))
            alpha[i] = max(0.0, 1.0 - sim)

        alpha = gaussian_filter1d(alpha, sigma=self.smooth_sigma)

        # Blend F0: low alpha → keep output F0; high alpha → use transferred F0
        voiced = f0 > 0
        f0_blended = f0.copy()
        f0_blended[voiced] = (
            (1.0 - alpha[voiced]) * f0[voiced]
            + alpha[voiced]       * f0_new[voiced]
        )

        corrected = pw.synthesize(f0_blended, sp, ap, sr).astype(np.float32)

        # ---------- Per-frame energy warp (dominance-guided) ------------
        hop     = 512
        out_rms = librosa.feature.rms(y=corrected, hop_length=hop)[0]
        src_rms = source_prosody.energy
        min_len = min(len(out_rms), len(src_rms))
        if min_len > 0 and out_rms[:min_len].mean() > 0:
            ratio = np.clip(
                src_rms[:min_len] / (out_rms[:min_len] + 1e-8),
                self.energy_clip[0], self.energy_clip[1],
            )
            alpha_e = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(alpha)),
                alpha,
            )
            energy_scale = 1.0 + alpha_e * (ratio - 1.0)
            energy_scale = gaussian_filter1d(energy_scale, sigma=3)
            # Frame centre positions; np.interp fills last value past the end
            frame_centers = np.arange(min_len) * hop + hop // 2
            # Guarantee full coverage: pad with boundary values if audio longer
            if frame_centers[-1] < len(corrected) - 1:
                frame_centers = np.append(frame_centers, len(corrected) - 1)
                energy_scale  = np.append(energy_scale,  energy_scale[-1])
            scale_samples = np.interp(
                np.arange(len(corrected)), frame_centers, energy_scale,
            )
            corrected = (corrected * scale_samples).astype(np.float32)

        log.debug("temporal correction: %d frames, mean_alpha=%.3f, f0_mode=%s",
                  n_frames, float(alpha.mean()), self.f0_mode)
        return corrected
