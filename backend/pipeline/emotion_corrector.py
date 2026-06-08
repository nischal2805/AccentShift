"""Emotion correction: F0 contour transfer + energy envelope matching.

Pipeline:
  Source F0 : librosa.pyin (feature_extractor — already stored in ProsodyFeatures)
  Output F0 : librosa.pyin (NOT PyWorld HARVEST — PYIN handles neural-vocoded output)
  Transfer  : log-norm contour mapping (shape preserved, register adapted)
  Synthesis : PyWorld CheapTrick+D4C+synthesize driven by PYIN F0
  Energy    : per-frame RMS ratio scaling (non-destructive, always applied)

Why PYIN over HARVEST for output F0:
  PyWorld HARVEST is a glottal-pulse tracker optimised for natural speech.
  Seed-VC output is neural-vocoded — it has pseudo-periodicity from the
  neural network, not real glottal pulses. HARVEST mis-estimates F0 on this,
  causing CheapTrick to use wrong analysis windows → severe buzziness.
  PYIN uses autocorrelation + dynamic programming, which works on both natural
  and neural-vocoded speech (it makes no assumption about glottal structure).

Why not Parselmouth OLA:
  Praat TD-PSOLA requires pitch-synchronous frames aligned to glottal closures.
  On neural vocoder output these don't exist → the "two voices" comb-filter
  artifact reported in testing.

Research basis:
  - Rizwan et al. 2022: F0 69%, energy 18% of perceived emotion
  - De Cheveigné & Kawahara 2002: YIN — the basis for PYIN
  - Morise et al. 2016: WORLD vocoder — CheapTrick/D4C/synthesize chain
  - Log F0 normalisation: Ming et al. 2016, standard in prosody transfer
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
import librosa
import pyworld
from .types import EmotionVec, EmotionTrajectory, ProsodyFeatures

log = logging.getLogger("emotion_corrector")

_PYIN_FMIN = librosa.note_to_hz("C2")   # 65 Hz — floor for all voice types
_PYIN_FMAX = librosa.note_to_hz("C7")   # 2093 Hz — covers excited/child speech
_PYIN_HOP  = 80                          # 5 ms at 16 kHz — MUST match PyWorld frame period
                                         # PyWorld synthesize hardcodes 5ms; mismatched hop
                                         # makes output 3.2× shorter (the "5 second" bug)


def _pyin_f0(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 and timeaxis from audio using probabilistic YIN.

    Returns (f0, timeaxis) where f0=0.0 for unvoiced frames (PyWorld convention).
    PYIN is accurate on neural-vocoded speech; HARVEST is not.
    """
    f0, _vf, _vp = librosa.pyin(
        y=audio,
        fmin=_PYIN_FMIN,
        fmax=_PYIN_FMAX,
        sr=sr,
        hop_length=_PYIN_HOP,
        fill_na=0.0,
    )
    timeaxis = np.arange(len(f0)) * _PYIN_HOP / sr
    return f0, timeaxis


def _log_norm_transfer(
    src_f0: np.ndarray,
    src_timeaxis: np.ndarray,
    out_f0: np.ndarray,
    out_timeaxis: np.ndarray,
    f0_clip: tuple,
) -> np.ndarray:
    """Log-normalised F0 contour transfer.

    Maps source emotion dynamics (shape of F0 contour) onto the output speaker's
    pitch register. Normalised time axis handles duration mismatch between source
    and output segments.

    Returns new_f0 array aligned to out_timeaxis, 0.0 for unvoiced frames.
    """
    src_voiced = src_f0 > 0
    out_voiced = out_f0 > 0

    if src_voiced.sum() < 2 or out_voiced.sum() < 2:
        return out_f0.copy()

    src_dur = float(src_timeaxis[-1]) if len(src_timeaxis) > 0 else 1.0
    out_dur = float(out_timeaxis[-1]) if len(out_timeaxis) > 0 else 1.0

    # Map source voiced F0 onto output timeaxis using normalised [0,1] axis
    out_norm    = out_timeaxis / (out_dur + 1e-8)
    src_at_out  = np.interp(
        out_norm * src_dur,
        src_timeaxis[src_voiced],
        src_f0[src_voiced],
        left=float(src_f0[src_voiced][0]),
        right=float(src_f0[src_voiced][-1]),
    )
    src_at_out = np.maximum(src_at_out, 1.0)

    # Log-norm shift: keep source contour SHAPE, adapt to output register
    out_log_mean = float(np.log(out_f0[out_voiced]).mean())
    src_log_mean = float(np.log(src_at_out[out_voiced]).mean())

    new_f0 = out_f0.copy()
    transferred = np.exp(np.log(src_at_out[out_voiced]) - src_log_mean + out_log_mean)

    # Smooth to remove rapid jumps (source accent prosody ≠ output accent prosody)
    transferred = gaussian_filter1d(transferred, sigma=2.5)

    lo = float(out_f0[out_voiced].min() * f0_clip[0])
    hi = float(out_f0[out_voiced].max() * f0_clip[1])
    new_f0[out_voiced] = np.clip(transferred, max(lo, 50.0), min(hi, 800.0))

    return new_f0


def _world_correct(
    src_f0: np.ndarray,
    src_timeaxis: np.ndarray,
    src_energy: np.ndarray,
    out_wav: np.ndarray,
    out_sr: int,
    f0_clip: tuple,
    energy_clip: tuple,
) -> np.ndarray:
    """PYIN-driven WORLD synthesis for F0 transfer.

    Extracts output F0 with PYIN (robust on neural-vocoded speech), uses it
    to drive WORLD CheapTrick/D4C spectral analysis, applies log-norm F0
    transfer, then re-synthesizes. PYIN eliminates the bad-F0→bad-spectral-
    window chain that made HARVEST-driven WORLD synthesis buzzy.
    """
    src_voiced = src_f0 > 0
    if src_voiced.sum() < 2:
        return _energy_only(src_energy, out_wav, out_sr, energy_clip)

    try:
        # Output F0 via PYIN — accurate on neural vocoder output
        out_f0, out_t = _pyin_f0(out_wav, out_sr)
        out_voiced = out_f0 > 0

        if out_voiced.sum() < 2:
            return _energy_only(src_energy, out_wav, out_sr, energy_clip)

        # WORLD spectral analysis driven by PYIN F0 (NOT HARVEST)
        # Better F0 → CheapTrick uses correct analysis windows → cleaner SP
        x = out_wav.astype(np.float64)
        sp = pyworld.cheaptrick(x, out_f0, out_t, out_sr)   # type: ignore[attr-defined]
        ap = pyworld.d4c(x, out_f0, out_t, out_sr)          # type: ignore[attr-defined]

        # Log-norm F0 transfer
        new_f0 = _log_norm_transfer(src_f0, src_timeaxis, out_f0, out_t, f0_clip)

        # Synthesize with transferred F0 + original SP/AP (keeps accent timbre)
        result = pyworld.synthesize(new_f0, sp, ap, out_sr).astype(np.float32)  # type: ignore[attr-defined]

        # Trim/pad to match input length (WORLD can shift by a few samples)
        tgt = len(out_wav)
        if len(result) >= tgt:
            result = result[:tgt]
        else:
            result = np.pad(result, (0, tgt - len(result)))

        log.info(
            "PYIN+WORLD: %d voiced frames, src_mean=%.0f Hz, out_mean=%.0f Hz",
            int(out_voiced.sum()),
            float(out_f0[out_voiced].mean()),
            float(new_f0[out_voiced].mean()) if new_f0[out_voiced].sum() > 0 else 0.0,
        )

    except Exception as exc:
        log.warning("PYIN+WORLD correction failed (%s) — energy-only fallback", exc)
        result = out_wav.copy()

    return _energy_only(src_energy, result, out_sr, energy_clip)


def _energy_only(
    src_energy: np.ndarray,
    out_wav: np.ndarray,
    out_sr: int,
    energy_clip: tuple,
) -> np.ndarray:
    """Per-frame RMS energy scaling to match source loudness dynamics.

    Non-destructive (multiplication only). Captures 18% of perceived emotion
    (Rizwan 2022) without any risk of synthesis artifacts.
    """
    hop = _PYIN_HOP
    out_rms = librosa.feature.rms(y=out_wav, hop_length=hop)[0]
    src_rs  = np.interp(
        np.linspace(0, 1, len(out_rms)),
        np.linspace(0, 1, len(src_energy)),
        src_energy,
    )
    ratio = np.clip(src_rs / (out_rms + 1e-8), energy_clip[0], energy_clip[1])
    ratio = gaussian_filter1d(ratio.astype(np.float64), sigma=3).astype(np.float32)

    centers = np.arange(len(ratio)) * hop + hop // 2
    if len(centers) > 1 and centers[-1] < len(out_wav) - 1:
        centers = np.append(centers, len(out_wav) - 1)
        ratio   = np.append(ratio,   ratio[-1])

    scale = np.interp(np.arange(len(out_wav)), centers, ratio).astype(np.float32)
    return (out_wav * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmotionCorrector:
    def __init__(self, cfg: dict):
        ec = cfg["emotion_correction"]
        self.method             = ec.get("method", "energy")  # "energy" | "world"
        self.always_correct_f0  = ec.get("always_correct_f0", False)
        self.f0_corr_threshold  = ec.get("f0_corr_threshold", 0.6)
        self.f0_clip            = tuple(ec["f0_scale_clip"])
        self.energy_clip        = tuple(ec["energy_scale_clip"])

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
        """PYIN+WORLD F0 transfer when Seed-VC drifts; energy-only when F0 preserved.

        method="energy" (default): per-frame RMS scaling only. Non-destructive —
        NO re-vocoding, so voice quality and accent (spectral detail) are untouched.
        This is the only correction that survives neural-vocoded Seed-VC output;
        WORLD/PSOLA resynthesis smears formants (accent) and adds buzz (robotic).

        method="world": legacy PYIN+WORLD F0 transfer. Kept for experiments only —
        produces robotic output on neural-vocoded audio. Do not use in production.
        """
        if self.method == "none":
            log.info("emotion correction DISABLED (method=none) — pure accent, raw Seed-VC")
            return wav

        if self.method == "energy":
            log.info("energy-only correction (method=energy, non-destructive)")
            return _energy_only(source_prosody.energy, wav, sr, self.energy_clip)

        should_correct = self.always_correct_f0 or (f0_corr < self.f0_corr_threshold)

        if should_correct:
            log.info("F0+energy correction: f0_corr=%.3f < %.2f → PYIN+WORLD",
                     f0_corr, self.f0_corr_threshold)
            return _world_correct(
                src_f0=source_prosody.f0,
                src_timeaxis=source_prosody.timeaxis,
                src_energy=source_prosody.energy,
                out_wav=wav,
                out_sr=sr,
                f0_clip=self.f0_clip,
                energy_clip=self.energy_clip,
            )

        log.info("energy correction only: f0_corr=%.3f ≥ %.2f (F0 preserved by Seed-VC)",
                 f0_corr, self.f0_corr_threshold)
        return _energy_only(source_prosody.energy, wav, sr, self.energy_clip)
