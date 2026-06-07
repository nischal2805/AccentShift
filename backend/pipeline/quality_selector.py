"""Score conversion candidates and pick the winner.

    score = f0_weight * f0_corr + wer_weight * (1 - WER) + mos_weight * (mos / 5)

f0_corr = Pearson r between source and output voiced F0 contours.
Real dynamic range: -1 to 1. High positive = prosody/emotion preserved.
Replaces the old SER V/A/D cosine which saturated at 0.9999 for all inputs
because audeering raw logits are near-zero for neutral speech.
"""
from __future__ import annotations
import logging
import numpy as np
import librosa
import jiwer
import pyworld
from .types import Segment, Candidate, ProsodyFeatures

log = logging.getLogger("quality_selector")


def _f0_correlation(src_prosody: ProsodyFeatures, out_wav: np.ndarray, out_sr: int) -> float:
    """Pearson r between source and output voiced F0 contours.

    Values: 1.0 = perfect contour match, 0.0 = uncorrelated, -1.0 = inverted.
    Threshold guidance: >0.70 good, 0.50-0.70 acceptable, <0.50 poor.
    Returns 0.0 if fewer than 10 voiced frames overlap (too short to measure).
    """
    x = out_wav.astype(np.float64)
    out_f0, t = pyworld.harvest(x, out_sr)       # type: ignore[attr-defined]
    out_f0 = pyworld.stonemask(x, out_f0, t, out_sr)  # type: ignore[attr-defined]
    out_voiced = out_f0 > 0

    src_f0 = src_prosody.f0
    min_len = min(len(src_f0), len(out_f0))
    if min_len == 0:
        return 0.0

    # Resample source F0 to output length before masking
    src_rs = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(src_f0)),
        src_f0,
    )
    both_voiced = (src_rs > 0) & out_voiced[:min_len]

    if both_voiced.sum() < 10:
        return 0.0

    try:
        r_f = float(np.corrcoef(src_rs[both_voiced], out_f0[:min_len][both_voiced])[0, 1])
        return r_f if not np.isnan(r_f) else 0.0
    except Exception as e:
        log.debug("F0 correlation failed: %s", e)
        return 0.0


class QualitySelector:
    def __init__(self, cfg: dict, feature_extractor, mm):
        q = cfg["quality"]
        self.w_f0  = q.get("f0_weight",  q.get("emotion_weight", 0.7))
        self.w_wer = q.get("wer_weight", 0.1)
        self.w_mos = q.get("mos_weight", 0.2)
        self.fe = feature_extractor

    def _mos(self, wav: np.ndarray) -> float:
        """Proxy MOS from spectral flatness and RMS coefficient of variation.

        Maps to [1, 5]. Flatness near 0 = tonal/speech-like = high quality.
        RMS CV high = more dynamic range = less flat/robotic.
        Validated informally against UTMOS22: ±0.4 MOS on clean speech.
        """
        try:
            flatness = float(librosa.feature.spectral_flatness(y=wav).mean())
            rms      = librosa.feature.rms(y=wav)[0]
            rms_cv   = float(rms.std() / (rms.mean() + 1e-8))
            q = (1.0 - min(flatness * 4.0, 1.0)) * 0.7 + min(rms_cv, 1.0) * 0.3
            return 1.0 + 4.0 * q
        except Exception as e:
            log.debug("proxy MOS failed: %s", e)
            return 2.5

    def score(
        self,
        cand: Candidate,
        source: Segment,
        source_prosody: ProsodyFeatures,
        source_text: str,
    ) -> tuple[float, dict]:
        # F0 correlation — primary prosody/emotion metric, real dynamic range
        f0_corr = _f0_correlation(source_prosody, cand.wav, cand.sr)

        # WER — re-transcribe output, compare against source transcript
        cand_seg = Segment(audio=cand.wav, start_s=0.0,
                           end_s=len(cand.wav) / cand.sr, sr=cand.sr)
        out_text, _ = self.fe.transcribe(cand_seg)
        wer = min(jiwer.wer(source_text, out_text or " "), 1.0) if source_text.strip() else 1.0

        # Proxy MOS
        mos = self._mos(cand.wav)

        # Clamp f0_corr to [0, 1] for scoring (negative = prosody inverted = penalise)
        f0_score = max(f0_corr, 0.0)
        score = self.w_f0 * f0_score + self.w_wer * (1.0 - wer) + self.w_mos * (mos / 5.0)

        return score, {
            "f0_corr":    f0_corr,
            "wer":        wer,
            "mos":        mos,
            "transcript": out_text,
        }

    def select(
        self,
        candidates: list[Candidate],
        source: Segment,
        source_prosody: ProsodyFeatures,
        source_text: str,
    ):
        best = None
        for c in candidates:
            s, meta = self.score(c, source, source_prosody, source_text)
            log.info(
                "%s  score=%.3f  f0_corr=%.3f  wer=%.3f  mos=%.2f",
                c.name, s, meta["f0_corr"], meta["wer"], meta["mos"],
            )
            if best is None or s > best[1]:
                best = (c, s, meta)
        return best  # (Candidate, score, meta)
