"""Score conversion candidates and pick the winner. Formula locked in config:
    score = emotion_weight*emotion_sim + wer_weight*(1-WER) + mos_weight*(UTMOS/5)
"""
from __future__ import annotations
import logging
import numpy as np
import jiwer
from .types import Segment, Candidate, EmotionVec

log = logging.getLogger("quality_selector")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class QualitySelector:
    def __init__(self, cfg: dict, feature_extractor, mm):
        self.w_emo = cfg["quality"]["emotion_weight"]
        self.w_wer = cfg["quality"]["wer_weight"]
        self.w_mos = cfg["quality"]["mos_weight"]
        self.fe = feature_extractor
        self.mm = mm

    def _utmos(self, wav: np.ndarray, sr: int) -> float:
        try:
            import torch
            model = self.mm.load_utmos()
            if model is None:
                return 3.5
            t = torch.from_numpy(wav).unsqueeze(0).float()
            return float(model(t, sr))
        except Exception as e:  # noqa: BLE001
            log.warning("UTMOS failed (%s); default 3.5", e)
            return 3.5

    def score(self, cand: Candidate, source: Segment,
              source_emotion: EmotionVec, source_text: str) -> tuple[float, dict]:
        cand_seg = Segment(audio=cand.wav, start_s=0.0, end_s=0.0, sr=cand.sr)
        out_emotion = self.fe.emotion(cand_seg)
        emo_sim = _cosine(source_emotion.to_centered_array(), out_emotion.to_centered_array())
        out_text, _ = self.fe.transcribe(cand_seg)
        wer = min(jiwer.wer(source_text, out_text or " "), 1.0) if source_text.strip() else 1.0
        mos = self._utmos(cand.wav, cand.sr)
        score = self.w_emo * emo_sim + self.w_wer * (1 - wer) + self.w_mos * (mos / 5.0)
        return score, {"emotion_sim": emo_sim, "wer": wer, "mos": mos,
                       "emotion_output": out_emotion, "transcript": out_text}

    def select(self, candidates: list[Candidate], source: Segment,
               source_emotion: EmotionVec, source_text: str):
        best = None
        for c in candidates:
            s, meta = self.score(c, source, source_emotion, source_text)
            log.info("%s score=%.3f (emo=%.3f wer=%.3f mos=%.2f)",
                     c.name, s, meta["emotion_sim"], meta["wer"], meta["mos"])
            if best is None or s > best[1]:
                best = (c, s, meta)
        return best  # (Candidate, score, meta)
