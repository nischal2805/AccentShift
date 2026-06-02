"""Evaluation metrics: WER, emotion similarity, UTMOS, speaker agnosticism."""
from __future__ import annotations
import logging
import numpy as np
import jiwer

log = logging.getLogger("metrics")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def word_error_rate(ref_text: str, hyp_text: str) -> float:
    if not ref_text.strip():
        return 1.0
    return min(jiwer.wer(ref_text, hyp_text or " "), 1.0)


def emotion_similarity(src_vec: np.ndarray, out_vec: np.ndarray) -> float:
    return cosine(src_vec, out_vec)


def utmos(mm, wav: np.ndarray, sr: int) -> float:
    import torch
    model = mm.load_utmos()
    t = torch.from_numpy(wav).unsqueeze(0).float()
    return float(model(t, sr))


def speaker_agnosticism(mm, src_wav: np.ndarray, out_wav: np.ndarray, sr: int) -> float:
    """ECAPA embedding cosine between source and output. Target < 0.5 (dissimilar speaker)."""
    import torch
    enc = mm.load_ecapa()
    e1 = enc.encode_batch(torch.from_numpy(src_wav).unsqueeze(0)).squeeze().cpu().numpy()
    e2 = enc.encode_batch(torch.from_numpy(out_wav).unsqueeze(0)).squeeze().cpu().numpy()
    return cosine(e1.reshape(-1), e2.reshape(-1))
