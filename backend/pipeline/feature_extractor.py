"""Per-segment feature extraction: ASR (Whisper), SER (wav2vec2 V/A/D), prosody (pyworld)."""
from __future__ import annotations
import logging
import librosa
import numpy as np
import pyworld
import torch
from .types import Segment, EmotionVec, ProsodyFeatures

log = logging.getLogger("feature_extractor")


class FeatureExtractor:
    def __init__(self, mm):
        self.mm = mm
        self.device = mm.device

    @torch.no_grad()
    def transcribe(self, seg: Segment) -> tuple[str, list[dict]]:
        proc, model = self.mm.whisper_processor, self.mm.whisper
        feats = proc(seg.audio, sampling_rate=seg.sr, return_tensors="pt")
        input_features = feats.input_features.to(self.device, dtype=model.dtype)
        out = model.generate(input_features, language="en", task="transcribe")
        text = proc.batch_decode(out, skip_special_tokens=True)
        text = text[0].strip() if text else ""
        # Word-level timestamps are not required downstream (speaking_rate uses word count).
        return text, []

    @torch.no_grad()
    def emotion(self, seg: Segment) -> EmotionVec:
        """audeering MSP-dim wav2vec2: logits = [arousal, dominance, valence].
        Confirm ordering against the installed model if results look off."""
        ext, model = self.mm.ser_extractor, self.mm.ser
        inputs = ext(seg.audio, sampling_rate=seg.sr, return_tensors="pt")
        iv = {k: v.to(self.device) for k, v in inputs.items()}
        out = model(**iv)
        logits = out.logits if hasattr(out, "logits") else out[1]
        vals = logits.squeeze().float().cpu().numpy().reshape(-1)
        arousal, dominance, valence = float(vals[0]), float(vals[1]), float(vals[2])
        return EmotionVec(valence=valence, arousal=arousal, dominance=dominance)

    def prosody(self, seg: Segment, n_words: int) -> ProsodyFeatures:
        x = seg.audio.astype(np.float64)  # pyworld REQUIRES float64
        f0, timeaxis = pyworld.harvest(x, seg.sr)
        f0 = pyworld.stonemask(x, f0, timeaxis, seg.sr)
        energy = librosa.feature.rms(y=seg.audio)[0]
        rate = n_words / seg.duration_s if seg.duration_s > 0 else 0.0
        return ProsodyFeatures(f0=f0, timeaxis=timeaxis, energy=energy, speaking_rate=rate)
