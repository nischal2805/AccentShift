"""Per-segment feature extraction: ASR (Whisper), SER (wav2vec2 V/A/D), prosody (pyworld)."""
from __future__ import annotations
import logging
import librosa
import numpy as np
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
        whisper_dev = self.mm.whisper_device  # cpu on <12GB cards, cuda on big droplets
        feats = proc(seg.audio, sampling_rate=seg.sr, return_tensors="pt")
        input_features = feats.input_features.to(whisper_dev, dtype=model.dtype)
        # Force English transcription — critical for accent-converted speech that may
        # sound like a different language to Whisper's language-detection head.
        forced_decoder_ids = proc.get_decoder_prompt_ids(language="en", task="transcribe")
        out = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        text = proc.batch_decode(out, skip_special_tokens=True)
        text = text[0].strip() if text else ""
        return text, []

    @torch.no_grad()
    def emotion(self, seg: Segment) -> EmotionVec:
        """audeering MSP-dim wav2vec2: logits = [arousal, dominance, valence]."""
        ext, model = self.mm.ser_extractor, self.mm.ser
        ser_dev = self.mm.ser_device  # travels with Whisper (cpu on <12GB, else cuda)
        inputs = ext(seg.audio, sampling_rate=seg.sr, return_tensors="pt")
        iv = {k: v.to(ser_dev) for k, v in inputs.items()}
        out = model(**iv)
        logits = out.logits if hasattr(out, "logits") else out[1]
        # Model outputs raw regression logits, NOT [0,1].  Sigmoid maps to [0,1] so
        # the downstream centering at 0.5 produces a meaningful [-0.5, 0.5] signal.
        vals = torch.sigmoid(logits.squeeze().float()).cpu().numpy().reshape(-1)
        arousal, dominance, valence = float(vals[0]), float(vals[1]), float(vals[2])
        return EmotionVec(valence=valence, arousal=arousal, dominance=dominance)

    def prosody(self, seg: Segment, n_words: int) -> ProsodyFeatures:
        # PYIN (probabilistic YIN) for F0 — more robust than PyWorld HARVEST on
        # emotional/expressive speech (handles high-arousal pitch excursions, breathy voice).
        # Returns 0.0 for unvoiced frames (matching PyWorld convention).
        hop = 256  # 16ms at 16kHz — matches energy RMS hop for aligned arrays
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            y=seg.audio,
            fmin=librosa.note_to_hz("C2"),   # 65 Hz — floor for all voice types
            fmax=librosa.note_to_hz("C7"),   # 2093 Hz — covers excited/child speech
            sr=seg.sr,
            hop_length=hop,
            fill_na=0.0,                     # 0.0 for unvoiced (matches PyWorld)
        )
        timeaxis = np.arange(len(f0)) * hop / seg.sr
        energy = librosa.feature.rms(y=seg.audio, hop_length=hop)[0]
        rate = n_words / seg.duration_s if seg.duration_s > 0 else 0.0
        return ProsodyFeatures(f0=f0, timeaxis=timeaxis, energy=energy, speaking_rate=rate)
