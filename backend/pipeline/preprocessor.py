"""Load/resample/normalise audio; silero VAD segmentation with 30s hard cap."""
from __future__ import annotations
import logging
from typing import List
import librosa
import numpy as np
import torch
from .types import Segment

log = logging.getLogger("preprocessor")
SR = 16000


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load any supported file, resample to `sr`, mono, peak-normalised float32."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = librosa.util.normalize(audio).astype(np.float32)
    return audio


class Preprocessor:
    def __init__(self, cfg: dict):
        self.sr = cfg["audio"]["sample_rate"]
        self.max_seg = cfg["audio"]["max_segment_s"]
        self.vad_threshold = cfg["vad"]["threshold"]
        self.min_speech_ms = cfg["vad"]["min_speech_ms"]
        self.min_silence_ms = cfg["vad"]["min_silence_ms"]
        self._model = None
        self._get_ts = None

    def _ensure_vad(self) -> None:
        if self._model is None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad()
            self._get_ts = get_speech_timestamps

    def segment(self, audio: np.ndarray) -> List[Segment]:
        """Split into speech segments, never exceeding max_segment_s (Seed-VC hard cap)."""
        self._ensure_vad()
        ts = self._get_ts(
            torch.from_numpy(audio), self._model,
            threshold=self.vad_threshold, sampling_rate=self.sr,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )
        segments: List[Segment] = []
        max_samples = int(self.max_seg * self.sr)
        for t in ts:
            s, e = int(t["start"]), int(t["end"])
            cur = s
            while cur < e:  # force-split runs longer than the cap
                end = min(cur + max_samples, e)
                segments.append(Segment(audio=audio[cur:end].copy(),
                                        start_s=cur / self.sr, end_s=end / self.sr, sr=self.sr))
                cur = end
        if not segments:  # fallback: whole clip (capped) as one segment
            end = min(len(audio), max_samples)
            segments.append(Segment(audio=audio[:end].copy(), start_s=0.0,
                                    end_s=end / self.sr, sr=self.sr))
        log.info("VAD produced %d segments", len(segments))
        return segments
