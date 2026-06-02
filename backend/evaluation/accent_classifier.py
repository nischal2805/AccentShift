"""XVector-style accent classifier. Trains a logistic head on ECAPA embeddings; predicts top-1.

Training requires a labelled dataset (VCTK/GLOBE) the user fetches; layout: <accent>/*.wav.
Code is provided here; training is gated on dataset presence.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger("accent_classifier")


class AccentClassifier:
    def __init__(self, mm):
        self.mm = mm
        self._clf = None

    def _embed(self, wav: np.ndarray) -> np.ndarray:
        import torch
        enc = self.mm.load_ecapa()  # reuse ECAPA as the embedding frontend
        emb = enc.encode_batch(torch.from_numpy(wav).unsqueeze(0)).squeeze().cpu().numpy()
        return emb.reshape(-1)

    def train(self, dataset_dir: str, out_path: str) -> None:
        from sklearn.linear_model import LogisticRegression
        import joblib
        import librosa
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Accent dataset missing: {root}")
        X, y = [], []
        for accent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for wav_path in accent_dir.glob("*.wav"):
                wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
                X.append(self._embed(wav))
                y.append(accent_dir.name)
        if not X:
            raise RuntimeError(f"No wavs found under {root}")
        clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
        joblib.dump(clf, out_path)
        log.info("trained accent classifier -> %s (%d samples, %d classes)",
                 out_path, len(y), len(set(y)))

    def load(self, path: str) -> None:
        import joblib
        self._clf = joblib.load(path)

    def predict(self, wav: np.ndarray) -> str:
        if self._clf is None:
            raise RuntimeError("classifier not loaded; call load() or train() first")
        return str(self._clf.predict(self._embed(wav).reshape(1, -1))[0])
