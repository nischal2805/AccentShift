"""Singleton model manager. Loads heavy shared models once. Never call inside per-segment loops.

Holds Whisper (ASR/WER) + wav2vec2 SER. ECAPA, UTMOS, BigVGAN are lazy (eval-only / on demand).
The two conversion models (Seed-VC, Vevo) are NOT here: they are owned by their backends in
converter.py, which are themselves constructed once before the segment loop.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import torch
import yaml

log = logging.getLogger("model_manager")
ROOT = Path(__file__).resolve().parents[1]


def _ckpt(model_id: str, ckpt_root: Path) -> str:
    """Local snapshot dir for an HF model id (matches download_models layout)."""
    return str(ckpt_root / model_id.replace("/", "__"))


def _src(model_id: str, ckpt_root: Path) -> str:
    local = _ckpt(model_id, ckpt_root)
    return local if Path(local).exists() else model_id


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self, config_path: Optional[str] = None):
        cfg_path = config_path or str(ROOT / "configs" / "pipeline_config.yaml")
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.device = self.cfg["devices"]["main"] if torch.cuda.is_available() else "cpu"
        self.ckpt_root = ROOT / self.cfg["paths"]["checkpoints"]
        self.vevo_device = self._resolve_vevo_device()
        log.info("device=%s vevo_device=%s", self.device, self.vevo_device)
        self._load_whisper()
        self._load_ser()
        self._bigvgan = None
        self._ecapa = None
        self._utmos = None

    @classmethod
    def get(cls, config_path: Optional[str] = None) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def _resolve_vevo_device(self) -> str:
        want = self.cfg["vevo"]["device"]
        if want == "cpu" or not torch.cuda.is_available():
            return "cpu"
        free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda" if free_gb >= self.cfg["devices"]["vram_gb_threshold"] else "cpu"

    def _load_whisper(self) -> None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        src = _src(self.cfg["models"]["whisper"], self.ckpt_root)
        log.info("loading Whisper from %s", src)
        self.whisper_processor = WhisperProcessor.from_pretrained(src)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            src, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _load_ser(self) -> None:
        # audeering MSP-dim model returns (hidden_states, logits) where logits are the
        # [arousal, dominance, valence] regression. AutoModelForAudioClassification loads
        # the published weights; feature_extractor.emotion() reads logits in that order.
        # If the installed transformers version exposes a different forward, adjust there.
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        src = _src(self.cfg["models"]["ser"], self.ckpt_root)
        log.info("loading SER from %s", src)
        self.ser_extractor = AutoFeatureExtractor.from_pretrained(src)
        self.ser = AutoModelForAudioClassification.from_pretrained(src).to(self.device).eval()

    def load_bigvgan(self):
        if self._bigvgan is None:
            from transformers import AutoModel
            src = _src(self.cfg["models"]["bigvgan"], self.ckpt_root)
            self._bigvgan = AutoModel.from_pretrained(
                src, trust_remote_code=True).to(self.device).eval()
        return self._bigvgan

    def load_ecapa(self):
        if self._ecapa is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._ecapa = EncoderClassifier.from_hparams(
                source=self.cfg["models"]["ecapa"],
                run_opts={"device": self.device},
            )
        return self._ecapa

    def load_utmos(self):
        if self._utmos is None:
            self._utmos = torch.hub.load(
                self.cfg["models"]["utmos_hub"], "utmos22_strong", trust_repo=True)
        return self._utmos
