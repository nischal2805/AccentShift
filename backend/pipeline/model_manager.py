"""Singleton model manager. Loads heavy shared models once. Never call inside per-segment loops.

Holds Whisper (ASR/WER) + wav2vec2 SER. ECAPA, UTMOS, BigVGAN are lazy (eval-only / on demand).
The two conversion models (Seed-VC, Vevo) are NOT here: they are owned by their backends in
converter.py, which are themselves constructed once before the segment loop.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional
import torch
import yaml

log = logging.getLogger("model_manager")
ROOT = Path(__file__).resolve().parents[1]

# Ensure HF_HOME points to the project-local cache so from_pretrained() finds Whisper
# (which is kept in HF cache only, not copied to checkpoints/).
_DEFAULT_HF_HOME = str(ROOT / ".hf_cache")
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = _DEFAULT_HF_HOME


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
        self.whisper_device = self._resolve_whisper_device()
        self.ser_device = self.whisper_device  # SER travels with Whisper
        log.info("device=%s whisper/ser_device=%s vevo_device=%s",
                 self.device, self.whisper_device, self.vevo_device)
        self._load_whisper()
        self._ser = None
        self._ser_extractor = None
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
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda" if vram_gb >= self.cfg["devices"]["vram_gb_threshold"] else "cpu"

    def _resolve_whisper_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        threshold = self.cfg["devices"].get("whisper_vram_gb_threshold", 12)
        return "cuda" if vram_gb >= threshold else "cpu"

    def _load_whisper(self) -> None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        src = _src(self.cfg["models"]["whisper"], self.ckpt_root)
        log.info("loading Whisper from %s on %s", src, self.whisper_device)
        self.whisper_processor = WhisperProcessor.from_pretrained(src)
        dtype = torch.float16 if self.whisper_device == "cuda" else torch.float32
        _w = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=dtype)
        _w.to(self.whisper_device)  # type: ignore[arg-type]
        self.whisper = _w.eval()

    def load_ser(self):
        """Lazy-load SER model. Not called in the main pipeline (EmotionEncoder removed)."""
        if self._ser is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            src = _src(self.cfg["models"]["ser"], self.ckpt_root)
            log.info("loading SER from %s on %s", src, self.ser_device)
            self._ser_extractor = AutoFeatureExtractor.from_pretrained(src)
            _s = AutoModelForAudioClassification.from_pretrained(src)
            _s.to(self.ser_device)  # type: ignore[arg-type]
            self._ser = _s.eval()
        return self._ser

    @property
    def ser(self):
        return self.load_ser()

    @property
    def ser_extractor(self):
        if self._ser_extractor is None:
            self.load_ser()
        return self._ser_extractor

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
            try:
                # torch.hub on Windows may extract to a path where hubconf.py is missing.
                # Force hub_dir inside project to avoid %APPDATA% path issues.
                hub_dir = str(ROOT / ".hf_cache" / "torch_hub")
                torch.hub.set_dir(hub_dir)
                self._utmos = torch.hub.load(
                    self.cfg["models"]["utmos_hub"], "utmos22_strong",
                    trust_repo=True, force_reload=False)
            except Exception as e:  # noqa: BLE001
                log.warning("UTMOS unavailable (%s) — quality scores will use default 3.5", e)
                self._utmos = None
        return self._utmos
