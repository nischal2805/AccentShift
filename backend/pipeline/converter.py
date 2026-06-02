"""Conversion backends. ConverterBackend ABC + real Seed-VC V2 and Vevo wrappers.

Both backends load their (heavy) models ONCE in __init__ and reuse them per segment.
Heavy third-party imports are done lazily inside methods so this module imports cleanly
without the repos' dependencies installed.

Each repo uses relative paths for its configs/checkpoints, so conversion runs inside a
chdir context into the repo directory.
"""
from __future__ import annotations
import abc
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import librosa
import numpy as np
import soundfile as sf
from .types import Segment, Candidate

log = logging.getLogger("converter")
ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _write_tmp(audio: np.ndarray, sr: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, audio, sr)
    f.close()
    return f.name


class ConverterBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        ...


class SeedVCBackend(ConverterBackend):
    """Seed-VC V2 (Plachtaa/seed-vc), accent+style conversion via convert_voice_with_streaming."""
    name = "seed_vc"

    def __init__(self, cfg: dict, device: str):
        self.repo = (ROOT / cfg["seed_vc"]["repo_dir"]).resolve()
        self.cfg = cfg["seed_vc"]
        self.device_str = device
        self.out_sr = cfg["audio"]["sample_rate"]
        if not self.repo.exists():
            raise FileNotFoundError(f"Seed-VC repo missing: {self.repo}. Run download_models.py")
        self._wrapper = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._wrapper is not None:
            return
        import torch
        self._torch = torch
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        with _chdir(self.repo):  # load_v2_models reads relative configs/v2/vc_wrapper.yaml
            from inference_v2 import load_v2_models
            args = SimpleNamespace(
                ar_checkpoint_path=self.cfg["ar_checkpoint_path"],
                cfm_checkpoint_path=self.cfg["cfm_checkpoint_path"],
                compile=self.cfg["compile"],
            )
            self._wrapper = load_v2_models(args)
        log.info("Seed-VC V2 wrapper loaded (device=%s)", self.device_str)

    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        self._ensure_loaded()
        torch = self._torch
        dtype = torch.float16 if self.device_str == "cuda" else torch.float32
        src_path = _write_tmp(source.audio, source.sr)
        ref_abs = str(Path(ref_wav_path).resolve())
        with _chdir(self.repo):
            gen = self._wrapper.convert_voice_with_streaming(
                source_audio_path=src_path,
                target_audio_path=ref_abs,
                diffusion_steps=self.cfg["diffusion_steps"],
                length_adjust=self.cfg["length_adjust"],
                intelligebility_cfg_rate=self.cfg["intelligibility_cfg_rate"],
                similarity_cfg_rate=self.cfg["similarity_cfg_rate"],
                top_p=self.cfg["top_p"],
                temperature=self.cfg["temperature"],
                repetition_penalty=self.cfg["repetition_penalty"],
                convert_style=self.cfg["convert_style"],
                anonymization_only=self.cfg["anonymization_only"],
                device=torch.device(self.device_str),
                dtype=dtype,
                stream_output=True,
            )
            full_audio = None
            for _, full_audio in gen:  # last yield carries the assembled audio
                pass
        if full_audio is None:
            raise RuntimeError("Seed-VC produced no audio")
        sr, wav = full_audio
        wav = np.asarray(wav, dtype=np.float32)
        if sr != self.out_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.out_sr)
        return Candidate(name=self.name, wav=wav, sr=self.out_sr)


class VevoBackend(ConverterBackend):
    """Vevo (open-mmlab/Amphion). Builds VevoInferencePipeline once; convert via inference_ar_and_fm.

    Uses the reference clip for BOTH style and timbre (output is speaker-agnostic, so we adopt
    the target-accent reference's identity rather than preserving the source speaker)."""
    name = "vevo"

    def __init__(self, cfg: dict, device: str):
        self.repo = (ROOT / cfg["vevo"]["repo_dir"]).resolve()
        self.cfg = cfg["vevo"]
        self.device_str = device
        self.out_sr = cfg["audio"]["sample_rate"]
        if not self.repo.exists():
            raise FileNotFoundError(f"Amphion repo missing: {self.repo}. Run download_models.py")
        self._pipeline = None
        self._save_audio = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        self._torch = torch
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        cache = str((ROOT / self.cfg["cache_dir"]).resolve())
        repo_id = self.cfg["hf_repo"]

        def dl(pattern: str) -> str:
            return snapshot_download(repo_id=repo_id, repo_type="model",
                                     cache_dir=cache, allow_patterns=[pattern])

        with _chdir(self.repo):  # config json paths below are relative to repo root
            from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio
            self._save_audio = save_audio
            content_tok = os.path.join(dl("tokenizer/vq32/*"),
                                       "tokenizer/vq32/hubert_large_l18_c32.pkl")
            cs_tok = os.path.join(dl("tokenizer/vq8192/*"), "tokenizer/vq8192")
            ar_dir = dl("contentstyle_modeling/Vq32ToVq8192/*")
            fm_dir = dl("acoustic_modeling/Vq8192ToMels/*")
            voc_dir = dl("acoustic_modeling/Vocoder/*")
            self._pipeline = VevoInferencePipeline(
                content_tokenizer_ckpt_path=content_tok,
                content_style_tokenizer_ckpt_path=cs_tok,
                ar_cfg_path="./models/vc/vevo/config/Vq32ToVq8192.json",
                ar_ckpt_path=os.path.join(ar_dir, "contentstyle_modeling/Vq32ToVq8192"),
                fmt_cfg_path="./models/vc/vevo/config/Vq8192ToMels.json",
                fmt_ckpt_path=os.path.join(fm_dir, "acoustic_modeling/Vq8192ToMels"),
                vocoder_cfg_path="./models/vc/vevo/config/Vocoder.json",
                vocoder_ckpt_path=os.path.join(voc_dir, "acoustic_modeling/Vocoder"),
                device=torch.device(self.device_str),
            )
        log.info("Vevo pipeline loaded (device=%s)", self.device_str)

    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        self._ensure_loaded()
        src_path = _write_tmp(source.audio, source.sr)
        ref_abs = str(Path(ref_wav_path).resolve())
        with _chdir(self.repo):
            gen_audio = self._pipeline.inference_ar_and_fm(
                src_wav_path=src_path,
                src_text=None,
                style_ref_wav_path=ref_abs,
                timbre_ref_wav_path=ref_abs,
                flow_matching_steps=self.cfg["flow_matching_steps"],
            )
        wav = gen_audio.squeeze().detach().cpu().numpy().astype(np.float32)
        vevo_sr = 24000  # save_audio default; Vevo vocoder output rate
        if vevo_sr != self.out_sr:
            wav = librosa.resample(wav, orig_sr=vevo_sr, target_sr=self.out_sr)
        return Candidate(name=self.name, wav=wav, sr=self.out_sr)
