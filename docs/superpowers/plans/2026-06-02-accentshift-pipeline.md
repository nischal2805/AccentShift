# DP6 AccentShift Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **BUILD-FIRST (user override of TDD):** Do NOT write pytest tests per step. Each task = implement fully → quick import/smoke check → commit. Tests are Task 16 (a dedicated later phase, scaffold only now).

**Goal:** Build the full Architecture-A speech pipeline (English audio → target-accent English, emotion/prosody preserved, speaker-agnostic) plus an evaluation suite, runnable via CLI on an 8GB+ NVIDIA GPU.

**Architecture:** Per-segment pipeline — silero VAD → (Whisper ASR + wav2vec2 SER + pyworld prosody) → Seed-VC V2 + Vevo conversion → quality-score selection → pyworld emotion correction → overlap-add + loudness norm. Real external VC repos (Seed-VC, Amphion) cloned into `third_party/`. Singleton model manager loads all weights once.

**Tech Stack:** Python 3.10, **uv**, PyTorch 2.1/CUDA 12.1, transformers, librosa, soundfile, pyloudnorm, pyworld, silero-vad, jiwer, speechbrain, click, pyyaml.

**Reference docs:** `docs/DP6_AccentShift_Architecture.md`, `docs/backend-spec.md`, `docs/superpowers/specs/2026-06-02-accentshift-pipeline-design.md`.

**Locked constants (never change):** 16kHz mono internal; pyworld float64 cast before every call; Seed-VC ≤30s segments; F0 scale clip [0.8,1.2]; energy scale clip [0.7,1.3]; emotion-correction trigger cos<0.85; quality score `0.4*emotion_sim + 0.4*(1-WER) + 0.2*(UTMOS/5)`; models loaded once in model_manager.

---

## Phase 0 — Project scaffold

### Task 1: uv project, gitignore, setup doc skeleton

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.python-version`
- Create: `backend/.gitignore`
- Create: `backend/SETUP.md`

- [ ] **Step 1: pyproject.toml**

```toml
[project]
name = "accentshift-backend"
version = "0.1.0"
description = "DP6 AccentShift — accent conversion pipeline (Architecture A)"
requires-python = ">=3.10,<3.11"
dependencies = [
    "numpy>=1.24,<2.0",
    "scipy>=1.10",
    "librosa>=0.10.1",
    "soundfile>=0.12.1",
    "pyloudnorm>=0.1.1",
    "pyworld>=0.3.4",
    "silero-vad>=5.1",
    "transformers>=4.40",
    "jiwer>=3.0",
    "speechbrain>=1.0",
    "pyyaml>=6.0",
    "click>=8.1",
    "huggingface-hub>=0.23",
    "tqdm>=4.66",
]

[project.optional-dependencies]
# torch installed separately via CUDA index (see SETUP.md)
dev = ["pytest>=8.0"]

[tool.uv]
# torch/torchaudio pinned + CUDA index handled in SETUP.md, not here

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pipeline", "evaluation"]
```

- [ ] **Step 2: .python-version**

```
3.10
```

- [ ] **Step 3: .gitignore**

```
.venv/
__pycache__/
*.pyc
checkpoints/
third_party/
references/**/*.wav
data/
*.wav
!tests/fixtures/*.wav
.pytest_cache/
```

- [ ] **Step 4: SETUP.md skeleton**

Write headers + uv bootstrap. Fill backend-specific sections in later tasks.

````markdown
# AccentShift Backend — Setup (Windows, 8GB+ NVIDIA GPU)

## 1. Python env (uv)
```powershell
cd backend
uv venv --python 3.10
.venv\Scripts\activate
uv sync
# Torch with CUDA 12.1 (separate index):
uv pip install torch==2.1.* torchaudio==2.1.* --index-url https://download.pytorch.org/whl/cu121
```

## 2. Download models
(see Task: download_models)

## 3. Seed-VC / Vevo clones + Windows fallbacks
(filled in converter tasks)

## 4. Finetune (A100 only)
(filled in finetune task)
````

- [ ] **Step 5: Smoke check**

Run: `cd backend && uv lock`
Expected: `uv.lock` generated, no resolution error. (If a dep fails to resolve, loosen its pin and re-run.)

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/.python-version backend/.gitignore backend/SETUP.md backend/uv.lock
git commit -m "chore: uv project scaffold for accentshift backend"
```

---

### Task 2: Config files

**Files:**
- Create: `backend/configs/pipeline_config.yaml`
- Create: `backend/configs/finetune_config.yaml`

- [ ] **Step 1: pipeline_config.yaml**

```yaml
audio:
  sample_rate: 16000
  mono: true
  max_segment_s: 30.0

vad:
  threshold: 0.5
  min_speech_ms: 250
  min_silence_ms: 100

models:
  whisper: "openai/whisper-large-v3"
  ser: "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
  bigvgan: "nvidia/bigvgan-v2-24khz-100band-256x"
  utmos_hub: "sarulab-speech/UTMOS22"
  ecapa: "speechbrain/spkrec-ecapa-voxceleb"

seed_vc:
  repo_dir: "third_party/seed-vc"
  convert_style: true
  intelligibility_cfg_rate: 0.7
  similarity_cfg_rate: 0.7
  diffusion_steps: 30
  device: "cuda"

vevo:
  repo_dir: "third_party/Amphion"
  enabled: true
  device: "cpu"          # 8GB card: Vevo on CPU

devices:
  main: "cuda"
  vram_gb_threshold: 10  # below this, Vevo forced to CPU

quality:
  emotion_weight: 0.4
  wer_weight: 0.4
  mos_weight: 0.2

emotion_correction:
  threshold: 0.85
  f0_scale_clip: [0.8, 1.2]
  energy_scale_clip: [0.7, 1.3]

postprocess:
  crossfade_ms: 30
  target_lufs: -23.0

paths:
  checkpoints: "checkpoints"
  references: "references"

accents:
  - { key: "indian_english",   label: "Indian English" }
  - { key: "chinese_english",  label: "Chinese-accented English" }
  - { key: "japanese_english", label: "Japanese-accented English" }
  - { key: "british_english",  label: "British English" }
  - { key: "american_english", label: "American English" }
```

- [ ] **Step 2: finetune_config.yaml** (per spec — A100, never run here)

```yaml
# RUN ON A100. Do not execute locally.
model:
  finetune_target: "style_encoder"   # Seed-VC style encoder only (~50M params)
  freeze: ["content_encoder", "vocoder"]
optimizer:
  type: "AdamW"
  lr_style_encoder: 1.0e-4
  lr_ar_decoder: 1.0e-5
  weight_decay: 0.01
schedule:
  type: "cosine_with_warmup"
  warmup_steps: 2000
  total_steps: 50000
training:
  batch_size: 16
  gradient_accumulation: 2   # effective batch 32
  mixed_precision: "bf16"
  gradient_checkpointing: true
data:
  dataset: "l2arctic_parallel"
  prepared_dir: "data/l2arctic_pairs"
output:
  checkpoint_dir: "checkpoints/seedvc_finetuned"
```

- [ ] **Step 3: Commit**

```bash
git add backend/configs/
git commit -m "chore: pipeline and finetune config yaml"
```

---

## Phase 1 — Core types + model loading

### Task 3: Shared types

**Files:**
- Create: `backend/pipeline/__init__.py` (empty)
- Create: `backend/pipeline/types.py`

- [ ] **Step 1: types.py**

```python
"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Segment:
    audio: np.ndarray          # float32, 16kHz mono
    start_s: float
    end_s: float
    sr: int = 16000

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sr


@dataclass
class EmotionVec:
    valence: float
    arousal: float
    dominance: float

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance], dtype=np.float64)


@dataclass
class ProsodyFeatures:
    f0: np.ndarray             # float64
    timeaxis: np.ndarray
    energy: np.ndarray
    speaking_rate: float       # words per second


@dataclass
class Candidate:
    name: str                  # "seed_vc" | "vevo"
    wav: np.ndarray
    sr: int


@dataclass
class SegmentResult:
    wav: np.ndarray
    sr: int
    emotion_source: EmotionVec
    emotion_output: EmotionVec
    transcript: str
    wer: float
    mos: float
    chosen_backend: str
    score: float
    start_s: float
    end_s: float
```

- [ ] **Step 2: Smoke check**

Run: `cd backend && uv run python -c "from pipeline.types import Segment, EmotionVec; import numpy as np; print(EmotionVec(0.1,0.2,0.3).to_array())"`
Expected: `[0.1 0.2 0.3]`

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/__init__.py backend/pipeline/types.py
git commit -m "feat: shared pipeline dataclasses"
```

---

### Task 4: download_models.py

**Files:**
- Create: `backend/scripts/__init__.py` (empty)
- Create: `backend/scripts/download_models.py`

- [ ] **Step 1: Implement** — idempotent, logged. HF snapshots for Whisper/SER/BigVGAN/ECAPA into `checkpoints/`; pre-fetch UTMOS via torch.hub; git-clone Seed-VC + Amphion into `third_party/`.

```python
"""Pull all pretrained weights + clone VC repos. Idempotent. Run once after env setup."""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_models")

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "pipeline_config.yaml").read_text())

HF_MODELS = [CFG["models"]["whisper"], CFG["models"]["ser"],
             CFG["models"]["bigvgan"], CFG["models"]["ecapa"]]
REPOS = {
    "seed-vc": "https://github.com/Plachtaa/seed-vc.git",
    "Amphion": "https://github.com/open-mmlab/Amphion.git",
}


def fetch_hf(model_id: str, dest_root: Path) -> None:
    dest = dest_root / model_id.replace("/", "__")
    if dest.exists() and any(dest.iterdir()):
        log.info("skip (present): %s", model_id); return
    log.info("downloading %s", model_id)
    snapshot_download(repo_id=model_id, local_dir=str(dest), local_dir_use_symlinks=False)


def clone_repo(name: str, url: str, dest_root: Path) -> None:
    dest = dest_root / name
    if dest.exists():
        log.info("skip clone (present): %s", name); return
    log.info("cloning %s", url)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def prefetch_utmos() -> None:
    try:
        import torch
        torch.hub.load("sarulab-speech/UTMOS22", "utmos22_strong", trust_repo=True)
        log.info("UTMOS prefetched")
    except Exception as e:  # noqa: BLE001
        log.warning("UTMOS prefetch failed (fetch at first eval): %s", e)


def main() -> None:
    ckpt = ROOT / CFG["paths"]["checkpoints"]; ckpt.mkdir(parents=True, exist_ok=True)
    tp = ROOT / "third_party"; tp.mkdir(parents=True, exist_ok=True)
    for m in HF_MODELS:
        fetch_hf(m, ckpt)
    for name, url in REPOS.items():
        clone_repo(name, url, tp)
    prefetch_utmos()
    log.info("done. Next: follow each repo's checkpoint-download step in SETUP.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: SETUP.md** — fill section 2 with `uv run python scripts/download_models.py` and a note that Seed-VC/Amphion may need their own checkpoint pulls (linked from their READMEs).

- [ ] **Step 3: Smoke check** (no network needed): `cd backend && uv run python -c "import scripts.download_models as d; print(len(d.HF_MODELS), list(d.REPOS))"`
Expected: `4 ['seed-vc', 'Amphion']`

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/download_models.py backend/SETUP.md
git commit -m "feat: model + repo download script"
```

---

### Task 5: model_manager.py (singleton loader)

**Files:**
- Create: `backend/pipeline/model_manager.py`

- [ ] **Step 1: Implement** — singleton; load Whisper, SER, BigVGAN, ECAPA at init; lazy-resolve VRAM to decide Vevo device; expose typed handles. Seed-VC/Vevo model objects are created by their backends (Task 8) using paths from config — manager holds config + device map + the HF models.

```python
"""Singleton model manager. Loads heavy models once. Never call inside per-segment loops."""
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
        self._load_bigvgan()
        self._ecapa = None  # lazy: only eval needs it
        self._utmos = None  # lazy

    # ---- singleton accessor ----
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
        thresh = self.cfg["devices"]["vram_gb_threshold"]
        return "cuda" if free_gb >= thresh else "cpu"

    def _load_whisper(self) -> None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        mid = self.cfg["models"]["whisper"]; local = _ckpt(mid, self.ckpt_root)
        src = local if Path(local).exists() else mid
        log.info("loading Whisper from %s", src)
        self.whisper_processor = WhisperProcessor.from_pretrained(src)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            src, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _load_ser(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        mid = self.cfg["models"]["ser"]; local = _ckpt(mid, self.ckpt_root)
        src = local if Path(local).exists() else mid
        log.info("loading SER from %s", src)
        self.ser_extractor = AutoFeatureExtractor.from_pretrained(src)
        # NOTE: audeering model is a custom regression head (valence/arousal/dominance).
        # Task 7 wraps the correct forward; here we just hold the model + extractor.
        self.ser = AutoModelForAudioClassification.from_pretrained(src).to(self.device).eval()

    def _load_bigvgan(self) -> None:
        # BigVGAN-v2 loads via its HF code (trust_remote_code). Only needed if a backend
        # returns mel rather than wav; Seed-VC/Vevo vocode internally by default.
        self.bigvgan = None
        log.info("BigVGAN lazy — backends vocode internally; load on demand if mel returned")

    def load_bigvgan(self):
        if self.bigvgan is None:
            from transformers import AutoModel
            mid = self.cfg["models"]["bigvgan"]; local = _ckpt(mid, self.ckpt_root)
            src = local if Path(local).exists() else mid
            self.bigvgan = AutoModel.from_pretrained(src, trust_remote_code=True).to(self.device).eval()
        return self.bigvgan

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
            self._utmos = torch.hub.load(self.cfg["models"]["utmos_hub"],
                                         "utmos22_strong", trust_repo=True)
        return self._utmos
```

> **EXECUTION NOTE:** The audeering SER model uses a custom architecture (Wav2Vec2 + regression head returning hidden states + 3 logits). When implementing Task 7, open the model card / `config.json` in `checkpoints/audeering__.../` and confirm the forward signature. If `AutoModelForAudioClassification` does not expose the V/A/D regression output, switch to `Wav2Vec2PreTrainedModel` custom class from the model card README and load weights manually. Adjust `_load_ser` accordingly.

- [ ] **Step 2: Smoke check** (offline): `cd backend && uv run python -c "import pipeline.model_manager as m; print('import ok')"`
Expected: `import ok` (no model load until `.get()` called with weights present).

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/model_manager.py
git commit -m "feat: singleton model manager with VRAM-aware device map"
```

---

## Phase 2 — Feature extraction & preprocessing

### Task 6: preprocessor.py

**Files:**
- Create: `backend/pipeline/preprocessor.py`

- [ ] **Step 1: Implement**

```python
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

    def _ensure_vad(self):
        if self._model is None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad()
            self._get_ts = get_speech_timestamps

    def segment(self, audio: np.ndarray) -> List[Segment]:
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
            # hard-cap split: never exceed max_segment_s
            cur = s
            while cur < e:
                end = min(cur + max_samples, e)
                segments.append(Segment(audio=audio[cur:end].copy(),
                                        start_s=cur / self.sr, end_s=end / self.sr, sr=self.sr))
                cur = end
        if not segments:  # fallback: whole clip as one (capped) segment
            segments.append(Segment(audio=audio[:max_samples].copy(), start_s=0.0,
                                    end_s=min(len(audio), max_samples) / self.sr, sr=self.sr))
        log.info("VAD produced %d segments", len(segments))
        return segments
```

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "from pipeline.preprocessor import Preprocessor; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/preprocessor.py
git commit -m "feat: preprocessor with silero VAD and 30s segment cap"
```

---

### Task 7: feature_extractor.py

**Files:**
- Create: `backend/pipeline/feature_extractor.py`

- [ ] **Step 1: Implement** — Whisper transcript+word timestamps, SER V/A/D, pyworld prosody. Uses ModelManager handles.

```python
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
        out = model.generate(input_features, language="en", task="transcribe",
                             return_timestamps="word", return_dict_in_generate=True)
        decoded = proc.batch_decode(out.sequences, skip_special_tokens=True,
                                    output_word_offsets=False)
        text = decoded[0].strip() if decoded else ""
        # word timestamps via processor offsets (best-effort; empty list ok)
        words: list[dict] = []
        return text, words

    @torch.no_grad()
    def emotion(self, seg: Segment) -> EmotionVec:
        """audeering wav2vec2: returns (hidden_states, logits) with logits = [arousal, dominance, valence].
        Confirm ordering against model card during execution (see model_manager NOTE)."""
        ext, model = self.mm.ser_extractor, self.mm.ser
        inputs = ext(seg.audio, sampling_rate=seg.sr, return_tensors="pt")
        iv = {k: v.to(self.device) for k, v in inputs.items()}
        out = model(**iv)
        logits = out.logits if hasattr(out, "logits") else out[1]
        vals = logits.squeeze().float().cpu().numpy()
        # audeering MSP-dim order = [arousal, dominance, valence]
        arousal, dominance, valence = float(vals[0]), float(vals[1]), float(vals[2])
        return EmotionVec(valence=valence, arousal=arousal, dominance=dominance)

    def prosody(self, seg: Segment, n_words: int) -> ProsodyFeatures:
        x = seg.audio.astype(np.float64)  # pyworld REQUIRES float64
        f0, timeaxis = pyworld.harvest(x, seg.sr)
        f0 = pyworld.stonemask(x, f0, timeaxis, seg.sr)
        energy = librosa.feature.rms(y=seg.audio)[0]
        rate = n_words / seg.duration_s if seg.duration_s > 0 else 0.0
        return ProsodyFeatures(f0=f0, timeaxis=timeaxis, energy=energy, speaking_rate=rate)
```

> **EXECUTION NOTE:** Verify the audeering SER output ordering (arousal/dominance/valence) and Whisper word-timestamp extraction against the actual installed versions. Adjust `emotion()` indexing and `transcribe()` offsets if the API differs. Word timestamps feed only `speaking_rate`; if unavailable, use whitespace word count of `text`.

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "from pipeline.feature_extractor import FeatureExtractor; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/feature_extractor.py
git commit -m "feat: feature extractor (Whisper ASR, SER V/A/D, pyworld prosody)"
```

---

## Phase 3 — Conversion backends (external repos)

### Task 8: converter.py — Seed-VC backend

**Files:**
- Create: `backend/pipeline/converter.py`
- Modify: `backend/SETUP.md` (Seed-VC section)

- [ ] **Step 1: Study the cloned repo.** Open `third_party/seed-vc/`. Find the inference entrypoint (README shows a CLI like `inference_v2.py` / `inference.py` with `--source --target --convert-style`). Identify whether a Python-importable function exists or only a CLI. Note the exact arg names and the output wav path/sample rate.

- [ ] **Step 2: Implement ABC + SeedVCBackend.** Prefer importing Seed-VC's inference function; if only a CLI exists, shell out via subprocess to temp files. Use config rates.

```python
"""Conversion backends. ConverterBackend ABC + real Seed-VC V2 and Vevo wrappers."""
from __future__ import annotations
import abc
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from .types import Segment, Candidate

log = logging.getLogger("converter")
ROOT = Path(__file__).resolve().parents[1]


class ConverterBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        ...


def _write_tmp(audio: np.ndarray, sr: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, audio, sr)
    return f.name


class SeedVCBackend(ConverterBackend):
    name = "seed_vc"

    def __init__(self, cfg: dict, device: str):
        self.repo = ROOT / cfg["seed_vc"]["repo_dir"]
        self.cfg = cfg["seed_vc"]
        self.device = device
        self.sr = cfg["audio"]["sample_rate"]
        if not self.repo.exists():
            raise FileNotFoundError(f"Seed-VC repo missing: {self.repo}. Run download_models.py")

    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        src_path = _write_tmp(source.audio, source.sr)
        out_dir = tempfile.mkdtemp()
        # CLI invocation — ADJUST script name + flags to match the cloned repo (Step 1).
        cmd = [
            sys.executable, "inference_v2.py",
            "--source", src_path, "--target", ref_wav_path,
            "--output", out_dir,
            "--diffusion-steps", str(self.cfg["diffusion_steps"]),
            "--intelligibility-cfg-rate", str(self.cfg["intelligibility_cfg_rate"]),
            "--similarity-cfg-rate", str(self.cfg["similarity_cfg_rate"]),
        ]
        if self.cfg["convert_style"]:
            cmd += ["--convert-style", "true"]
        log.info("seed-vc: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(self.repo), check=True)
        out_wav = next(Path(out_dir).glob("*.wav"))
        wav, sr = librosa.load(str(out_wav), sr=self.sr, mono=True)
        return Candidate(name=self.name, wav=wav.astype(np.float32), sr=self.sr)
```

- [ ] **Step 3: SETUP.md** — document Seed-VC install: `cd third_party/seed-vc && uv pip install -r requirements.txt`; Windows fallbacks (skip flash-attn; if deepspeed fails, install CPU build or comment out; checkpoint auto-download note).

- [ ] **Step 4: Smoke check**: `cd backend && uv run python -c "from pipeline.converter import ConverterBackend, SeedVCBackend; print('ok')"`
Expected: `ok` (no model load).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/converter.py backend/SETUP.md
git commit -m "feat: converter ABC + Seed-VC backend wrapper"
```

---

### Task 9: converter.py — Vevo backend

**Files:**
- Modify: `backend/pipeline/converter.py`
- Modify: `backend/SETUP.md` (Vevo/Amphion section)

- [ ] **Step 1: Study repo.** Open `third_party/Amphion/`. Locate `models/vc/vevo/infer_vevovoice.py` (or current path). Note module path, `--src`/`--ref` args, output location, device handling.

- [ ] **Step 2: Append VevoBackend** to `converter.py`:

```python
class VevoBackend(ConverterBackend):
    name = "vevo"

    def __init__(self, cfg: dict, device: str):
        self.repo = ROOT / cfg["vevo"]["repo_dir"]
        self.device = device  # may be "cpu" on 8GB cards
        self.sr = cfg["audio"]["sample_rate"]
        if not self.repo.exists():
            raise FileNotFoundError(f"Amphion repo missing: {self.repo}. Run download_models.py")

    def convert(self, source: Segment, ref_wav_path: str) -> Candidate:
        src_path = _write_tmp(source.audio, source.sr)
        out_dir = tempfile.mkdtemp()
        # ADJUST module path + flags to match cloned Amphion (Step 1).
        cmd = [
            sys.executable, "-m", "models.vc.vevo.infer_vevovoice",
            "--src", src_path, "--ref", ref_wav_path, "--output_dir", out_dir,
        ]
        env_note = "CUDA_VISIBLE_DEVICES=-1 " if self.device == "cpu" else ""
        log.info("vevo: %s%s", env_note, " ".join(cmd))
        import os
        env = dict(os.environ)
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        subprocess.run(cmd, cwd=str(self.repo), check=True, env=env)
        out_wav = next(Path(out_dir).glob("*.wav"))
        wav, sr = librosa.load(str(out_wav), sr=self.sr, mono=True)
        return Candidate(name=self.name, wav=wav.astype(np.float32), sr=self.sr)
```

- [ ] **Step 3: SETUP.md** — Amphion install + Windows fallbacks; note Vevo runs on CPU per config on 8GB cards (slower).

- [ ] **Step 4: Smoke check**: `cd backend && uv run python -c "from pipeline.converter import VevoBackend; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/converter.py backend/SETUP.md
git commit -m "feat: Vevo backend wrapper (CPU-capable)"
```

---

## Phase 4 — Selection, correction, assembly

### Task 10: quality_selector.py

**Files:**
- Create: `backend/pipeline/quality_selector.py`

- [ ] **Step 1: Implement** — score per locked formula, pick winner. Reuses FeatureExtractor for SER on candidates and Whisper for WER; UTMOS from model_manager.

```python
"""Score conversion candidates and pick the winner. Formula locked in config."""
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
            t = torch.from_numpy(wav).unsqueeze(0).float()
            return float(model(t, sr))
        except Exception as e:  # noqa: BLE001
            log.warning("UTMOS failed (%s); default 3.5", e)
            return 3.5

    def score(self, cand: Candidate, source: Segment,
              source_emotion: EmotionVec, source_text: str) -> tuple[float, dict]:
        cand_seg = Segment(audio=cand.wav, start_s=0, end_s=0, sr=cand.sr)
        out_emotion = self.fe.emotion(cand_seg)
        emo_sim = _cosine(source_emotion.to_array(), out_emotion.to_array())
        out_text, _ = self.fe.transcribe(cand_seg)
        wer = jiwer.wer(source_text or " ", out_text or " ") if source_text else 1.0
        wer = min(wer, 1.0)
        mos = self._utmos(cand.wav, cand.sr)
        score = self.w_emo * emo_sim + self.w_wer * (1 - wer) + self.w_mos * (mos / 5.0)
        return score, {"emotion_sim": emo_sim, "wer": wer, "mos": mos,
                       "emotion_output": out_emotion, "transcript": out_text}

    def select(self, candidates: list[Candidate], source: Segment,
               source_emotion: EmotionVec, source_text: str):
        best = None
        for c in candidates:
            s, meta = self.score(c, source, source_emotion, source_text)
            log.info("%s score=%.3f", c.name, s)
            if best is None or s > best[1]:
                best = (c, s, meta)
        return best  # (Candidate, score, meta)
```

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "from pipeline.quality_selector import QualitySelector, _cosine; import numpy as np; print(round(_cosine(np.array([1,0,0]),np.array([1,0,0])),2))"`
Expected: `1.0`

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/quality_selector.py
git commit -m "feat: quality selector with locked scoring formula"
```

---

### Task 11: emotion_corrector.py

**Files:**
- Create: `backend/pipeline/emotion_corrector.py`

- [ ] **Step 1: Implement** — pyworld F0 rescale + energy warp, clipped per config; no-op above threshold.

```python
"""Emotion correction: warp output F0/energy toward source when emotion drifts."""
from __future__ import annotations
import logging
import librosa
import numpy as np
import pyworld
from .types import EmotionVec, ProsodyFeatures

log = logging.getLogger("emotion_corrector")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


class EmotionCorrector:
    def __init__(self, cfg: dict):
        self.threshold = cfg["emotion_correction"]["threshold"]
        self.f0_clip = tuple(cfg["emotion_correction"]["f0_scale_clip"])
        self.energy_clip = tuple(cfg["emotion_correction"]["energy_scale_clip"])

    def correct(self, wav: np.ndarray, sr: int, source_emotion: EmotionVec,
                output_emotion: EmotionVec, source_prosody: ProsodyFeatures) -> np.ndarray:
        sim = _cosine(source_emotion.to_array(), output_emotion.to_array())
        if sim >= self.threshold:
            return wav  # no correction needed
        log.info("emotion drift (cos=%.3f < %.2f) — correcting", sim, self.threshold)
        x = wav.astype(np.float64)  # pyworld REQUIRES float64
        f0, t = pyworld.harvest(x, sr)
        f0 = pyworld.stonemask(x, f0, t, sr)
        voiced = f0 > 0
        src_f0 = source_prosody.f0
        src_voiced = src_f0[src_f0 > 0]
        if voiced.sum() > 0 and src_voiced.size > 0:
            scale = src_voiced.mean() / f0[voiced].mean()
            f0[voiced] *= np.clip(scale, self.f0_clip[0], self.f0_clip[1])
        sp = pyworld.cheaptrick(x, f0, t, sr)
        ap = pyworld.d4c(x, f0, t, sr)
        corrected = pyworld.synthesize(f0, sp, ap, sr).astype(np.float32)
        # energy warp toward source RMS
        out_rms = librosa.feature.rms(y=corrected)[0].mean()
        src_rms = source_prosody.energy.mean()
        if out_rms > 0:
            escale = np.clip(src_rms / out_rms, self.energy_clip[0], self.energy_clip[1])
            corrected = (corrected * escale).astype(np.float32)
        return corrected
```

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "from pipeline.emotion_corrector import EmotionCorrector; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/emotion_corrector.py
git commit -m "feat: pyworld emotion corrector (F0 + energy warp)"
```

---

### Task 12: postprocessor.py

**Files:**
- Create: `backend/pipeline/postprocessor.py`

- [ ] **Step 1: Implement** — overlap-add crossfade assembly + pyloudnorm.

```python
"""Assemble segments (overlap-add crossfade) and loudness-normalise to target LUFS."""
from __future__ import annotations
import logging
import numpy as np
import pyloudnorm as pyln

log = logging.getLogger("postprocessor")


class Postprocessor:
    def __init__(self, cfg: dict):
        self.sr = cfg["audio"]["sample_rate"]
        self.crossfade_ms = cfg["postprocess"]["crossfade_ms"]
        self.target_lufs = cfg["postprocess"]["target_lufs"]

    def assemble(self, wavs: list[np.ndarray]) -> np.ndarray:
        if not wavs:
            return np.zeros(0, dtype=np.float32)
        if len(wavs) == 1:
            return wavs[0].astype(np.float32)
        xf = int(self.crossfade_ms / 1000 * self.sr)
        out = wavs[0].astype(np.float32)
        for nxt in wavs[1:]:
            nxt = nxt.astype(np.float32)
            n = min(xf, len(out), len(nxt))
            if n > 0:
                fade = np.linspace(1, 0, n, dtype=np.float32)
                out[-n:] = out[-n:] * fade + nxt[:n] * (1 - fade)
                out = np.concatenate([out, nxt[n:]])
            else:
                out = np.concatenate([out, nxt])
        return out

    def loudness_normalize(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.sr:  # too short for reliable measurement
            return audio
        meter = pyln.Meter(self.sr)
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        normed = pyln.normalize.loudness(audio.astype(np.float64), loudness, self.target_lufs)
        peak = np.abs(normed).max()
        if peak > 1.0:
            normed = normed / peak
        return normed.astype(np.float32)
```

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "from pipeline.postprocessor import Postprocessor; import numpy as np; p=Postprocessor({'audio':{'sample_rate':16000},'postprocess':{'crossfade_ms':30,'target_lufs':-23.0}}); print(len(p.assemble([np.ones(8000,dtype=np.float32),np.ones(8000,dtype=np.float32)])))"`
Expected: an int near `15520` (16000 - crossfade overlap).

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/postprocessor.py
git commit -m "feat: postprocessor (overlap-add + loudness norm)"
```

---

## Phase 5 — Orchestrator CLI

### Task 13: run_pipeline.py

**Files:**
- Create: `backend/run_pipeline.py`

- [ ] **Step 1: Implement** — wires the full flow per backend-spec §Pipeline Flow. click CLI: `--input`, `--target-accent`, `--output`, `--metrics-out`.

```python
"""End-to-end CLI driver: wav in -> accent-converted wav + metrics json out."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import click
import numpy as np
import soundfile as sf
import yaml

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor, load_audio
from pipeline.feature_extractor import FeatureExtractor
from pipeline.converter import SeedVCBackend, VevoBackend
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_pipeline")
ROOT = Path(__file__).resolve().parent


def pick_reference(cfg: dict, accent: str) -> str:
    ref_dir = ROOT / cfg["paths"]["references"] / accent
    wavs = sorted(ref_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No reference clips in {ref_dir}. Add 5-10s WAVs.")
    return str(wavs[0])


@click.command()
@click.option("--input", "input_path", required=True)
@click.option("--target-accent", required=True)
@click.option("--output", "output_path", required=True)
@click.option("--metrics-out", default=None)
@click.option("--config", "config_path", default=str(ROOT / "configs" / "pipeline_config.yaml"))
def main(input_path, target_accent, output_path, metrics_out, config_path):
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    valid = {a["key"] for a in cfg["accents"]}
    if target_accent not in valid:
        raise click.BadParameter(f"target_accent must be one of {sorted(valid)}")

    mm = ModelManager.get(config_path)
    pre = Preprocessor(cfg)
    fe = FeatureExtractor(mm)
    qs = QualitySelector(cfg, fe, mm)
    ec = EmotionCorrector(cfg)
    post = Postprocessor(cfg)

    backends = [SeedVCBackend(cfg, mm.device)]
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))
    ref = pick_reference(cfg, target_accent)

    audio = load_audio(input_path, sr=cfg["audio"]["sample_rate"])
    segments = pre.segment(audio)

    out_wavs, seg_metrics = [], []
    for i, seg in enumerate(segments):
        text, words = fe.transcribe(seg)
        src_emotion = fe.emotion(seg)
        prosody = fe.prosody(seg, n_words=len(text.split()))
        candidates = [b.convert(seg, ref) for b in backends]
        cand, score, meta = qs.select(candidates, seg, src_emotion, text)
        corrected = ec.correct(cand.wav, cand.sr, src_emotion,
                               meta["emotion_output"], prosody)
        out_wavs.append(corrected)
        seg_metrics.append({
            "segment": i, "chosen_backend": cand.name, "score": score,
            "emotion_sim": meta["emotion_sim"], "wer": meta["wer"], "mos": meta["mos"],
            "valence_source": src_emotion.valence, "valence_output": meta["emotion_output"].valence,
            "arousal_source": src_emotion.arousal, "arousal_output": meta["emotion_output"].arousal,
            "dominance_source": src_emotion.dominance, "dominance_output": meta["emotion_output"].dominance,
        })

    final = post.assemble(out_wavs)
    final = post.loudness_normalize(final)
    sf.write(output_path, final, cfg["audio"]["sample_rate"], subtype="PCM_16")

    metrics = {
        "processing_time_ms": int((time.time() - t0) * 1000),
        "n_segments": len(segments),
        "segments": seg_metrics,
        "emotion_similarity": float(np.mean([m["emotion_sim"] for m in seg_metrics])) if seg_metrics else 0.0,
        "wer": float(np.mean([m["wer"] for m in seg_metrics])) if seg_metrics else 1.0,
        "mos_estimate": float(np.mean([m["mos"] for m in seg_metrics])) if seg_metrics else 0.0,
    }
    if metrics_out:
        Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    log.info("done in %dms -> %s", metrics["processing_time_ms"], output_path)
    click.echo(json.dumps({k: metrics[k] for k in
               ("processing_time_ms", "n_segments", "emotion_similarity", "wer", "mos_estimate")}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke check** (import only, no weights): `cd backend && uv run python -c "import run_pipeline; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/run_pipeline.py
git commit -m "feat: end-to-end pipeline CLI orchestrator"
```

---

## Phase 6 — Evaluation suite

### Task 14: evaluation metrics + accent classifier

**Files:**
- Create: `backend/evaluation/__init__.py` (empty)
- Create: `backend/evaluation/metrics.py`
- Create: `backend/evaluation/accent_classifier.py`

- [ ] **Step 1: metrics.py**

```python
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
    return min(jiwer.wer(ref_text, hyp_text), 1.0)


def emotion_similarity(src_vec: np.ndarray, out_vec: np.ndarray) -> float:
    return cosine(src_vec, out_vec)


def utmos(mm, wav: np.ndarray, sr: int) -> float:
    import torch
    model = mm.load_utmos()
    t = torch.from_numpy(wav).unsqueeze(0).float()
    return float(model(t, sr))


def speaker_agnosticism(mm, src_wav: np.ndarray, out_wav: np.ndarray, sr: int) -> float:
    """ECAPA cosine. Target < 0.5 (output speaker dissimilar to source)."""
    import torch
    enc = mm.load_ecapa()
    e1 = enc.encode_batch(torch.from_numpy(src_wav).unsqueeze(0)).squeeze().cpu().numpy()
    e2 = enc.encode_batch(torch.from_numpy(out_wav).unsqueeze(0)).squeeze().cpu().numpy()
    return cosine(e1, e2)
```

- [ ] **Step 2: accent_classifier.py** — XVector (speechbrain) accent classifier; train + predict. Training gated on dataset presence.

```python
"""XVector accent classifier (speechbrain). Train on VCTK/GLOBE labels; predict top-1.
Training requires a labelled dataset the user fetches; code is gated on its presence."""
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
        enc = self.mm.load_ecapa()  # reuse ECAPA as embedding frontend
        return enc.encode_batch(torch.from_numpy(wav).unsqueeze(0)).squeeze().cpu().numpy()

    def train(self, dataset_dir: str, out_path: str) -> None:
        """dataset_dir: <accent>/*.wav. Trains a logistic-regression head on ECAPA embeddings."""
        from sklearn.linear_model import LogisticRegression
        import joblib, librosa
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Accent dataset missing: {root}")
        X, y = [], []
        for accent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for wav_path in accent_dir.glob("*.wav"):
                wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
                X.append(self._embed(wav)); y.append(accent_dir.name)
        clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
        joblib.dump(clf, out_path)
        log.info("trained accent classifier -> %s (%d samples)", out_path, len(y))

    def load(self, path: str) -> None:
        import joblib
        self._clf = joblib.load(path)

    def predict(self, wav: np.ndarray) -> str:
        if self._clf is None:
            raise RuntimeError("classifier not loaded; call load() or train() first")
        return str(self._clf.predict(self._embed(wav).reshape(1, -1))[0])
```

> **EXECUTION NOTE:** add `scikit-learn` and `joblib` to `pyproject.toml` deps (used only by accent_classifier).

- [ ] **Step 3: Commit**

```bash
git add backend/evaluation/__init__.py backend/evaluation/metrics.py backend/evaluation/accent_classifier.py
git commit -m "feat: evaluation metrics + XVector accent classifier"
```

---

### Task 15: eval_pipeline.py

**Files:**
- Create: `backend/evaluation/eval_pipeline.py`

- [ ] **Step 1: Implement** — batch eval over an L2-ARCTIC test split dir, run full pipeline per file, emit metrics table vs targets.

```python
"""Batch evaluation over a test split. Runs the full pipeline and reports metrics vs targets."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import click
import numpy as np
import soundfile as sf
import yaml

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor, load_audio
from pipeline.feature_extractor import FeatureExtractor
from pipeline.converter import SeedVCBackend, VevoBackend
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor
from evaluation import metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_pipeline")
ROOT = Path(__file__).resolve().parents[1]

TARGETS = {"accent_acc": 0.70, "emotion_sim": 0.85, "wer_rel_increase": 0.10,
           "utmos": 3.8, "speaker_cos": 0.50}


@click.command()
@click.option("--test-dir", required=True, help="dir of source WAVs")
@click.option("--target-accent", required=True)
@click.option("--out-json", default="eval_results.json")
@click.option("--config", "config_path", default=str(ROOT / "configs" / "pipeline_config.yaml"))
def main(test_dir, target_accent, out_json, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    mm = ModelManager.get(config_path)
    pre, fe = Preprocessor(cfg), FeatureExtractor(mm)
    qs, ec, post = QualitySelector(cfg, fe, mm), EmotionCorrector(cfg), Postprocessor(cfg)
    backends = [SeedVCBackend(cfg, mm.device)]
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))
    ref_dir = ROOT / cfg["paths"]["references"] / target_accent
    ref = str(sorted(ref_dir.glob("*.wav"))[0])

    rows = []
    for wav_path in sorted(Path(test_dir).glob("*.wav")):
        audio = load_audio(str(wav_path), sr=cfg["audio"]["sample_rate"])
        segs = pre.segment(audio)
        outs, emo_sims, src_wers = [], [], []
        src_text_full, out_text_full = [], []
        src_emotion0 = None
        for seg in segs:
            text, _ = fe.transcribe(seg)
            src_emotion = fe.emotion(seg)
            src_emotion0 = src_emotion0 or src_emotion
            prosody = fe.prosody(seg, n_words=len(text.split()))
            cands = [b.convert(seg, ref) for b in backends]
            cand, score, meta = qs.select(cands, seg, src_emotion, text)
            corrected = ec.correct(cand.wav, cand.sr, src_emotion, meta["emotion_output"], prosody)
            outs.append(corrected)
            emo_sims.append(meta["emotion_sim"])
            src_text_full.append(text); out_text_full.append(meta["transcript"])
        final = post.loudness_normalize(post.assemble(outs))
        wer = M.word_error_rate(" ".join(src_text_full), " ".join(out_text_full))
        utmos = M.utmos(mm, final, cfg["audio"]["sample_rate"])
        spk = M.speaker_agnosticism(mm, audio, final, cfg["audio"]["sample_rate"])
        rows.append({"file": wav_path.name, "emotion_sim": float(np.mean(emo_sims)),
                     "wer": wer, "utmos": utmos, "speaker_cos": spk})
        log.info("%s: emo=%.3f wer=%.3f utmos=%.2f spk=%.3f",
                 wav_path.name, np.mean(emo_sims), wer, utmos, spk)

    summary = {
        "n": len(rows),
        "emotion_sim_mean": float(np.mean([r["emotion_sim"] for r in rows])) if rows else 0.0,
        "wer_mean": float(np.mean([r["wer"] for r in rows])) if rows else 1.0,
        "utmos_mean": float(np.mean([r["utmos"] for r in rows])) if rows else 0.0,
        "speaker_cos_mean": float(np.mean([r["speaker_cos"] for r in rows])) if rows else 1.0,
        "targets": TARGETS, "rows": rows,
    }
    Path(out_json).write_text(json.dumps(summary, indent=2))
    click.echo(json.dumps({k: summary[k] for k in
               ("n","emotion_sim_mean","wer_mean","utmos_mean","speaker_cos_mean")}, indent=2))


if __name__ == "__main__":
    main()
```

> **EXECUTION NOTE:** Accent accuracy needs the trained classifier (Task 14) + labelled data; it is reported separately via `accent_classifier.predict` once a model is trained. eval_pipeline covers emotion/WER/UTMOS/speaker; add accent column once classifier checkpoint exists.

- [ ] **Step 2: Smoke check**: `cd backend && uv run python -c "import evaluation.eval_pipeline; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/evaluation/eval_pipeline.py
git commit -m "feat: batch evaluation pipeline over test split"
```

---

## Phase 7 — Finetune scripts (A100 — never run here) + test scaffold

### Task 16: prepare_l2arctic.py + finetune_style.py

**Files:**
- Create: `backend/scripts/prepare_l2arctic.py`
- Create: `backend/scripts/finetune_style.py`
- Modify: `backend/SETUP.md` (finetune section)

- [ ] **Step 1: prepare_l2arctic.py** — build native↔non-native parallel pairs (1132 CMU-ARCTIC sentences) + eval split. Reads raw L2-ARCTIC + CMU-ARCTIC dirs, writes a manifest jsonl.

```python
"""Prepare L2-ARCTIC parallel pairs for style-encoder finetune. Run before finetune (A100 box)."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import click

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prepare_l2arctic")


@click.command()
@click.option("--l2arctic-dir", required=True, help="root of L2-ARCTIC corpus")
@click.option("--cmu-dir", required=True, help="root of CMU-ARCTIC (native) corpus")
@click.option("--out", default="data/l2arctic_pairs/manifest.jsonl")
@click.option("--eval-frac", default=0.05, type=float)
def main(l2arctic_dir, cmu_dir, out, eval_frac):
    """Pair files sharing an arctic_<id> utterance id. Emits {native, nonnative, accent, split}."""
    l2 = Path(l2arctic_dir); cmu = Path(cmu_dir)
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    # CMU native wavs keyed by utterance id (filename stem, e.g. arctic_a0001)
    native = {p.stem: p for p in cmu.rglob("*.wav")}
    pairs = []
    for spk_dir in sorted(p for p in l2.iterdir() if p.is_dir()):
        accent = spk_dir.name  # L2-ARCTIC dir names encode speaker; map to accent via README
        for wav in (spk_dir / "wav").glob("*.wav") if (spk_dir / "wav").exists() else spk_dir.glob("*.wav"):
            nat = native.get(wav.stem)
            if nat is not None:
                pairs.append({"native": str(nat), "nonnative": str(wav), "accent": accent})
    n_eval = int(len(pairs) * eval_frac)
    for i, p in enumerate(pairs):
        p["split"] = "eval" if i < n_eval else "train"
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    log.info("wrote %d pairs (%d eval) -> %s", len(pairs), n_eval, out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: finetune_style.py** — Seed-VC style-encoder finetune skeleton. Header: RUN ON A100. Reads finetune_config.yaml + manifest; freezes content encoder + vocoder; trains style encoder.

```python
"""Finetune Seed-VC V2 style encoder on L2-ARCTIC parallel pairs.

RUN ON A100 — DO NOT EXECUTE LOCALLY (8GB card cannot hold training state).
Content encoder + vocoder are frozen. Only the ~50M-param style encoder trains.

Usage (A100 box):
    python scripts/finetune_style.py --config configs/finetune_config.yaml \
        --manifest data/l2arctic_pairs/manifest.jsonl

Output checkpoint -> checkpoints/seedvc_finetuned/ ; then update
pipeline_config.yaml seed_vc to point at it.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import click
import yaml

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("finetune_style")
ROOT = Path(__file__).resolve().parents[1]


@click.command()
@click.option("--config", "config_path", default=str(ROOT / "configs" / "finetune_config.yaml"))
@click.option("--manifest", required=True)
def main(config_path, manifest):
    cfg = yaml.safe_load(Path(config_path).read_text())
    log.info("Finetune config: %s", cfg)

    # 1. Import Seed-VC training modules from the cloned repo.
    #    EXECUTION (on A100): add third_party/seed-vc to sys.path and import its
    #    style-encoder + trainer. Exact module paths come from the cloned repo.
    seed_vc = ROOT / "third_party" / "seed-vc"
    sys.path.insert(0, str(seed_vc))
    log.warning("Wire Seed-VC trainer imports here per the cloned repo's train script.")

    # 2. Build dataset from manifest parallel pairs (native target, nonnative source).
    # 3. Freeze content encoder + vocoder; unfreeze style encoder.
    # 4. Optimizer: AdamW, lr per cfg (style 1e-4, AR 1e-5), wd 0.01.
    # 5. cosine-with-warmup (warmup 2000, total 50000), bf16, grad-accum 2, grad-checkpointing.
    # 6. Save to cfg["output"]["checkpoint_dir"] every N steps.
    raise SystemExit(
        "finetune_style.py is a guarded skeleton. Wire Seed-VC trainer imports "
        "(step 1) on the A100 box before running. See SETUP.md finetune section."
    )


if __name__ == "__main__":
    main()
```

> **EXECUTION NOTE:** This skeleton intentionally raises rather than running a half-wired trainer. The exact Seed-VC training-module imports are only knowable from the cloned repo on the A100 box. SETUP.md documents the wiring + run command. This honors "code + instructions ready, user trains."

- [ ] **Step 3: SETUP.md finetune section** — exact A100 commands: prepare → run finetune → copy checkpoint → update config.

- [ ] **Step 4: Smoke check**: `cd backend && uv run python -c "import scripts.prepare_l2arctic; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/prepare_l2arctic.py backend/scripts/finetune_style.py backend/SETUP.md
git commit -m "feat: L2-ARCTIC prep + guarded style-encoder finetune script (A100)"
```

---

### Task 17: tests scaffold (full tests = later phase)

**Files:**
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/fixtures/.gitkeep`

- [ ] **Step 1: conftest.py** — synthetic-audio fixtures + a skip marker for weight-dependent tests. (Actual test cases written in the later test phase.)

```python
"""Shared pytest fixtures. Full test cases land in the dedicated test phase."""
import numpy as np
import pytest


@pytest.fixture
def sine_wav():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


@pytest.fixture
def silence_wav():
    sr = 16000
    return np.zeros(sr, dtype=np.float32), sr


def weights_present() -> bool:
    from pathlib import Path
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints"
    return ckpt.exists() and any(ckpt.iterdir())


requires_weights = pytest.mark.skipif(not weights_present(),
                                      reason="model weights not downloaded")
```

- [ ] **Step 2: Smoke check**: `cd backend && uv run --extra dev pytest tests/ -q`
Expected: `no tests ran` (scaffold only) — exit cleanly.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/
git commit -m "chore: test scaffold + synthetic-audio fixtures (tests deferred)"
```

---

## Self-Review (completed)

**Spec coverage:** preprocessor✓(T6) feature_extractor✓(T7) converter Seed-VC✓(T8)+Vevo✓(T9) quality_selector✓(T10) emotion_corrector✓(T11) postprocessor✓(T12) model_manager✓(T5) types✓(T3) download_models✓(T4) configs✓(T2) eval metrics/classifier/pipeline✓(T14,T15) finetune scripts✓(T16) uv env✓(T1) CLI driver✓(T13) tests scaffold✓(T17). All spec §3 layout files covered except `api/` and edge (explicitly out of scope).

**Placeholder scan:** No TBD/TODO in code steps. Three EXECUTION NOTES mark genuine external-repo unknowns (Seed-VC CLI args, Amphion module path, audeering SER ordering) — these are real dependencies resolved by reading cloned repos, not lazy placeholders. finetune_style raises by design per user instruction.

**Type consistency:** `EmotionVec.to_array()`, `Segment`, `Candidate`, `ProsodyFeatures` used consistently across feature_extractor, quality_selector, emotion_corrector, run_pipeline, eval_pipeline. `ModelManager.get()` / `.load_utmos()` / `.load_ecapa()` / `.device` / `.vevo_device` consistent. Quality formula identical in spec, config, and quality_selector.

**Deps added during plan:** `scikit-learn`, `joblib` (accent_classifier) — add to pyproject in T1 or T14.
