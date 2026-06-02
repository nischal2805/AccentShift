# DP6 AccentShift — Pipeline Core + Evaluation Design

**Date:** 2026-06-02
**Status:** Approved (pending final user review)
**Scope:** Architecture A pipeline core + evaluation suite. Finetune = scripts only (run on A100 by user). API server + web frontend + edge (Arch B) = later scope.

---

## 1. Goal

End-to-end speech pipeline: English audio in → same speech in a target English accent out, preserving emotion / prosody / intensity, speaker-agnostic. Reference: `docs/DP6_AccentShift_Architecture.md` (Architecture A), `docs/backend-spec.md`.

This phase builds and codes out the **full** pipeline + evaluation. No incremental testing — code everything, then a dedicated test phase later.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Local HW | NVIDIA GPU 8GB+. Arch A real models, Vevo → CPU when tight. |
| Subsystems this phase | Pipeline core (Arch A) + Evaluation suite. |
| VC backends | Real Seed-VC V2 + Vevo, integrated now. No mock backend. |
| Python env | **uv** (pyproject.toml + uv.lock). Python 3.10. No conda. |
| Build style | Build-first: code everything fully, tests deferred to a later phase. |
| Finetune | Code + run instructions only. User trains on A100. Never executed here. |
| API / frontend / edge | Out of scope. Later (GPU droplet deploy). |

Constants are fixed by `docs/backend-spec.md` / project CLAUDE.md and MUST NOT change:
- 16kHz mono internal; resample only at input boundary.
- pyworld requires float64 — cast before every pyworld call.
- Seed-VC max segment 30s; VAD must not exceed.
- F0 correction scale clipped to [0.8, 1.2]; energy scale clipped to [0.7, 1.3].
- Emotion correction triggers when cosine_sim < 0.85.
- Quality score = `0.4*emotion_sim + 0.4*(1 - WER) + 0.2*(UTMOS/5)`.
- Models loaded once at startup in `model_manager.py`, never in handlers.

## 3. Repo Layout (this phase)

```
backend/
├── pyproject.toml            # uv-managed deps, Python 3.10
├── uv.lock
├── .python-version
├── pipeline/
│   ├── __init__.py
│   ├── types.py              # shared dataclasses: Segment, EmotionVec, ProsodyFeatures, Candidate, SegmentResult
│   ├── preprocessor.py       # load/resample/normalise, silero VAD, ≤30s segmentation
│   ├── feature_extractor.py  # Whisper large-v3 ASR+timestamps, wav2vec2 SER, pyworld prosody
│   ├── converter.py          # ConverterBackend ABC + SeedVCBackend + VevoBackend
│   ├── quality_selector.py   # score candidates, pick winner
│   ├── emotion_corrector.py  # pyworld F0 rescale + energy warp
│   ├── postprocessor.py      # overlap-add crossfade, pyloudnorm -23 LUFS
│   └── model_manager.py      # singleton loader, VRAM-aware device placement
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # WER, emotion_sim, UTMOS, speaker-agnosticism
│   ├── accent_classifier.py  # XVector accent classifier: train + infer
│   └── eval_pipeline.py      # batch eval over L2-ARCTIC test split
├── third_party/              # git clones (gitignored): seed-vc/, Amphion/
├── references/               # <accent>/*.wav reference clips (manual)
│   ├── indian_english/  chinese_english/  japanese_english/
│   ├── british_english/ american_english/
├── checkpoints/              # downloaded weights (gitignored)
├── scripts/
│   ├── download_models.py    # pull HF weights + clone VC repos + their checkpoints
│   ├── prepare_l2arctic.py   # parallel-pair prep for finetune + eval split
│   └── finetune_style.py     # style-encoder finetune (RUN ON A100 — do not execute here)
├── configs/
│   ├── pipeline_config.yaml  # paths, thresholds, model names, device map
│   └── finetune_config.yaml  # finetune hyperparameters (per spec)
├── tests/                    # later phase — directory + conftest scaffold only
├── run_pipeline.py           # CLI test driver: wav in → converted wav + metrics json
├── SETUP.md                  # uv env + downloads + Windows fallback instructions
└── .gitignore
```

## 4. Module Contracts

Shared types in `pipeline/types.py` (frozen dataclasses):
- `Segment(audio: np.ndarray, start_s: float, end_s: float, sr: int)`
- `EmotionVec(valence: float, arousal: float, dominance: float)` + `.to_array()`
- `ProsodyFeatures(f0: np.ndarray, timeaxis: np.ndarray, energy: np.ndarray, speaking_rate: float)`
- `Candidate(name: str, wav: np.ndarray, sr: int)`
- `SegmentResult(wav, emotion_source, emotion_output, transcript, wer, mos, chosen_backend)`

### preprocessor.py
- `load_audio(path) -> (np.ndarray float32, sr=16000, mono)`; resample at boundary; `librosa.util.normalize`.
- `segment(audio, sr) -> List[Segment]` via silero VAD (`threshold=0.5`); enforce ≤30s hard cap by force-splitting long voiced runs.

### feature_extractor.py
Takes `model_manager` handles. Per segment:
- `transcribe(seg) -> (text, word_timestamps)` — Whisper large-v3.
- `emotion(seg) -> EmotionVec` — wav2vec2-large SER (dimensional).
- `prosody(seg) -> ProsodyFeatures` — pyworld HARVEST+STONEMASK (float64), `librosa.feature.rms` energy, speaking rate from Whisper word count / duration.

### converter.py
- `ConverterBackend(ABC)`: `convert(source: Segment, ref_wav, sr) -> Candidate`.
- `SeedVCBackend`: wraps `third_party/seed-vc` inference (`convert_style=True`, `intelligibility_cfg_rate=0.7`, `similarity_cfg_rate=0.7`, `diffusion_steps=30`). Output candidate wav (vocoded via Seed-VC's own BigVGAN, or our BigVGAN where mel exposed — follow repo's native path).
- `VevoBackend`: wraps `third_party/Amphion` `infer_vevovoice`. Loadable on CPU (config `vevo_device`).
- Reference clips resolved from `references/<accent>/` (first/curated clip per config).

### quality_selector.py
- `score(candidate, source_seg, source_emotion, feature_extractor) -> float` using locked formula.
- `select(candidates, ...) -> (best_candidate, score, name)`.

### emotion_corrector.py
- `correct(wav, source_emotion, output_emotion, source_prosody, sr) -> np.ndarray`.
- No-op if cosine_sim ≥ 0.85. Else pyworld F0 rescale toward source (clip [0.8,1.2]) → cheaptrick/d4c/synthesize; energy warp (clip [0.7,1.3]). float64 casts.

### postprocessor.py
- `assemble(segment_wavs, sr, crossfade_ms=30) -> np.ndarray` overlap-add.
- `loudness_normalize(audio, sr, target_lufs=-23.0)` via pyloudnorm.

### model_manager.py
- Singleton. `get() -> ModelManager`. Loads at init from `pipeline_config.yaml`: Whisper large-v3, SER, BigVGAN-v2, Seed-VC, Vevo. VRAM-aware: Vevo → CPU when GPU < threshold (config). Exposes typed handles. Never loads inside request/segment loops.

### Orchestrator (in run_pipeline.py)
Wires the flow: decode → VAD → per-seg [ASR/SER/prosody → SeedVC+Vevo → score → correct] → overlap-add → loudnorm → write wav + metrics json. Matches `backend-spec.md` §Pipeline Flow exactly.

## 5. Evaluation Suite

- `metrics.py`:
  - `wer(ref_text, hyp_text)` via jiwer; transcripts from Whisper.
  - `emotion_similarity(src_vec, out_vec)` cosine.
  - `utmos(wav, sr)` via torch.hub `sarulab-speech/UTMOS22` (UTMOS-strong).
  - `speaker_agnosticism(src_wav, out_wav)` ECAPA-TDNN (speechbrain) cosine; target < 0.5.
- `accent_classifier.py`: XVector (speechbrain) accent classifier — `train(dataset)` + `predict(wav) -> accent`. Training needs VCTK/GLOBE labels (user fetches); code provided, training gated on data presence.
- `eval_pipeline.py`: batch over L2-ARCTIC test split, run full pipeline, emit metrics table vs targets (accent>70%, emotion>0.85, WER<10% rel, UTMOS>3.8, speaker cos<0.5).

## 6. Environment & Downloads (uv)

- `pyproject.toml`: Python 3.10; torch 2.1 / CUDA 12.1, torchaudio, transformers, librosa, soundfile, pyloudnorm, pyworld, silero-vad, jiwer, speechbrain, pyyaml, numpy, scipy, click (CLI). Pinned; `uv.lock` committed.
- `SETUP.md`: `uv venv` → `uv sync`; CUDA torch index URL; Windows fallbacks for Seed-VC / Amphion dep failures (skip flash-attn/deepspeed/triton, CPU paths, manual checkpoint download links).
- `download_models.py`: idempotent, logged. HF snapshot for Whisper / SER / BigVGAN into `checkpoints/`. Git-clone Seed-VC + Amphion into `third_party/` and pull their checkpoints per each repo's instructions.

## 7. Finetune (scripts only — never run here)

- `finetune_style.py`: finetune Seed-VC style encoder (~50M params) on L2-ARCTIC parallel pairs; content encoder + vocoder frozen. Header docstring: "RUN ON A100. Do not execute locally." Reads `finetune_config.yaml`.
- `finetune_config.yaml`: lr 1e-4 (style) / 1e-5 (AR), batch 16, grad-accum 2, warmup 2000, total 50000, AdamW wd 0.01, cosine schedule, bf16.
- `prepare_l2arctic.py`: build native↔non-native parallel pairs (1132 CMU-ARCTIC sentences) + eval split.
- `SETUP.md` section: exact A100 run command + checkpoint output path → update `pipeline_config.yaml`.

## 8. Build Order

1. `pyproject.toml`, `.python-version`, `.gitignore`, `SETUP.md` skeleton.
2. `configs/pipeline_config.yaml`, `configs/finetune_config.yaml`.
3. `pipeline/types.py`.
4. `scripts/download_models.py`.
5. `pipeline/model_manager.py`.
6. `pipeline/preprocessor.py`.
7. `pipeline/feature_extractor.py`.
8. `pipeline/converter.py` (Seed-VC backend, then Vevo backend).
9. `pipeline/quality_selector.py`.
10. `pipeline/emotion_corrector.py`.
11. `pipeline/postprocessor.py`.
12. `run_pipeline.py` orchestrator CLI.
13. `evaluation/metrics.py`, `accent_classifier.py`, `eval_pipeline.py`.
14. `scripts/prepare_l2arctic.py`, `scripts/finetune_style.py`.
15. `tests/` scaffold (conftest + fixtures only; full tests = later phase).

No per-step test gating. Tests are a dedicated later phase.

## 9. Risks

- **Windows dep hell** (Seed-VC / Amphion: deepspeed, triton, flash-attn). Isolated to `converter.py` backends — rest of pipeline unaffected. `SETUP.md` documents fallbacks. If a repo can't install on the 8GB box, that backend's wrapper raises a clear error; pipeline can run single-backend.
- **8GB VRAM tight** (~7.2GB Arch A). Vevo → CPU. Config-driven device map; option to disable Vevo leg.
- **Eval datasets** (L2-ARCTIC, VCTK/GLOBE) are large external downloads — user fetches; eval/accent-train code gated on presence.
- **UTMOS / speechbrain / silero** pull weights at runtime — `download_models.py` pre-fetches to avoid surprise downloads.
```
