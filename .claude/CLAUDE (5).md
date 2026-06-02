# DP6: Accent Shift — Claude Code Instructions

## Reference Docs (Read These First)

- `docs/backend-spec.md` — full backend pipeline, API endpoints, constraints, fine-tune details
- `docs/frontend-spec.md` — Next.js app spec, components, design tokens, API proxy routes
- `docs/DP6_AccentShift_Architecture.md` — architecture decisions, model rationale, evaluation metrics

Do not make decisions that contradict these docs. If something is unspecified, ask before implementing.

## Project Overview

Two-part system:
1. **Backend** (`backend/`) — FastAPI pipeline: audio in → accent-converted audio + metrics out
2. **Frontend** (`frontend/`) — Next.js 14 web app: upload audio, select accent, play converted audio

Full repo structure defined in `docs/backend-spec.md` (backend) and `docs/frontend-spec.md` (frontend).

## Core Models (Backend)

- Seed-VC V2 — primary accent+style converter (`--convert-style true`)
- Vevo-Voice (Amphion) — ensemble second leg
- Whisper large-v3 — ASR + word timestamps
- wav2vec2-large (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) — emotion SER
- BigVGAN-v2 (`nvidia/bigvgan-v2-24khz-100band-256x`) — vocoder
- pyworld — prosody extraction + F0 correction
- silero-vad — speech segmentation

## Critical Constraints (Never Violate)

- All audio at 16kHz mono internally. Resample at input boundary only.
- pyworld requires float64. Cast before every pyworld call: `audio.astype(np.float64)`
- Seed-VC max segment: 30s. VAD must not produce longer segments.
- F0 correction scale factor: clipped to [0.8, 1.2]. Hard cap.
- Emotion correction threshold: 0.85 cosine similarity. Do not change.
- Models loaded once at startup in `model_manager.py`. Never inside request handlers.
- VRAM: Whisper(3GB) + Seed-VC(2.5GB) + SER(1.2GB) + BigVGAN(0.5GB) = 7.2GB on RTX 4060. Vevo on CPU if needed. On A100/droplet: all on GPU.
- Quality score formula: `0.4 * emotion_sim + 0.4 * (1 - WER) + 0.2 * (UTMOS / 5.0)`

## Fine-tuning (NOT Claude Code's job)

Fine-tuning runs on A100. Scripts are in `backend/scripts/finetune_style.py` and config in `backend/configs/finetune_config.yaml`. Claude Code writes these scripts but does NOT run them. All hyperparameters are specified in `docs/backend-spec.md` — do not change them.

## Code Style

- Python: type hints everywhere, module-level logger (no prints), specific exceptions, config from yaml never hardcoded
- TypeScript: strict mode, no `any`, Zod for all API response validation
- Components: one component per file, named exports
- No inline styles — Tailwind classes only

## What to Ask Claude Code

**Backend:**
- "Implement `preprocessor.py` — VAD segmentation using silero-vad per backend-spec.md"
- "Implement `converter.py` — Seed-VC V2 and Vevo inference wrappers"
- "Implement `quality_selector.py` — ensemble scoring using the formula in CLAUDE.md"
- "Implement `emotion_corrector.py` — F0/energy correction via pyworld"
- "Implement `model_manager.py` — singleton loader with VRAM management"
- "Implement `routes.py` — /convert endpoint per backend-spec.md schemas"
- "Write `scripts/download_models.py` — pull all pretrained weights"
- "Write `scripts/finetune_style.py` — style encoder fine-tune per finetune_config.yaml"
- "Write tests for [module] with mocked models and synthetic audio"

**Frontend:**
- "Scaffold the Next.js 14 app with TypeScript, Tailwind, shadcn/ui per frontend-spec.md"
- "Implement the `AudioUploader` component per frontend-spec.md"
- "Implement the `WaveformPlayer` component using WaveSurfer.js"
- "Implement the `EmotionMetrics` component — source vs output bars"
- "Implement the `MetricsRow` component with colour-coded stat cards"
- "Implement `app/api/convert/route.ts` — proxy to FastAPI backend"
- "Wire up the full page layout per the diagram in frontend-spec.md"

## Do NOT Ask Claude Code To

- Choose models (decided — see architecture doc)
- Change any threshold or formula (all specified above)
- Pick reference audio clips (manual task)
- Run fine-tuning (A100 task, you run it)
- Make design decisions not in frontend-spec.md
