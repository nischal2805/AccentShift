# DP6: Accent Shift — Backend Specification (FastAPI)

## Stack

- Python 3.10
- FastAPI + uvicorn
- PyTorch 2.1+ / CUDA 12.1
- torchaudio, librosa, soundfile, pyloudnorm
- pyworld (prosody), silero-vad
- transformers (Whisper large-v3, wav2vec2-large SER)
- Seed-VC V2: github.com/Plachtaa/seed-vc
- Vevo (Amphion): github.com/open-mmlab/Amphion
- BigVGAN-v2: nvidia/bigvgan-v2-24khz-100band-256x

## Repo Structure

```
backend/
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py       # VAD, normalisation, segmentation (silero-vad)
│   ├── feature_extractor.py  # Whisper ASR, wav2vec2 SER, pyworld prosody
│   ├── converter.py          # Seed-VC V2 + Vevo wrappers
│   ├── quality_selector.py   # ensemble scoring: emotion_sim + WER + UTMOS
│   ├── emotion_corrector.py  # F0/energy correction via pyworld
│   ├── postprocessor.py      # overlap-add, loudness norm (-23 LUFS)
│   └── model_manager.py      # singleton loader, VRAM management
├── api/
│   ├── main.py               # FastAPI app, startup model loading
│   ├── routes.py             # /convert, /accents, /evaluate, /health
│   └── schemas.py            # Pydantic request/response models
├── evaluation/
│   ├── eval_pipeline.py      # batch eval on L2-ARCTIC test split
│   ├── metrics.py            # WER, emotion_sim, UTMOS, accent_acc
│   └── accent_classifier.py  # XVector accent accuracy
├── references/               # reference clips per accent (5-10s WAVs)
│   ├── indian_english/
│   ├── chinese_english/
│   ├── japanese_english/
│   ├── british_english/
│   └── american_english/
├── scripts/
│   ├── download_models.py    # pull all pretrained weights
│   ├── prepare_l2arctic.py   # dataset prep for fine-tuning
│   └── finetune_style.py     # style encoder fine-tune (run on A100)
├── configs/
│   ├── pipeline_config.yaml  # thresholds, paths, model names
│   └── finetune_config.yaml  # fine-tune hyperparameters
├── tests/
│   ├── test_preprocessor.py
│   ├── test_converter.py
│   └── test_emotion_correction.py
└── requirements.txt
```

## API Endpoints

### `POST /convert`

```
Request:  multipart/form-data
  - audio: File (.wav/.mp3/.flac/.m4a, max 50MB)
  - target_accent: str (indian_english | chinese_english | japanese_english | british_english | american_english)

Response: application/json
  - audio_b64: str          # base64-encoded 16kHz WAV
  - metrics: {
      emotion_similarity: float   # cosine sim, 0-1
      wer: float                  # word error rate, 0-1
      mos_estimate: float         # UTMOS score, 1-5
      valence_source: float
      valence_output: float
      arousal_source: float
      arousal_output: float
      dominance_source: float
      dominance_output: float
    }
  - processing_time_ms: int
```

### `GET /accents`
Returns `{ accents: [{ key: str, label: str }] }`

### `POST /evaluate`
Batch evaluation endpoint for L2-ARCTIC. Internal use only.

### `GET /health`
Returns model load status. Frontend polls this on startup to know when backend is ready.

## Pipeline Flow (per request)

```
1. Receive audio bytes → decode → resample to 16kHz mono
2. Silero VAD → split into speech segments
3. Per segment (parallelised where possible):
   a. Whisper large-v3 → transcript + word timestamps
   b. wav2vec2-large SER → [valence, arousal, dominance]
   c. pyworld HARVEST+STONEMASK → F0 contour, energy, duration
   d. Seed-VC V2 (--convert-style true) → candidate_1
   e. Vevo-Voice → candidate_2
   f. Quality score both → pick winner
   g. Emotion correction if cosine_sim < 0.85
4. Overlap-add segments → full audio
5. pyloudnorm to -23 LUFS
6. Base64 encode → return with metrics
```

## Critical Constraints

- All audio at 16kHz mono internally. Resample at input boundary.
- pyworld requires float64. Cast before every pyworld call.
- Seed-VC max segment: 30s. VAD must not produce longer segments.
- F0 correction capped at ±20% (scale factor clipped to [0.8, 1.2]).
- Emotion correction threshold: cosine_sim < 0.85 triggers correction.
- All models loaded once at startup in model_manager.py. Never inside handlers.
- VRAM budget (RTX 4060 8GB): Whisper(3GB) + Seed-VC(2.5GB) + SER(1.2GB) + BigVGAN(0.5GB) = 7.2GB.
  Vevo loads on CPU if 8GB card. On A100/droplet: load everything on GPU.

## Quality Scoring Formula

```python
score = 0.4 * emotion_cosine_sim + 0.4 * (1 - WER) + 0.2 * (UTMOS / 5.0)
```

Pick whichever of Seed-VC or Vevo scores higher per segment.

## Fine-tuning (A100 — do not run locally)

See `scripts/finetune_style.py` and `configs/finetune_config.yaml`.

What gets fine-tuned: Seed-VC V2 style encoder only (~50M params). Content encoder and vocoder frozen.
Dataset: L2-ARCTIC parallel pairs (same 1132 CMU-ARCTIC sentences, native vs non-native speakers).
Duration: ~2-3 days on A100 for 50k steps.

Hyperparameters (in finetune_config.yaml):
- lr: 1e-4 (style encoder), 1e-5 (AR decoder)
- batch_size: 16 (A100 can handle this)
- gradient_accumulation: 2 (effective batch 32)
- warmup_steps: 2000
- total_steps: 50000
- optimizer: AdamW, weight_decay: 0.01
- scheduler: cosine with warmup
- mixed_precision: bf16

After fine-tuning: copy checkpoint to `checkpoints/seedvc_finetuned/` and update path in pipeline_config.yaml.
