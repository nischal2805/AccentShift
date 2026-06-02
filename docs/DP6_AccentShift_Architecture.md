# DP6: Accent Shift — System Architecture & Implementation Guide

## 1. Problem Framing

The task is an **end-to-end speech processing pipeline** that takes English audio as input and outputs the same speech in a target English accent (Chinese-accented, Japanese-accented, Indian English, etc.) while:

- Preserving the original speaker's **emotion** (valence, arousal, expressiveness)
- Preserving **prosody** (pitch dynamics, speaking rate, energy contour)
- Preserving **voice intensity**
- Being **speaker-agnostic** on output (no need to clone original timbre)
- Supporting multiple target accents via a single system

The speaker-agnostic constraint is actually a relaxation — you don't need to preserve the original timbre, only the *emotional and prosodic content*. This makes the problem cleaner than full voice conversion.

---

## 2. Core Technical Challenge: Disentanglement

Speech carries several entangled attributes simultaneously:

| Attribute | What it encodes | Must preserve? |
|---|---|---|
| **Linguistic content** | Phonemes, words, sentences | Yes — highest priority |
| **Emotion** | Valence, arousal, dominance; expressiveness | Yes — explicit requirement |
| **Prosody** | F0 contour, energy, duration, speaking rate | Yes — carries emotion |
| **Accent** | Phonetic patterns, vowel realisation, stress timing | No — this is what we *change* |
| **Speaker identity** | Timbre, voice texture, resonance | No — speaker-agnostic output |

A successful pipeline must extract content + emotion + prosody, discard accent + speaker identity, then synthesise using a target accent reference. This is a 5-way disentanglement problem.

---

## 3. Architecture A — Maximum Quality (GPU Server / RTX 4060)

This architecture uses the absolute best open-weight models for each stage with no concession to speed or memory. Target hardware: RTX 4060 8GB or A-series cloud GPU droplet.

### 3.1 System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
│   Source WAV (any English accent, any emotion, any length)   │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────┼──────────────────┐
         ▼             ▼                  ▼
┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐
│ Whisper     │  │ wav2vec2-     │  │ pyworld / parselmouth │
│ large-v3    │  │ large SER     │  │                       │
│ (ASR)       │  │ (Emotion)     │  │ (Prosody Extraction)  │
│             │  │               │  │                       │
│ → transcript│  │ → valence     │  │ → F0 contour          │
│   + timing  │  │   arousal     │  │   energy envelope     │
│   + word    │  │   dominance   │  │   speaking rate       │
│   timestamps│  │   emotion cls │  │   duration per phone  │
└──────┬──────┘  └───────┬───────┘  └───────────┬───────────┘
       │                 │                       │
       └─────────────────┼───────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   SEED-VC V2  (Primary Converter)            │
│                                                              │
│  Architecture: AR Transformer + Diffusion Transformer (CFM) │
│                                                              │
│  Inputs:                                                     │
│    - source_wav (source audio)                               │
│    - target_reference_clip (5-10s of target accent speaker)  │
│    - --convert-style true  (enables accent+emotion AR path)  │
│    - --intelligibility-cfg-rate 0.7                          │
│    - --similarity-cfg-rate 0.7                               │
│                                                              │
│  Internally:                                                 │
│    Whisper encoder → content tokens                          │
│    AR transformer → style-converted acoustic tokens          │
│    CFM (Conditional Flow Matching) → mel spectrogram         │
│                                                              │
│  Output: converted_mel + converted_wav (draft)               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               VEVO-VOICE (Amphion) — Ensemble Leg            │
│                                                              │
│  Run in parallel with Seed-VC V2.                            │
│  Architecture: VQ-VAE content tokenizer (codebook=32)        │
│               + Flow Matching decoder                        │
│  Trained on: Emilia 101k hours (6 languages)                 │
│                                                              │
│  python -m models.vc.vevo.infer_vevovoice                    │
│    --src source.wav --ref target_accent_ref.wav              │
│                                                              │
│  Output: vevo_converted_wav (second candidate)               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               QUALITY SELECTION LAYER                        │
│                                                              │
│  Run SER on both candidates:                                 │
│    - Emotion cosine similarity vs source (higher = better)   │
│  Run Whisper WER on both candidates:                         │
│    - Compare transcript to source transcript (lower = better)│
│  Select best candidate per utterance.                        │
│                                                              │
│  This is the "innovation in pipeline architecture" angle.    │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               EMOTION CORRECTION LAYER                       │
│                                                              │
│  Check: cosine_sim(source_emotion, output_emotion) < 0.85    │
│  If drift detected:                                          │
│    1. F0 re-scaling: warp output pitch contour toward source │
│       using pyworld HARVEST + STONEMASK                      │
│    2. Energy warping: match RMS energy envelope per frame    │
│    3. Duration adjustment: time-stretch via librosa          │
│       phase vocoder to match speaking rate                   │
│                                                              │
│  This preserves the emotional "shape" even if the accent     │
│  conversion shifts prosody slightly.                         │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               BigVGAN-v2 VOCODER                             │
│                                                              │
│  Input: corrected mel spectrogram                            │
│  Output: final 24kHz WAV                                     │
│  Why BigVGAN over HiFi-GAN: better high-freq reconstruction, │
│  fewer artifacts on non-native speech patterns               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
                  ┌─────────────┐
                  │  Output WAV │
                  │             │
                  │  Target     │
                  │  accent ✓   │
                  │  Emotion ✓  │
                  │  Content ✓  │
                  │  Speaker-   │
                  │  agnostic ✓ │
                  └─────────────┘
```

### 3.2 Model Selection Rationale

**Whisper large-v3 (ASR + timing)**
The benchmark leader for English ASR with word-level timestamps. Used for two purposes: transcript generation (for WER evaluation) and phoneme timing extraction (for duration-aware prosody correction). OpenAI, open weights, MIT license.

**wav2vec2-large SER (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`)**
Fine-tuned on MSP-Podcast for dimensional emotion (valence/arousal/dominance), not categorical. Dimensional is better here because it captures subtle expressiveness gradients — a slightly angry utterance won't round to "angry" and get mis-handled. Produces a 3D emotion vector that can be compared pre/post conversion.

**Seed-VC V2 (Primary converter)**
The only open-source VC system with an explicit AR-based style+accent+emotion conversion path (`--convert-style true`). Uses a two-stage architecture: AR transformer for style token generation followed by a Conditional Flow Matching transformer for acoustic synthesis. Outperforms OpenVoice V2 and CosyVoice on speaker similarity and WER in ablations. Repo: `Plachtaa/seed-vc`.

**Vevo-Voice (Amphion, ensemble leg)**
ICLR 2025 paper from Meta/CMU. Uses VQ-VAE content tokenisation with compact codebook (size 32) to forcibly strip accent information, then flow-matching decoder conditioned on target style embedding. Trained on 101k hours Emilia. Produces qualitatively different outputs from Seed-VC, making ensemble selection meaningful. Repo: `open-mmlab/Amphion`.

**BigVGAN-v2 (Vocoder)**
Nvidia's universal vocoder trained on diverse speech domains. Produces fewer artefacts on non-native speech patterns and edge-case prosody than standard HiFi-GAN V1. Runs on GPU in milliseconds.

**Check first: `Berkeley-Speech-Group/StyleStream`**
Published Feb 2026, this is currently the highest-performing system — jointly converts timbre + accent + emotion, trained with text-supervised disentanglement (ASR loss on the bottleneck, guaranteeing linguistic content preservation). If pretrained weights are released by the time you build, replace Seed-VC with StyleStream as primary. The Destylizer (HuBERT-Large + conformers + FSQ) + Stylizer (DiT) architecture is cleanest for this problem.

---

## 4. Architecture B — Edge-Optimised (Raspberry Pi / Jetson / Laptop CPU)

Constraints: ≤4GB RAM, no discrete GPU, real-time latency target ≤500ms for 200ms audio chunks.

### 4.1 System Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
│   Streaming audio — 200ms chunks with 50ms overlap           │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐
│ Whisper      │  │ wav2vec2-    │  │ YAAPT (pitch)       │
│ base.en      │  │ base SER     │  │ + librosa RMS       │
│ (ONNX)       │  │ (ONNX, INT8) │  │ (lightweight F0)    │
│              │  │              │  │                     │
│ ~39M params  │  │ ~95M params  │  │ CPU-only, fast      │
│ ~80ms RTF    │  │ ~40ms        │  │ ~10ms               │
└──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘
       │                 │                      │
       └─────────────────┼──────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               CosyVoice2-0.5B (Core Converter)               │
│               Exported to ONNX, INT8 quantised               │
│                                                              │
│  Architecture: LLM-based TTS with voice conversion mode      │
│  0.5B parameters, runs on CPU with ONNX Runtime              │
│                                                              │
│  Input: transcript + source audio + target accent reference  │
│  Streaming inference: generates audio chunks in parallel     │
│  with input processing (bi-streaming mode)                   │
│                                                              │
│  Emotion control via instruct tokens:                        │
│  <|emotion|>happy<|/emotion|> injected per utterance         │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               Lightweight Prosody Correction                 │
│                                                              │
│  F0 scaling only (skip energy warping for speed)             │
│  librosa pitch_shift on output chunks                        │
│  Target: match source F0 mean and dynamic range              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               HiFi-GAN V1 (Vocoder, ONNX)                    │
│               ~20ms per 200ms chunk                          │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
              Streaming Output WAV
              (overlap-add synthesis)
```

### 4.2 Key Trade-offs vs Architecture A

| Component | Arch A | Arch B | Trade-off |
|---|---|---|---|
| ASR | Whisper large-v3 | Whisper base.en (ONNX) | ~4% WER increase |
| SER | wav2vec2-large (full) | wav2vec2-base (INT8 ONNX) | Emotion accuracy ~8% lower |
| Core VC | Seed-VC V2 + Vevo ensemble | CosyVoice2-0.5B (ONNX) | Single pass, no ensemble |
| Prosody | F0 + energy + duration | F0 only | Less emotion fidelity |
| Vocoder | BigVGAN-v2 | HiFi-GAN V1 (ONNX) | Slight artifact increase |
| Latency | ~2-4s per utterance | ~200-400ms per chunk | 10x faster |
| VRAM | ~6GB | ~0GB (CPU-only) | No GPU needed |

---

## 5. Data and Training Strategy

### 5.1 Do You Need to Train from Scratch?

**No.** Both Seed-VC V2 and Vevo are zero-shot — they generalise to unseen accents from a 5-10 second reference clip. You need:

1. A curated set of reference clips per target accent (collected, not trained)
2. Optionally: a fine-tune of the style encoder for better accent-specific representations

Fine-tuning is only worth doing if you want sharper accent accuracy on specific pairs (e.g., American → Indian English). For a competition demo, zero-shot inference is sufficient.

### 5.2 Datasets

#### Primary Training / Fine-tune Datasets

| Dataset | Content | Size | Accents | Access |
|---|---|---|---|---|
| **L2-ARCTIC** | Non-native English read speech | 24h | Hindi, Korean, Mandarin, Spanish, Arabic | Free, CMU |
| **VCTK** | Native English multi-speaker | 44h | 110 speakers, various UK/US accents | Free, Edinburgh |
| **LibriTTS-R** | Clean native English multi-speaker | 585h | American English | Free, HuggingFace |
| **GLOBE** | High-quality English, 164 accents | ~200h | 164 distinct English accents worldwide | Free, request |
| **MSP-Podcast** | Emotionally diverse real speech | 180h | American English with emotion labels | Free, request |
| **ESD** | Emotional speech, 10 speakers × 5 emotions | 29h | Chinese + English | Free, GitHub |

L2-ARCTIC is the most directly useful because it is **parallel** — the same 1,132 CMU-ARCTIC sentences are recorded by both native (CMU-ARCTIC) and non-native speakers. This lets you construct direct source-target pairs for fine-tuning the style encoder, which is a significant advantage over non-parallel approaches.

#### Reference Clips (Inference-time, no training needed)

For zero-shot inference you need 5-10 second reference clips of native speakers of each target accent. Sources:

- **Common Voice** (Mozilla) — tagged by accent, freely downloadable, large
- **VCTK** — includes British, Scottish, Irish, Canadian speakers
- **Forvo** — pronunciation database with native speaker recordings
- **YouTube** — manually curated native speaker clips (Chinese, Japanese, Indian English presenters)

Collect 10 reference clips per accent, run inference with each, select the most natural-sounding one per accent as your fixed reference.

### 5.3 Fine-tuning Strategy (If Required)

**What to fine-tune:** Only the style encoder of Seed-VC V2. The content encoder (Whisper-based) and vocoder are frozen. This is a ~50M parameter fine-tune, not a full model retrain.

**Dataset for fine-tune:** L2-ARCTIC + CMU-ARCTIC parallel pairs (same transcript, different accent). This provides explicit source-target training signal.

**Hardware:** RTX 4060 8GB. Batch size 4, gradient checkpointing enabled.

**Expected duration:** 2-3 days for 50k steps.

**Hyperparameters:**
```
learning_rate: 1e-4 (style encoder), 1e-5 (AR decoder)
batch_size: 4
gradient_accumulation: 8 (effective batch 32)
warmup_steps: 2000
total_steps: 50000
optimizer: AdamW, weight_decay=0.01
scheduler: cosine with warmup
```

**What this buys:** Better accent-specific style embeddings, particularly for Indian/Chinese/Japanese-accented English which are the most commonly requested target accents. Zero-shot already works; fine-tuning pushes accent accuracy from ~70% to ~85% (subjective evaluator agreement).

---

## 6. Full System Flow (Architecture A — Production)

### 6.1 Pre-processing

```python
# 1. Load and normalise audio
audio, sr = librosa.load(input_path, sr=16000, mono=True)
audio = librosa.util.normalize(audio)

# 2. Voice Activity Detection — split into utterances
# Use Silero VAD (lightweight, accurate)
segments = silero_vad.get_speech_timestamps(audio, model, threshold=0.5)

# 3. Each segment is processed independently
# Max segment length: 30 seconds (Seed-VC auto-chunks at 30s anyway)
```

### 6.2 Feature Extraction (per segment)

```python
# Emotion extraction
emotion_model = load_wav2vec2_ser()  # audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
source_emotion = emotion_model(segment)  # [valence, arousal, dominance]

# Prosody extraction
f0, timeaxis = pyworld.harvest(segment.astype(float), sr)
f0 = pyworld.stonemask(segment.astype(float), f0, timeaxis, sr)
energy = librosa.feature.rms(y=segment)
speaking_rate = estimate_speaking_rate(whisper_timestamps, len(segment)/sr)
```

### 6.3 Accent Conversion

```python
# Select reference clip for target accent
ref_clip = accent_reference_bank[target_accent]  # pre-curated 5-10s clips

# Seed-VC V2 conversion
seed_vc_output = seed_vc.inference(
    source=segment,
    reference=ref_clip,
    convert_style=True,
    intelligibility_cfg_rate=0.7,
    similarity_cfg_rate=0.7,
    diffusion_steps=30
)

# Vevo-Voice conversion (parallel)
vevo_output = vevo.infer_vevovoice(
    src=segment,
    ref=ref_clip
)

# Quality selection
seed_vc_score = quality_score(seed_vc_output, segment, source_emotion)
vevo_score = quality_score(vevo_output, segment, source_emotion)
best_output = seed_vc_output if seed_vc_score > vevo_score else vevo_output
```

### 6.4 Quality Scoring Function

```python
def quality_score(converted, source, source_emotion):
    # Emotion preservation (40% weight)
    output_emotion = emotion_model(converted)
    emotion_sim = cosine_similarity(source_emotion, output_emotion)
    
    # Intelligibility (40% weight)
    output_transcript = whisper.transcribe(converted)["text"]
    source_transcript = whisper.transcribe(source)["text"]
    wer = word_error_rate(source_transcript, output_transcript)
    intelligibility = 1.0 - min(wer, 1.0)
    
    # Naturalness proxy — UTMOS automated MOS (20% weight)
    mos = utmos_predictor(converted)
    
    return 0.4 * emotion_sim + 0.4 * intelligibility + 0.2 * (mos / 5.0)
```

### 6.5 Emotion Correction

```python
def emotion_correction(converted_wav, source_emotion, output_emotion, source_f0, sr):
    if cosine_similarity(source_emotion, output_emotion) >= 0.85:
        return converted_wav  # no correction needed
    
    # Extract converted F0
    conv_f0, timeaxis = pyworld.harvest(converted_wav.astype(float), sr)
    conv_f0 = pyworld.stonemask(converted_wav.astype(float), conv_f0, timeaxis, sr)
    
    # F0 shift: scale to match source mean and std
    voiced = conv_f0 > 0
    if voiced.sum() > 0:
        scale = source_f0[source_f0 > 0].mean() / conv_f0[conv_f0 > 0].mean()
        conv_f0[voiced] *= np.clip(scale, 0.8, 1.2)  # cap shift at ±20%
    
    # Re-synthesise with pyworld
    sp = pyworld.cheaptrick(converted_wav.astype(float), conv_f0, timeaxis, sr)
    ap = pyworld.d4c(converted_wav.astype(float), conv_f0, timeaxis, sr)
    corrected = pyworld.synthesize(conv_f0, sp, ap, sr)
    
    # Energy warping
    source_energy = librosa.feature.rms(y=source_wav)
    output_energy = librosa.feature.rms(y=corrected)
    energy_scale = (source_energy.mean() / output_energy.mean())
    corrected = corrected * np.clip(energy_scale, 0.7, 1.3)
    
    return corrected
```

### 6.6 Post-processing and Assembly

```python
# Assemble segments back into full audio
# Use overlap-add for seamless joins at segment boundaries
output_audio = overlap_add_synthesis(corrected_segments, crossfade_ms=30)

# Final normalisation
output_audio = pyloudnorm.normalize.loudness(output_audio, sr, target_lufs=-23.0)

# Export
soundfile.write(output_path, output_audio, sr, subtype='PCM_16')
```

---

## 7. Orchestration

### 7.1 API Layer (FastAPI)

```
POST /convert
  body: { audio_file: bytes, target_accent: str, source_sample_rate: int }
  returns: { converted_audio: bytes, metadata: ConversionMetadata }

GET /accents
  returns: { available_accents: List[str] }

POST /evaluate
  body: { source_audio: bytes, converted_audio: bytes }
  returns: { emotion_similarity: float, wer: float, mos_estimate: float }
```

### 7.2 Service Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
│                    (main.py)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────────────┐
│  Audio       │ │ Model    │ │ Reference Clip Bank       │
│  Preprocessor│ │ Manager  │ │                          │
│              │ │          │ │ /refs/indian_english/*.wav│
│  - VAD       │ │ Seed-VC  │ │ /refs/chinese_english/   │
│  - Norm      │ │ Vevo     │ │ /refs/japanese_english/  │
│  - Segment   │ │ Whisper  │ │ /refs/british_english/   │
│              │ │ SER      │ │ /refs/american_english/  │
└──────────────┘ │ BigVGAN  │ └──────────────────────────┘
                 └──────────┘
```

### 7.3 Model Loading Strategy

Load all models once at server startup, keep in GPU memory. On RTX 4060 8GB:

| Model | VRAM | Load time |
|---|---|---|
| Whisper large-v3 | ~3.0 GB | ~8s |
| Seed-VC V2 (AR + CFM) | ~2.5 GB | ~5s |
| Vevo-Voice | ~1.5 GB | ~4s |
| wav2vec2-large SER | ~1.2 GB | ~3s |
| BigVGAN-v2 | ~0.5 GB | ~2s |
| **Total** | **~8.7 GB** | - |

> RTX 4060 has 8GB VRAM — this is tight. **Mitigation**: either load Vevo on CPU (slower but works) or use the 4-bit quantised Seed-VC checkpoint and drop Vevo from the ensemble for production, only using Vevo during evaluation/competition demo with pre-computed outputs.

Alternatively, on a GPU droplet (A10G = 24GB, ~$1/hr on Lambda Labs or Vast.ai): load everything comfortably, run full ensemble.

### 7.4 Inference Pipeline (per request)

```
Request received
    │
    ▼
Decode audio bytes → WAV tensor (CPU)
    │
    ▼
VAD segmentation → list of (start, end, audio) segments
    │
    ▼
For each segment (parallelisable across GPU streams):
    ├─ Whisper transcription
    ├─ SER emotion extraction
    ├─ Prosody feature extraction (pyworld)
    ├─ Seed-VC V2 conversion
    ├─ Vevo conversion
    ├─ Quality selection
    └─ Emotion correction if needed
    │
    ▼
Overlap-add assembly
    │
    ▼
Loudness normalisation
    │
    ▼
Return WAV bytes
```

---

## 8. Evaluation Setup

Implement these metrics as an automated evaluation script:

| Metric | Tool | Target |
|---|---|---|
| Accent accuracy | Train XVector accent classifier on VCTK/GLOBE, report top-1 | >70% on held-out set |
| Emotion preservation | Cosine similarity of SER embeddings (source vs output) | >0.85 average |
| Intelligibility | WER via Whisper large-v3 | <10% relative increase vs source |
| Voice naturalness | UTMOS-strong (automated MOS predictor) | >3.8 / 5.0 |
| Speaker agnosticism | Verify output speaker not similar to source via ECAPA-TDNN embedding | Cosine sim < 0.5 |

---

## 9. Key Papers

| Paper | Year | Relevance |
|---|---|---|
| StyleStream (arxiv 2602.20113) | 2026 | First joint accent+emotion+timbre conversion, SOTA |
| Seed-VC (arxiv 2411.09943) | 2024 | Primary converter; DiT + CFM architecture |
| Vevo (ICLR 2025) | 2025 | Ensemble leg; VQ-VAE disentanglement, 101k hour training |
| Convert and Speak (arxiv 2408.10096) | 2024 | Zero-shot accent conversion, Microsoft Research |
| PSDN — ByteDance (Interspeech 2022) | 2022 | Zero-shot reference-free accent baseline |
| StarGAN-VC++ (arxiv 2309.07592) | 2023 | Emotion-preserving VC with emotion-aware losses |
| L2-ARCTIC corpus | 2018 | Primary fine-tune dataset; parallel non-native English |

---

## 10. Recommended Build Order

1. **Day 1-2**: Set up environment, download all pretrained weights (Seed-VC V2, Vevo, Whisper large-v3, SER model, BigVGAN). Verify each model runs independently on sample audio.

2. **Day 3**: Build the orchestration pipeline — VAD → feature extraction → Seed-VC conversion → emotion correction → output. Test on 10 source utterances across 3 target accents.

3. **Day 4**: Add Vevo ensemble leg and quality selection layer. Collect reference clips for 5 target accents from Common Voice.

4. **Day 5**: Build FastAPI server, evaluation script with all 5 metrics. Run automated evaluation on L2-ARCTIC test split (100 utterances).

5. **Day 6-7**: Fine-tune style encoder on L2-ARCTIC parallel pairs if accent accuracy is below 65%. Otherwise, spend time on demo UI (Gradio) and reference clip curation.

6. **Check StyleStream repo** at any point — if weights are released, integrate as primary model replacing Seed-VC in step 2.
