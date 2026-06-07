# AccentFlow — New Pipeline Architecture

**Goal**: Convert accent (e.g., American → Indian English) while preserving source speaker's voice, emotion, and intelligibility. Zero model training. Implementable in 2 days.

---

## Why the Old Pipeline Fails

### The single most critical bug

In `backend/pipeline/converter.py:189`, Vevo is called like this:

```python
self._pipeline.inference_ar_and_fm(
    src_wav_path=src_path,
    style_ref_wav_path=ref_abs,   # accent reference
    timbre_ref_wav_path=ref_abs,  # SAME audio — WRONG
)
```

Vevo has separate slots for **timbre** (who the speaker sounds like) and **style** (accent/speaking style). Passing the same audio for both = full voice conversion to the reference speaker. You lose the source speaker's voice entirely.

The fix is one line: pass the **source audio** as `timbre_ref_wav_path`.

```python
self._pipeline.inference_ar_and_fm(
    src_wav_path=src_path,
    style_ref_wav_path=accent_ref_path,  # target accent
    timbre_ref_wav_path=src_path,         # source speaker's own voice
)
```

This is the architectural difference between voice conversion and accent conversion. Vevo was designed to support this. The old pipeline did not use it.

### Other failures

| Problem | Old Approach | Impact |
|---|---|---|
| Emotion metric | Audeering V/A/D cosine on near-zero vectors | Always 0.9999 — useless |
| WER | Whisper on accent-converted output | 78-96% — meaningless |
| MOS | UTMOS22 stub = 3.5 always | Zero discriminative power |
| Emformer emotion | ASR model → emotion projection | Untested, probably noise |
| CWT F0 mode | Custom wavelet, scipy removed | Fragile |
| Seed-VC as primary | Voice conversion model | Changes speaker identity |

---

## New Architecture

```
Input: source_audio.wav + target_accent (string key)
Output: output.wav + metrics.json

┌─────────────────────────────────────────────────────────┐
│ Stage 1: Ingest                                          │
│   Load → resample 16kHz mono → peak-normalize           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 2: VAD Segmentation                                │
│   Silero VAD → speech segments, max 20s per segment      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 3: Source Feature Snapshot  (before conversion)    │
│   Per segment:                                           │
│   ├── PyWorld HARVEST → F0 contour + voiced mask         │
│   ├── Librosa RMS frames → energy envelope               │
│   ├── ECAPA-TDNN → 192-d speaker embedding               │
│   └── Whisper (source only) → transcript                 │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 4: Reference Selection                             │
│   Load reference clip pool for target accent             │
│   Select = argmin cosine_dist(ECAPA(src), ECAPA(ref))    │
│   Rationale: closest timbre → minimal residual drift     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 5: Vevo-Style Conversion  ← CORE CHANGE            │
│   inference_ar_and_fm(                                   │
│     src_wav_path      = segment,                         │
│     timbre_ref_wav_path = segment,   ← SOURCE VOICE      │
│     style_ref_wav_path  = accent_ref ← TARGET ACCENT     │
│   )                                                      │
│   Output: target-accent phonetics, source timbre kept    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 6: Prosody Restoration  (conditional)              │
│   Extract output F0 (PyWorld)                            │
│   Compute: r = pearson_r(src_f0_voiced, out_f0_voiced)   │
│   If r < 0.72:                                           │
│     Apply log-norm F0 transfer (preserve contour shape)  │
│   Always: RMS envelope match (frame-level gain scaling)  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 7: Quality Evaluation  (real metrics only)         │
│   speaker_sim   = cosine(ECAPA(src), ECAPA(out))         │
│   f0_corr       = pearson_r(src_f0_v, out_f0_v)          │
│   wer           = jiwer(out_asr, src_transcript)         │
│   Log all. Warn if speaker_sim < 0.80.                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Stage 8: Assemble                                        │
│   Crossfade segments (30ms overlap-add)                  │
│   Loudness normalize → -23 LUFS                          │
│   Write output.wav + metrics.json                        │
└─────────────────────────────────────────────────────────┘
```

---

## Stage Implementation Details

### Stage 1 — Ingest

```python
import librosa, soundfile as sf, numpy as np

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95
    return wav
```

### Stage 2 — VAD Segmentation

Use Silero VAD exactly as in the old preprocessor. No change needed.
Max segment: **20 seconds** (Vevo internal limit, safer than 30s Seed-VC limit).

```python
# Silero VAD — same as old pipeline
# torch.hub.load("snakers4/silero-vad", "silero_vad")
# Merge segments < 200ms apart, split any > 20s
MAX_SEG_S = 20.0
```

### Stage 3 — Source Feature Snapshot

```python
import pyworld as pw
import librosa
import numpy as np

def extract_source_features(wav: np.ndarray, sr: int = 16000) -> dict:
    # F0 + voiced mask
    wav64 = wav.astype(np.float64)
    f0, t = pw.harvest(wav64, sr, frame_period=5.0)  # 5ms frames
    voiced = f0 > 0

    # Energy envelope (frame-level RMS)
    hop = 512
    frames = librosa.util.frame(wav, frame_length=1024, hop_length=hop)
    energy = np.sqrt(np.mean(frames ** 2, axis=0))

    return {
        "f0": f0,               # shape [T_frames]
        "voiced": voiced,       # bool mask
        "energy": energy,       # shape [T_energy_frames]
        "sr": sr,
    }
```

**ECAPA speaker embedding** — use speechbrain:

```python
from speechbrain.inference.speaker import EncoderClassifier

encoder = EncoderClassifier.from_hdf5("speechbrain/spkrec-ecapa-voxceleb", 
                                       run_opts={"device": "cuda"})

def get_speaker_embed(wav_path: str) -> np.ndarray:
    embed = encoder.encode_file(wav_path)
    return embed.squeeze().cpu().numpy()  # 192-d
```

**Whisper transcript** — run ONCE on source, store. Do NOT re-run on converted output.

```python
import whisper
model = whisper.load_model("large-v3")  # or "small" if VRAM constrained

def transcribe_source(wav_path: str) -> str:
    result = model.transcribe(wav_path, language="en", fp16=True)
    return result["text"]
```

### Stage 4 — Reference Selection

Pre-compute ECAPA embeddings for all reference clips at startup (once).

```python
def select_reference(
    src_embed: np.ndarray,
    ref_pool: dict[str, np.ndarray],  # {path: embed}
) -> str:
    def cosine_dist(a, b):
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    return min(ref_pool, key=lambda p: cosine_dist(src_embed, ref_pool[p]))
```

Reference clips: 3-10 second clean recordings of the target accent.
Store under `references/<accent_key>/ref_01.wav`, `ref_02.wav`, etc.

For best results: include references from multiple genders and pitch registers
so closest-match selection has meaningful choices.

### Stage 5 — Vevo-Style Conversion (THE CORE CHANGE)

```python
def convert_vevo(
    src_wav_path: str,
    accent_ref_path: str,
    pipeline: VevoInferencePipeline,
    flow_matching_steps: int = 32,
) -> np.ndarray:
    out_tensor = pipeline.inference_ar_and_fm(
        src_wav_path=src_wav_path,
        src_text=None,
        style_ref_wav_path=accent_ref_path,   # target accent style
        timbre_ref_wav_path=src_wav_path,      # SOURCE VOICE for timbre
        flow_matching_steps=flow_matching_steps,
    )
    # Returns torch.Tensor [1, T] at 24kHz
    wav = out_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return wav  # resample to 16kHz downstream if needed
```

**Critical**: `timbre_ref_wav_path = src_wav_path` is the entire architectural
difference from the old pipeline. This tells Vevo "sound like this voice,
but speak with this accent style."

**Vevo on GPU**: The config has `vevo.device: "cpu"` and `vram_gb_threshold: 10`.
Vevo weights are ~1.8GB. On 8GB GPU with headroom:
- Lower threshold: `vram_gb_threshold: 6`
- Set `vevo.device: "cuda"`
- GPU cuts ~11 min/segment → ~45 sec/segment

### Stage 6 — Prosody Restoration

Vevo with source-as-timbre-reference should already preserve F0 register well.
Apply correction only when measured correlation drops:

```python
def restore_prosody(
    src_features: dict,
    out_wav: np.ndarray,
    out_sr: int,
    f0_corr_threshold: float = 0.72,
) -> np.ndarray:
    # Extract output F0
    out64 = out_wav.astype(np.float64)
    out_f0, _ = pw.harvest(out64, out_sr, frame_period=5.0)
    out_voiced = out_f0 > 0

    src_f0 = src_features["f0"]
    src_voiced = src_features["voiced"]

    # Align to shorter
    min_len = min(len(src_f0), len(out_f0))
    s_f = src_f0[:min_len]
    o_f = out_f0[:min_len]
    both_voiced = src_voiced[:min_len] & out_voiced[:min_len]

    if both_voiced.sum() > 10:
        r = np.corrcoef(s_f[both_voiced], o_f[both_voiced])[0, 1]
    else:
        r = 1.0

    # F0 log-norm transfer if correlation dropped
    if r < f0_corr_threshold:
        s_log = np.log(s_f[both_voiced] + 1e-8)
        o_log = np.log(o_f[both_voiced] + 1e-8)
        shift = s_log.mean() - o_log.mean()
        # shift output F0 mean toward source, keep contour shape
        out_f0_corrected = out_f0.copy()
        out_f0_corrected[out_voiced] = np.exp(
            np.log(out_f0[out_voiced] + 1e-8) + shift
        )
        # Resynthesize via WORLD
        sp, ap = pw.wav2world(out64, out_sr)[1:]
        out_wav = pw.synthesize(out_f0_corrected, sp, ap, out_sr).astype(np.float32)

    # Energy envelope matching (always apply, lightweight)
    src_energy = src_features["energy"]
    hop = 512
    out_frames = librosa.util.frame(out_wav, frame_length=1024, hop_length=hop)
    out_energy = np.sqrt(np.mean(out_frames ** 2, axis=0))
    
    min_e = min(len(src_energy), len(out_energy))
    ratio = (src_energy[:min_e] + 1e-8) / (out_energy[:min_e] + 1e-8)
    ratio = np.clip(ratio, 0.5, 2.0)
    ratio = np.convolve(ratio, np.ones(5) / 5, mode="same")  # smooth
    
    scale = np.interp(
        np.arange(len(out_wav)),
        np.arange(min_e) * hop + hop // 2,
        ratio[:min_e],
    )
    out_wav = (out_wav * scale).astype(np.float32)

    return out_wav, r  # return r for logging
```

### Stage 7 — Quality Evaluation

Replace broken metrics with real ones:

```python
import jiwer
from scipy.stats import pearsonr

def evaluate_segment(
    src_embed: np.ndarray,
    out_embed: np.ndarray,
    src_f0: np.ndarray,
    out_f0: np.ndarray,
    src_voiced: np.ndarray,
    out_voiced: np.ndarray,
    src_transcript: str,
    out_transcript: str,
) -> dict:
    # 1. Speaker identity preserved?
    speaker_sim = float(
        np.dot(src_embed, out_embed) /
        (np.linalg.norm(src_embed) * np.linalg.norm(out_embed) + 1e-8)
    )

    # 2. Emotion proxy — F0 contour correlation
    min_len = min(len(src_f0), len(out_f0))
    both_v = src_voiced[:min_len] & out_voiced[:min_len]
    if both_v.sum() > 10:
        f0_corr = float(pearsonr(src_f0[:min_len][both_v], out_f0[:min_len][both_v])[0])
    else:
        f0_corr = 0.0

    # 3. Intelligibility — WER vs SOURCE transcript (not re-ASR)
    wer = jiwer.wer(src_transcript, out_transcript)

    return {
        "speaker_sim": speaker_sim,   # target: > 0.80
        "f0_corr": f0_corr,           # target: > 0.70 (emotion proxy)
        "wer": wer,                   # target: < 0.35
    }
```

**Why these metrics work:**

| Metric | Dynamic Range | What It Measures |
|---|---|---|
| ECAPA cosine | 0.0 – 1.0 (real range) | Speaker identity preservation |
| F0 pearson r | -1.0 – 1.0 (real range) | Prosodic/emotional contour similarity |
| WER vs source transcript | 0.0 – 1.0 (vs ground truth) | Intelligibility |

No saturation. No stubs. No broken UTMOS.

### Stage 8 — Assemble

Same as old postprocessor:
```python
# 30ms crossfade between segments
# pyloudnorm to -23 LUFS
# Write WAV + JSON sidecar
```

---

## Directory Structure

```
accentflow/
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py          # Stage 1
│   ├── segmenter.py       # Stage 2 (Silero VAD)
│   ├── features.py        # Stage 3 (F0, energy, ECAPA, Whisper)
│   ├── selector.py        # Stage 4 (reference selection)
│   ├── converter.py       # Stage 5 (Vevo wrapper)
│   ├── restorer.py        # Stage 6 (prosody restoration)
│   ├── evaluator.py       # Stage 7 (real metrics)
│   └── assembler.py       # Stage 8 (crossfade + LUFS)
├── models.py              # Load all models once at startup
├── run.py                 # CLI entrypoint
├── config.yaml
└── references/
    ├── indian_english/    # 3-10 WAV clips
    ├── chinese_english/
    ├── british_english/
    └── ...
```

---

## Model Setup

All pretrained, no training needed:

```
Model               Source                              VRAM
────────────────────────────────────────────────────────────
Silero VAD          torch.hub (snakers4/silero-vad)     ~50MB
Whisper large-v3    openai/whisper-large-v3             ~3GB
  (or small)        openai/whisper-small                ~500MB
ECAPA-TDNN          speechbrain/spkrec-ecapa-voxceleb   ~100MB
Vevo                amphion/Vevo (HuggingFace)          ~1.8GB
PyWorld             pip install pyworld                 CPU only
pyloudnorm          pip install pyloudnorm              CPU only
jiwer               pip install jiwer                   CPU only
```

**8GB GPU allocation strategy:**
- Whisper large-v3 on GPU (3GB) — run once per segment, then unload or keep
- Vevo on GPU (1.8GB) — keep loaded, used per segment
- ECAPA on GPU (100MB) — always keep loaded
- Total: ~5GB VRAM with headroom

If VRAM tight: use `whisper-small` (~500MB) or run Whisper on CPU (slower but works).

```yaml
# config.yaml
audio:
  sample_rate: 16000
  vevo_sr: 24000       # Vevo native, convert back after

vad:
  max_segment_s: 20.0  # Vevo limit
  threshold: 0.5
  min_speech_ms: 250

models:
  whisper: "openai/whisper-large-v3"
  ecapa: "speechbrain/spkrec-ecapa-voxceleb"
  device: "cuda"
  vevo_device: "cuda"   # set to cpu if OOM

vevo:
  flow_matching_steps: 32  # reduce to 16 for speed
  repo_dir: "../third_party/Amphion"
  hf_repo: "amphion/Vevo"
  cache_dir: "../checkpoints/Vevo"

prosody:
  f0_corr_threshold: 0.72
  energy_smooth_frames: 5
  energy_clip: [0.5, 2.0]

quality:
  speaker_sim_warn: 0.80
  f0_corr_warn: 0.70
  wer_warn: 0.35

paths:
  references: "references"
```

---

## Reference Clips — What to Use

You need 3-10 WAV clips per accent (3-10 seconds each, clean, single speaker).

Sources (no copyright issues):
- **L2-Arctic**: Already in `backend/references/` presumably. Use those.
- **LibriSpeech**: British English speakers exist.
- **VCTK corpus**: Multiple accents, clean studio recordings.
- **Common Voice**: Accent-labeled speakers available.
- **Record yourself / friends**: Fastest option for testing.

Quality requirements: 16kHz+, minimal background noise, single speaker talking at natural pace.

For reference selection to work well: include both male and female speakers per accent,
covering low/mid/high pitch registers. Minimum 2 references per accent; 5+ is better.

---

## run.py CLI

```python
#!/usr/bin/env python3
"""AccentFlow — accent conversion with emotion preservation."""
import argparse, json
from pathlib import Path
from models import load_models
from pipeline.ingest import load_audio
from pipeline.segmenter import segment_audio
from pipeline.features import extract_features
from pipeline.selector import select_reference
from pipeline.converter import convert_vevo
from pipeline.restorer import restore_prosody
from pipeline.evaluator import evaluate_segment
from pipeline.assembler import assemble

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Source audio file")
    p.add_argument("accent", help="Target accent key (e.g. indian_english)")
    p.add_argument("output", help="Output WAV path")
    args = p.parse_args()

    models = load_models()  # loads everything once

    wav = load_audio(args.input)
    segments = segment_audio(wav, models.vad)
    
    results = []
    converted_segments = []
    
    for seg in segments:
        features = extract_features(seg, models)
        ref_path = select_reference(features["speaker_embed"], 
                                    models.ref_pool[args.accent])
        out_wav = convert_vevo(seg.path, ref_path, models.vevo)
        out_wav, f0_corr = restore_prosody(features, out_wav, sr=24000)
        metrics = evaluate_segment(features, out_wav, models)
        
        converted_segments.append(out_wav)
        results.append(metrics)
        print(f"  speaker_sim={metrics['speaker_sim']:.3f}  "
              f"f0_corr={f0_corr:.3f}  wer={metrics['wer']:.3f}")

    final = assemble(converted_segments, sr=16000)
    
    output_path = Path(args.output)
    import soundfile as sf
    sf.write(output_path, final, 16000)
    
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Done → {output_path}")

if __name__ == "__main__":
    main()
```

---

## What This Fixes vs Old Pipeline

| Problem | Old Pipeline | New Pipeline |
|---|---|---|
| Vevo timbre ref | Same as accent ref (voice conversion) | Source audio (accent conversion) |
| Emotion metric | V/A/D cosine → 0.9999 always | F0 pearson r → real range |
| Speaker metric | Not measured | ECAPA cosine → real range |
| WER reference | Re-ASR on converted output | Source transcript (ground truth) |
| MOS | Stub 3.5 | Removed (not needed for ranking) |
| Emformer | ASR→emotion projection (wrong) | Removed |
| Seed-VC | Primary backend (voice converter) | Removed or optional fallback |
| F0 correction | Always runs, complex SER-gating | Conditional (triggers if F0 corr < 0.72) |
| CWT F0 mode | Custom wavelet, fragile | Removed |

---

## Known Limitations

**1. Vevo still needs a reference clip**
True zero-shot accent conversion (no reference speaker) is an unsolved research problem.
You always need at least one example of the target accent.
Mitigated by: reference selection matching source timbre.

**2. Japanese and British → other accents have no L2-Arctic training data**
Vevo is pretrained on 101k hours (Emilia, 6 languages) so it generalizes,
but quality will vary by accent pair.
Mitigation: use VCTK for British references, source additional Japanese clips.

**3. Emotion preservation is F0+energy only**
Spectral emotional qualities (breathiness, creakiness, tenseness) are NOT preserved.
These require explicit voice quality transfer, which needs a dedicated model.
F0+energy covers ~70-80% of perceived emotion — practical limit without training.

**4. Vevo flow_matching_steps = 32 is slow on CPU (~11 min/segment)**
On GPU: ~45 sec/segment at steps=32, ~20 sec at steps=16.
Reduce steps for demos: `flow_matching_steps: 16` cuts quality slightly but acceptable.

**5. Very short segments (<1s) degrade quality**
Both VAD and Vevo need enough signal. Filter segments < 0.5s.

---

## Build Order (2 days)

**Day 1:**
- [ ] models.py — load Silero VAD, ECAPA, Whisper, Vevo (all at startup)
- [ ] ingest.py + segmenter.py — reuse old preprocessor logic
- [ ] features.py — F0 (PyWorld), energy (librosa), ECAPA embed, Whisper transcript
- [ ] converter.py — Vevo with `timbre_ref=source`, `style_ref=accent_ref`
- [ ] Run end-to-end on one test file, check output sounds reasonable

**Day 2:**
- [ ] selector.py — ECAPA reference matching
- [ ] restorer.py — conditional log-norm F0 + energy envelope
- [ ] evaluator.py — ECAPA sim, F0 corr, WER vs source transcript
- [ ] assembler.py — crossfade + LUFS
- [ ] run.py CLI
- [ ] Test 3-4 accent pairs, review metrics

---

## Success Criteria

After day 2, these numbers are achievable with the new architecture:

| Metric | Old Pipeline | Expected New |
|---|---|---|
| speaker_sim (ECAPA) | Not measured | > 0.80 |
| f0_corr (emotion proxy) | 0.9999 (saturated) | 0.65 – 0.90 (real) |
| WER | 78-96% (broken) | 15-35% (vs source transcript) |
| Perceived accent change | Moderate (Seed-VC timbre drift) | Strong (Vevo style-only) |
