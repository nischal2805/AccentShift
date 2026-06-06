# AccentShift — Usage Guide

## What It Does

Converts English speech from one accent to another while preserving emotion and prosody.

Pipeline: VAD → ASR + SER → Seed-VC V2 (accent conversion) → emotion correction → loudness norm

## Requirements

- Python 3.10
- RTX 4060 8GB+ (runs Whisper-small + SER + Seed-VC on CUDA, ~5.5GB VRAM)
- 8GB+ RAM (16GB+ recommended)

## Setup

```powershell
cd D:\designathon_2\backend
# venv already created — activate:
.\.venv\Scripts\Activate.ps1
```

Set HF cache before every run:
```powershell
$env:HF_HOME = "D:\designathon_2\backend\.hf_cache"
```

## Run Command

```powershell
cd "D:\designathon_2\backend"
$env:HF_HOME = "D:\designathon_2\backend\.hf_cache"

.\.venv\Scripts\python.exe run_pipeline.py `
  --input  "path\to\input.wav" `
  --target-accent "indian_english" `
  --output "output_test_name.wav" `
  --metrics-out "output_test_name.json"
```

### All flags

| Flag | Required | Description |
|------|----------|-------------|
| `--input` | yes | Source WAV (any sample rate, mono/stereo) |
| `--target-accent` | yes | Target accent key (see below) |
| `--output` | yes | Output WAV path |
| `--metrics-out` | no | JSON file with per-segment metrics |
| `--config` | no | Override config file (default: `configs/pipeline_config.yaml`) |

### Valid accent keys

| Key | Description | References available |
|-----|-------------|---------------------|
| `indian_english` | Indian-accented English | ✅ |
| `chinese_english` | Chinese (Mandarin)-accented English | ✅ |
| `japanese_english` | Japanese-accented English | ⚠️ add WAVs to `references/japanese_english/` |
| `british_english` | British English | ⚠️ add WAVs to `references/british_english/` |
| `american_english` | American English | ⚠️ add WAVs to `references/american_english/` |

## Reference WAVs

References are 5–15s WAV clips of a native speaker of the target accent. The pipeline copies
accent style from the reference — better reference = better conversion.

Place WAVs in: `references/<accent_key>/<any_name>.wav`

**Sourcing references from L2-Arctic:**
```
E:\l2arctic_extracted\
  ASI\wav\   → indian_english
  BWC\wav\   → chinese_english
  HJK\wav\   → korean_english
  ...
```
Speaker mapping: ASI, RRBI, SVBI, TNI → indian | BWC, LXC, NCC, TXHC → chinese | HJK, HKK → korean

Pick 2–3 short utterances (5–10s) per accent and copy to `references/<key>/`.

**American / British English references:**
L2-Arctic does not include native American or British speakers.
Use LibriSpeech (American, freely available) or any clean recording.

## Output Files

| File | Description |
|------|-------------|
| `output_test_<name>.wav` | Accent-converted audio, 16kHz PCM-16 |
| `output_test_<name>.json` | Per-segment metrics |

### Reading metrics

```json
{
  "processing_time_ms": 69085,
  "n_segments": 2,
  "emotion_similarity": 0.9998,   // cosine sim of VAD/arousal/dominance vectors (1.0 = perfect)
  "wer": 0.25,                    // word error rate (0.0 = perfect transcription match)
  "mos_estimate": 3.5,            // UTMOS22 score (3.5 = fallback; real scores 1–5)
  "segments": [...]               // per-segment breakdown with chosen_backend
}
```

`emotion_similarity` near 1.0 = emotion preserved. `wer` < 0.5 = intelligibility good.

## Config

`configs/pipeline_config.yaml` — key settings:

```yaml
seed_vc:
  cfm_checkpoint_path: "checkpoints/seedvc_finetuned/all_ft/CFM_epoch_00016_step_14000.pth"
  diffusion_steps: 30        # lower = faster but worse quality (min ~10)

vevo:
  enabled: false             # CPU-only, ~11 min/run — keep disabled unless you have time

devices:
  whisper_vram_gb_threshold: 4   # whisper on GPU if VRAM >= 4GB (use 12 to force CPU)
```

## Performance (RTX 4060 8GB)

| Config | Time (7 segs) | VRAM |
|--------|--------------|------|
| Seed-VC only (Vevo off) | ~70–120s | ~5.5GB |
| Seed-VC + Vevo (CPU) | ~650s | ~5.5GB GPU + 28GB RAM |

## Checkpoint

Finetuned on L2-Arctic (all accents, 14000 steps):
`checkpoints/seedvc_finetuned/all_ft/CFM_epoch_00016_step_14000.pth`

AR checkpoint auto-downloads from `Plachta/Seed-VC` on first run (~400MB).

## Known Issues

| Issue | Status |
|-------|--------|
| UTMOS22 hubconf.py missing | Non-critical — MOS defaults to 3.5 |
| Vevo slow on CPU (~11 min) | Disabled by default |
| High WER on short clips | Expected — Whisper-small + heavy accent transfer |
| american/british reference WAVs missing | Add WAVs manually to `references/` |

## Roadmap

- [ ] Emformer-based emotion encoder for better prosody capture
- [ ] Sequential model loading to reduce peak RAM
- [ ] American/British reference WAVs from LibriSpeech
- [ ] UTMOS22 cache fix
- [ ] Vevo GPU support via sequential loading
