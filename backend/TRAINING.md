# AccentShift — A100 Fine-Tuning Guide

Complete guide for training Seed-VC V2 on L2-Arctic accent data and retrieving the checkpoint.
Run everything on the A100 box. The 8GB 4060 cannot train.

---

## What gets trained

Only the **CFM (flow-matching style encoder) + AR decoder** — ~100M combined params.
Content encoder, vocoder, and speaker encoder are frozen.
This teaches the model accent-phonetic patterns beyond what the reference clip alone provides.

Training on all accents mixed = one universal checkpoint that generalizes better.
Training per-accent = sharper accent output but one checkpoint per accent.
**Recommendation: train all accents together (default below).**

---

## Step 1 — Clone + setup on A100

```bash
git clone <your-repo-url> designathon_2
cd designathon_2/backend
bash scripts/a100_setup.sh
```

`a100_setup.sh` does:
- Installs uv
- Creates `.venv` with Python 3.10
- Installs torch cu121 + all deps
- Clones seed-vc + Amphion into `third_party/`
- Runs `download_models.py` (whisper-large-v3, SER, ECAPA, Vevo)

Expected time: ~20 min (mostly model downloads).

---

## Step 2 — Download L2-Arctic dataset

L2-Arctic v5.0 — 24 speakers, ~1150 utterances each (~3.7 GB total):

```bash
# All accents at once (recommended):
bash scripts/download_l2arctic.sh

# Or single accent:
bash scripts/download_l2arctic.sh indian_english
```

### Accent → L2-Arctic speaker mapping

| Accent key | L2-Arctic speakers | L1 background |
|---|---|---|
| `indian_english` | ASI, MBMPS | Hindi |
| `chinese_english` | HKK, YBAA | Mandarin |
| `korean_english` | YDCK, YKWK | Korean |
| `arabic_english` | ERMS, RRBI | Arabic |
| `spanish_english` | EBVS, NJS | Spanish |
| `vietnamese_english` | HQTV, PNV | Vietnamese |
| `japanese_english` | *(not in L2-Arctic)* | Provide your own WAVs |

**For japanese_english**: source 5–10 min of Japanese-accented English WAVs manually
(e.g., from Mozilla Common Voice `ja` accented English readings) and place in
`data/finetune/japanese_english/`.

After download: WAVs organized under `data/finetune/<accent>/` (~1150 files per accent).

---

## Step 3 — Train

### Option A: All accents combined (recommended)

```bash
python scripts/finetune_style.py \
    --accent all \
    --steps 80000 \
    --batch-size 16 \
    --save-every 2000 \
    --num-workers 8 \
    --mixed-precision bf16 \
    --train-ar
```

Merges all accent WAVs → `data/finetune_all/` → trains one universal checkpoint.
**Estimated time on A100 40GB: ~12–16 hours for 80k steps.**

### Option B: Per-accent (sharper but 6 separate runs)

```bash
for ACCENT in indian_english chinese_english korean_english arabic_english spanish_english; do
    python scripts/finetune_style.py \
        --accent "$ACCENT" \
        --steps 50000 \
        --batch-size 16 \
        --save-every 1000 \
        --num-workers 8 \
        --mixed-precision bf16 \
        --train-ar
done
```

**~8 hours per accent on A100.**

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--steps` | 50000 | 80k recommended for all-accent |
| `--batch-size` | 16 | Max for A100 40GB with bf16 |
| `--train-ar` | True | Fine-tunes AR decoder too (full accent transfer) |
| `--mixed-precision` | bf16 | A100: bf16. V100: fp16. Debug: no |
| `--save-every` | 1000 | Save checkpoint every N steps |

### Monitor training

```bash
# Loss should drop from ~3.0 to ~0.8-1.2 over 50k steps
tail -f runs/<run_name>/training.log

# Or tensorboard (if installed):
tensorboard --logdir runs/
```

---

## Step 4 — Retrieve checkpoint via SCP

Run this on your **local machine** after training:

```bash
bash scripts/after_training.sh root@<a100-ip> all_ft all
```

Or manually:
```bash
# From local machine:
scp -r root@<a100-ip>:/root/designathon_2/backend/runs/all_ft/ \
    D:/designathon_2/backend/checkpoints/seedvc_finetuned/all/
```

Checkpoint files:
- `CFM_epoch_XXXXX_step_XXXXX.pth` — style encoder weights
- `AR_epoch_XXXXX_step_XXXXX.pth` — AR decoder weights

---

## Step 5 — Wire checkpoint into pipeline

Edit `configs/pipeline_config.yaml`:

```yaml
seed_vc:
  # Point at the retrieved checkpoint:
  cfm_checkpoint_path: "checkpoints/seedvc_finetuned/all/CFM_epoch_00080_step_80000.pth"
  ar_checkpoint_path:  "checkpoints/seedvc_finetuned/all/AR_epoch_00080_step_80000.pth"
```

Then test:
```powershell
$env:HF_HOME = "D:\designathon_2\backend\.hf_cache"
cd D:\designathon_2\backend
.\.venv\Scripts\python.exe run_pipeline.py `
    --input references\indian_english\<ref>.wav `
    --target-accent indian_english `
    --output out_finetuned.wav `
    --metrics-out metrics.json
```

---

## Dataset sizes

| What | Size |
|---|---|
| L2-Arctic v5 zip | ~3.7 GB |
| Extracted WAVs | ~4.5 GB |
| Seed-VC + deps | ~8 GB (already in HF cache) |
| Training checkpoints | ~800 MB each (CFM) |
| **Total A100 disk needed** | ~20 GB |

---

## Troubleshooting

**OOM during training:**
- Reduce `--batch-size` to 8 and increase `--steps` proportionally
- Disable `--train-ar` (CFM only is faster + less memory)

**Loss not decreasing after 10k steps:**
- Learning rate might be too high — Seed-VC default is 2e-5 (hardcoded in `build_single_optimizer`)
- Check data: ensure WAVs are 1–30 seconds (shorter/longer are skipped by FT_Dataset)

**Japanese accent data:**
- L2-Arctic has no Japanese speakers
- Alternatives: JVS corpus (Japanese), Common Voice accented English, or record your own
- Place WAVs in `data/finetune/japanese_english/` before running finetune

**Checkpoint not improving accent quality:**
- Fine-tuned model trained on non-native speech — this teaches the model the accent patterns
- Quality also depends heavily on the reference WAV in `references/<accent>/`
- Use a clear 5–10s native/target-accent reference with no background noise
