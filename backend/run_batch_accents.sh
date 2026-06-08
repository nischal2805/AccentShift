#!/usr/bin/env bash
# Batch accent conversion — convert_style=false + energy-only EC, similarity 2.5.
# Source is Indian-accented; convert to every other accent. Clean output folder.
set -u

cd "$(dirname "$0")" || exit 1   # always run from backend/ (script dir)

IN="C:/Users/nisch/Downloads/test_nis.wav"
OUTDIR="tests/outputs/batch_styleoff"
mkdir -p "$OUTDIR"

export USE_TF=0
export USE_TORCH=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ACCENTS=(
  chinese_english
  japanese_english
  korean_english
  arabic_english
  british_english
  american_english
  nigerian_english
)

echo "=== batch start $(date) ===" | tee "$OUTDIR/_batch.log"
for acc in "${ACCENTS[@]}"; do
  echo ">>> $acc $(date +%H:%M:%S)" | tee -a "$OUTDIR/_batch.log"
  uv run python run_pipeline.py \
    --input "$IN" \
    --target-accent "$acc" \
    --output "$OUTDIR/${acc}.wav" \
    >> "$OUTDIR/_batch.log" 2>&1
  echo "<<< $acc done rc=$? $(date +%H:%M:%S)" | tee -a "$OUTDIR/_batch.log"
done
echo "=== batch done $(date) ===" | tee -a "$OUTDIR/_batch.log"
