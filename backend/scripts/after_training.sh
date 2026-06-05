#!/usr/bin/env bash
# Run on LOCAL machine after training completes on A100.
# Copies the checkpoint back and wires it into the pipeline config.
#
# Usage:
#   bash scripts/after_training.sh <user@a100-host> <run_name> <accent_key>
#
# Example:
#   bash scripts/after_training.sh root@123.45.67.89 indian_english_ft indian_english
#
# What this does:
#   1. SCP runs/<run_name>/ from A100 -> local checkpoints/seedvc_finetuned/<accent>/
#   2. Prints the pipeline_config.yaml line to set
set -euo pipefail

A100_HOST="${1:?Usage: $0 <user@host> <run_name> <accent_key>}"
RUN_NAME="${2:?}"
ACCENT="${3:?}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_DIR/checkpoints/seedvc_finetuned/$ACCENT"
mkdir -p "$DEST"

echo "=== Copying checkpoint from A100 ==="
# Adjust A100 project path if different
A100_PROJECT="/root/designathon_2/backend"
scp -r "${A100_HOST}:${A100_PROJECT}/runs/${RUN_NAME}/" "$DEST/"

echo ""
echo "=== Checkpoint saved to: $DEST ==="
echo ""

# Seed-VC max_keep=1 so only one checkpoint survives. Just grab it directly.
LATEST_CFM=$(find "$DEST" -name "CFM_epoch_*_step_*.pth" | head -1)
LATEST_AR=$(find "$DEST" -name "AR_epoch_*_step_*.pth" 2>/dev/null | head -1 || true)

if [ -z "$LATEST_CFM" ]; then
    echo "WARNING: No CFM checkpoint found in $DEST"
    exit 1
fi

echo "=== Update configs/pipeline_config.yaml ==="
echo ""
echo "Set these under seed_vc:"
echo "  cfm_checkpoint_path: \"$LATEST_CFM\""
if [ -n "$LATEST_AR" ]; then
    echo "  ar_checkpoint_path:  \"$LATEST_AR\""
fi
echo ""
echo "=== Then test the pipeline ==="
echo "python run_pipeline.py \\"
echo "  --input references/${ACCENT}/<ref>.wav \\"
echo "  --target-accent ${ACCENT} \\"
echo "  --output out_finetuned.wav \\"
echo "  --metrics-out metrics_finetuned.json"
