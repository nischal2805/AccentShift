#!/usr/bin/env bash
# Download L2-Arctic dataset and organize WAVs by accent for Seed-VC fine-tuning.
#
# Usage: bash scripts/download_l2arctic.sh [accent_key]
#   accent_key: one of indian_english | chinese_english | japanese_english |
#               korean_english | arabic_english | spanish_english | vietnamese_english
#   Omit to download ALL accents.
#
# L2-Arctic v5.0 (Zenodo): ~3.7 GB total, 24 speakers, ~1150 utterances/speaker.
# Each speaker maps to a native language. We organize by accent key.
#
# After this script: data/finetune/<accent_key>/ contains WAVs for training.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_DIR/data"
L2_RAW="$DATA_DIR/l2arctic_raw"
FT_DIR="$DATA_DIR/finetune"
mkdir -p "$L2_RAW" "$FT_DIR"

TARGET_ACCENT="${1:-all}"

# L2-Arctic speaker -> accent mapping (verified against L2-Arctic v5.0 README)
declare -A SPEAKER_ACCENT=(
    [ASI]="indian_english"   [RRBI]="indian_english"   [SVBI]="indian_english"  [TNI]="indian_english"
    [BWC]="chinese_english"  [LXC]="chinese_english"   [NCC]="chinese_english"  [TXHC]="chinese_english"
    [HJK]="korean_english"   [HKK]="korean_english"    [YDCK]="korean_english"  [YKWK]="korean_english"
    [HQTV]="vietnamese_english" [PNV]="vietnamese_english" [THV]="vietnamese_english" [TLV]="vietnamese_english"
    [EBVS]="spanish_english" [ERMS]="spanish_english"  [MBMPS]="spanish_english" [NJS]="spanish_english"
    [ABA]="arabic_english"   [SKA]="arabic_english"    [YBAA]="arabic_english"  [ZHAA]="arabic_english"
)

# Japanese L2-Arctic not in standard release — use Mozilla Common Voice
# for japanese_english or provide your own clips.

echo "=== Downloading L2-Arctic v5.0 from Zenodo ==="
# Zenodo record 7838566 — full corpus zip (~3.7 GB)
L2_ZIP="$L2_RAW/l2arctic_v5.zip"
if [ ! -f "$L2_ZIP" ]; then
    wget -c -O "$L2_ZIP" \
        "https://zenodo.org/records/7838566/files/l2arctic_v5.zip?download=1"
else
    echo "l2arctic_v5.zip already present, skipping download"
fi

echo "=== Extracting ==="
# Unzip main archive (overwrite silently with -o)
unzip -o -q "$L2_ZIP" -d "$L2_RAW"

# L2-Arctic v5 ships each speaker as a separate zip inside the main archive
for SPK_ZIP in "$L2_RAW"/*.zip; do
    [ "$SPK_ZIP" = "$L2_ZIP" ] && continue
    [ -f "$SPK_ZIP" ] || continue
    SPK_NAME=$(basename "$SPK_ZIP" .zip)
    echo "  Extracting speaker $SPK_NAME..."
    unzip -o -q "$SPK_ZIP" -d "$L2_RAW"
done

echo "=== Organizing WAVs by accent ==="
# Handle both flat layout (L2_RAW/SPEAKER/) and nested (L2_RAW/l2arctic_v5/SPEAKER/)
L2_ROOT="$L2_RAW/l2arctic_v5"

for SPEAKER in "${!SPEAKER_ACCENT[@]}"; do
    ACCENT="${SPEAKER_ACCENT[$SPEAKER]}"
    if [ "$TARGET_ACCENT" != "all" ] && [ "$ACCENT" != "$TARGET_ACCENT" ]; then
        continue
    fi

    SPK_WAV_DIR=""
    # Check nested layout first, then flat layout
    for CANDIDATE in "$L2_ROOT/$SPEAKER/wav" "$L2_ROOT/$SPEAKER" \
                     "$L2_RAW/$SPEAKER/wav" "$L2_RAW/$SPEAKER"; do
        if [ -d "$CANDIDATE" ]; then
            SPK_WAV_DIR="$CANDIDATE"
            break
        fi
    done

    if [ -z "$SPK_WAV_DIR" ]; then
        echo "WARNING: Speaker $SPEAKER not found in $L2_ROOT, skipping"
        continue
    fi

    OUT_DIR="$FT_DIR/$ACCENT"
    mkdir -p "$OUT_DIR"
    WAV_COUNT=$(find "$SPK_WAV_DIR" -name "*.wav" | wc -l)
    echo "  $SPEAKER -> $ACCENT ($WAV_COUNT WAVs)"
    find "$SPK_WAV_DIR" -name "*.wav" -exec cp {} "$OUT_DIR/" \;
done

echo ""
echo "=== Dataset ready ==="
for ACCENT_DIR in "$FT_DIR"/*/; do
    ACCENT=$(basename "$ACCENT_DIR")
    COUNT=$(find "$ACCENT_DIR" -name "*.wav" | wc -l)
    echo "  $ACCENT: $COUNT WAVs -> $ACCENT_DIR"
done

echo ""
echo "Next: python scripts/finetune_style.py --accent <accent_key>"
echo "   or: python scripts/finetune_style.py --data-dir data/finetune/<accent_key>"
