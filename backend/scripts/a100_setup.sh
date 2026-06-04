#!/usr/bin/env bash
# A100 box setup. Run once after cloning the repo.
# Tested on: Ubuntu 22.04, CUDA 12.1, Python 3.10, A100 40GB.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "=== [1/6] Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/6] Python venv (3.10) ==="
uv venv --python 3.10
source .venv/bin/activate

echo "=== [3/6] Torch cu121 (A100 default) ==="
# A100 typically ships CUDA 12.x. Adjust index URL if your CUDA differs.
uv pip install torch==2.6.0+cu121 torchaudio==2.6.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/6] Pipeline deps ==="
uv pip install \
    numpy "scipy>=1.10" "librosa>=0.10.2" soundfile pyloudnorm pyworld silero-vad \
    transformers jiwer speechbrain scikit-learn joblib pyyaml click \
    huggingface-hub hf_xet truststore tqdm hydra-core omegaconf einops munch \
    accelerate pydub tensorboard

echo "=== [5/6] Clone Seed-VC + Amphion if not present ==="
mkdir -p third_party
if [ ! -d third_party/seed-vc ]; then
    git clone --depth 1 --filter=blob:none --single-branch \
        https://github.com/Plachtaa/seed-vc.git third_party/seed-vc
fi
if [ ! -d third_party/Amphion ]; then
    git clone --depth 1 --filter=blob:none --single-branch \
        https://github.com/open-mmlab/Amphion.git third_party/Amphion
fi

echo "=== [6/6] Set HF_HOME and download models ==="
export HF_HOME="$REPO_DIR/.hf_cache"
python scripts/download_models.py

echo ""
echo "Setup complete. Next: bash scripts/download_l2arctic.sh <accent>"
echo "Then: python scripts/finetune_style.py --accent <accent>"
