"""Pull all pretrained weights + clone VC repos. Idempotent. Run once after env setup.

Strategy:
  - Large models (Whisper): stored in HF_HOME cache only — no local copy (too big to duplicate).
    from_pretrained() finds them via HF_HOME when local checkpoint dir is absent.
  - Small models (SER, ECAPA): copied to checkpoints/ for explicit local control.
    BigVGAN is NOT downloaded here — Seed-VC bundles its own BigVGAN internally.
  - Repos (Seed-VC, Amphion): cloned into third_party/ (skipped if present).
  - Vevo + UTMOS: prefetched into their respective cache dirs.

Run from backend/:
    $env:HF_HOME = "D:\\AccentShift\\backend\\.hf_cache"
    uv run python scripts/download_models.py
"""
from __future__ import annotations
import logging
import os
import subprocess
from pathlib import Path
import yaml

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_models")

# Skip non-PyTorch variants and redundant fp32 bins (safetensors preferred).
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.tflite", "*onnx*", "*.bin",
          "flax_model*", "tf_model*", "rust_model*"]

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "pipeline_config.yaml").read_text())

# Whisper: large model — kept in HF cache only (no local_dir copy).
CACHE_ONLY_MODELS = [CFG["models"]["whisper"]]

# Smaller models: copied to checkpoints/ for explicit local path.
# BigVGAN excluded — Seed-VC bundles its own vocoder internally.
LOCAL_MODELS = [CFG["models"]["ser"], CFG["models"]["ecapa"]]

REPOS = {
    "seed-vc": "https://github.com/Plachtaa/seed-vc.git",
    "Amphion": "https://github.com/open-mmlab/Amphion.git",
}


def fetch_hf_cache(model_id: str) -> None:
    """Download into HF cache only (no local copy). Used for large models like Whisper."""
    log.info("caching %s (HF cache only, no local copy)", model_id)
    snapshot_download(repo_id=model_id, ignore_patterns=IGNORE)
    log.info("cached %s", model_id)


def fetch_hf_local(model_id: str, dest_root: Path) -> None:
    """Download into checkpoints/<model> — resumes partial downloads via etag."""
    dest = dest_root / model_id.replace("/", "__")
    log.info("fetching %s -> %s", model_id, dest)
    snapshot_download(repo_id=model_id, local_dir=str(dest), ignore_patterns=IGNORE)
    log.info("done %s", model_id)


def clone_repo(name: str, url: str, dest_root: Path) -> None:
    dest = dest_root / name
    if dest.exists():
        log.info("skip clone (present): %s", name)
        return
    log.info("cloning %s", url)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--single-branch", url, str(dest)], check=True)


def prefetch_vevo() -> None:
    cache = ROOT / CFG["vevo"]["cache_dir"]
    repo = CFG["vevo"]["hf_repo"]
    patterns = [
        "tokenizer/vq32/*", "tokenizer/vq8192/*",
        "contentstyle_modeling/Vq32ToVq8192/*",
        "acoustic_modeling/Vq8192ToMels/*", "acoustic_modeling/Vocoder/*",
    ]
    try:
        for p in patterns:
            snapshot_download(repo_id=repo, repo_type="model",
                              cache_dir=str(cache), allow_patterns=[p])
        log.info("Vevo checkpoints prefetched -> %s", cache)
    except Exception as e:  # noqa: BLE001
        log.warning("Vevo prefetch failed (fetch on first inference): %s", e)


def prefetch_utmos() -> None:
    try:
        import torch
        torch.hub.load("sarulab-speech/UTMOS22", "utmos22_strong", trust_repo=True)
        log.info("UTMOS prefetched")
    except Exception as e:  # noqa: BLE001
        log.warning("UTMOS prefetch failed (fetch at first eval): %s", e)


def main() -> None:
    ckpt = ROOT / CFG["paths"]["checkpoints"]
    ckpt.mkdir(parents=True, exist_ok=True)
    tp = ROOT / "third_party"
    tp.mkdir(parents=True, exist_ok=True)

    for m in CACHE_ONLY_MODELS:
        fetch_hf_cache(m)

    for m in LOCAL_MODELS:
        fetch_hf_local(m, ckpt)

    for name, url in REPOS.items():
        clone_repo(name, url, tp)

    prefetch_vevo()
    prefetch_utmos()
    log.info("done. Next: follow SETUP.md to install third_party repo requirements.")


if __name__ == "__main__":
    main()
