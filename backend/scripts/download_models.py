"""Pull all pretrained weights + clone VC repos. Idempotent. Run once after env setup.

Downloads (into checkpoints/): Whisper large-v3, audeering SER, BigVGAN-v2, ECAPA.
Clones (into third_party/): Seed-VC, Amphion (skipped if already present).
Prefetches UTMOS via torch.hub and Vevo checkpoints via snapshot_download.

Seed-VC V2 checkpoints (Plachta/Seed-VC) auto-download on first inference via its
hydra wrapper, so they are not fetched here.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
import yaml

# Corporate/proxy networks inject a self-signed root CA that certifi doesn't trust.
# truststore makes Python's ssl use the OS (Windows) cert store, which DOES trust it.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_models")

# Variant weight files we never use — skip to save bandwidth (we use PyTorch/safetensors).
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*.tflite", "*onnx*"]

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "pipeline_config.yaml").read_text())

HF_MODELS = [CFG["models"]["whisper"], CFG["models"]["ser"],
             CFG["models"]["bigvgan"], CFG["models"]["ecapa"]]
REPOS = {
    "seed-vc": "https://github.com/Plachtaa/seed-vc.git",
    "Amphion": "https://github.com/open-mmlab/Amphion.git",
}


def fetch_hf(model_id: str, dest_root: Path) -> None:
    dest = dest_root / model_id.replace("/", "__")
    if dest.exists() and any(dest.iterdir()):
        log.info("skip (present): %s", model_id)
        return
    log.info("downloading %s", model_id)
    snapshot_download(repo_id=model_id, local_dir=str(dest), ignore_patterns=IGNORE)


def clone_repo(name: str, url: str, dest_root: Path) -> None:
    dest = dest_root / name
    if dest.exists():
        log.info("skip clone (present): %s", name)
        return
    log.info("cloning %s", url)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--single-branch", url, str(dest)], check=True)


def prefetch_vevo() -> None:
    """Pull Vevo tokenizers / AR / FM / vocoder checkpoints from HF."""
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
    for m in HF_MODELS:
        fetch_hf(m, ckpt)
    for name, url in REPOS.items():
        clone_repo(name, url, tp)
    prefetch_vevo()
    prefetch_utmos()
    log.info("done. Next: follow SETUP.md to install third_party repo requirements.")


if __name__ == "__main__":
    main()
