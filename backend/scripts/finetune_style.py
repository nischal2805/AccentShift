"""Fine-tune Seed-VC V2 CFM + AR on target-accent audio.

Run on L40S 48GB (DigitalOcean) or any GPU with >=20GB VRAM.

How it works:
  Seed-VC's Trainer walks a flat directory of WAV files and fine-tunes the CFM
  (flow-matching style encoder) and optionally the AR decoder.
  No paired data needed — just target-accent WAVs.

Quick start on L40S:
    # 1. Setup env (once):
    bash scripts/a100_setup.sh

    # 2. Put WAVs in data/finetune/<accent>/ (see TRAINING.md)

    # 3. Train:
    python scripts/finetune_style.py --accent indian_english

    # 4. SCP checkpoint back:
    scp -r root@<droplet-ip>:/root/AccentShift/backend/runs/indian_english_ft/ .

Output: runs/<run_name>/CFM_epoch_*_step_*.pth  (only latest kept, max_keep=1)
Epoch counter in filename reflects training loop epochs, NOT steps/1000.
After training: set seed_vc.cfm_checkpoint_path in pipeline_config.yaml to the .pth path.

NOTE: Seed-VC downloads its content extractor weights (HuBERT, CAMPPlus) from HF
on the FIRST training step — requires internet on the training box.
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_style")

ROOT = Path(__file__).resolve().parents[1]
SEED_VC = ROOT / "third_party" / "seed-vc"

# Seed-VC V2 training config (Hydra/OmegaConf, relative to seed-vc repo root)
SEED_VC_TRAIN_CONFIG = "configs/v2/vc_wrapper.yaml"

# Accent -> L2-Arctic speaker mapping (for --accent shortcut)
ACCENT_SPEAKERS = {
    "indian_english":     ["ASI", "MBMPS"],
    "chinese_english":    ["HKK", "YBAA"],
    "korean_english":     ["YDCK", "YKWK"],
    "vietnamese_english": ["HQTV", "PNV"],
    "spanish_english":    ["EBVS", "NJS"],
    "arabic_english":     ["ERMS", "RRBI"],
}


def _check_prereqs():
    if not SEED_VC.exists():
        raise SystemExit(f"Seed-VC repo not found at {SEED_VC}. Run: bash scripts/a100_setup.sh")
    config = SEED_VC / SEED_VC_TRAIN_CONFIG
    if not config.exists():
        raise SystemExit(f"Seed-VC train config missing: {config}")


def _inject_seed_vc():
    if str(SEED_VC) not in sys.path:
        sys.path.insert(0, str(SEED_VC))
    # Seed-VC reads its configs relative to its own repo root
    os.chdir(SEED_VC)


@click.command()
@click.option("--accent", default=None,
              help="Single accent key. Use 'all' to merge every accent under data/finetune/.")
@click.option("--data-dir", default=None,
              help="Explicit WAV directory (overrides --accent). Can be a merged dir.")
@click.option("--run-name", default=None,
              help="Checkpoint dir name. Saved to runs/<run-name>/.")
@click.option("--steps", default=15000, show_default=True,
              help="15k steps ~= 30-60 epochs on 2k WAVs. 80k risks overfitting 2-4 speakers.")
@click.option("--batch-size", default=8, show_default=True,
              help="8 safe for L40S 48GB with --train-ar. 16 CFM-only only.")
@click.option("--save-every", default=1000, show_default=True)
@click.option("--num-workers", default=4, show_default=True)
@click.option("--mixed-precision", default="bf16", show_default=True,
              help="bf16=A100, fp16=V100, no=debug")
@click.option("--train-ar", is_flag=True, default=False,
              help="Also fine-tune the AR decoder (more VRAM; recommended for strong accent transfer).")
@click.option("--pretrained-cfm", default=None)
@click.option("--pretrained-ar", default=None)
def main(accent, data_dir, run_name, steps, batch_size, save_every, num_workers,
         mixed_precision, train_ar, pretrained_cfm, pretrained_ar):
    _check_prereqs()

    ft_root = ROOT / "data" / "finetune"

    if data_dir is not None:
        data_path = Path(data_dir)
    elif accent is None:
        raise SystemExit(
            "Provide --accent <accent_key> or --data-dir <path>.\n"
            "Use --accent all to merge all accents under data/finetune/."
        )
    elif accent == "all":
        import shutil
        merged = ROOT / "data" / "finetune_all"
        if merged.exists():
            shutil.rmtree(merged)
        merged.mkdir(parents=True)
        total = 0
        for accent_dir in sorted(ft_root.iterdir()):
            if not accent_dir.is_dir():
                continue
            for wav in accent_dir.rglob("*.wav"):
                dst = merged / f"{accent_dir.name}__{wav.name}"
                try:
                    dst.symlink_to(wav.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(wav, dst)
                total += 1
        log.info("Staged %d WAVs (symlinks) from all accents -> %s", total, merged)
        data_path = merged
    else:
        data_path = ft_root / accent

    if not data_path.exists():
        raise SystemExit(
            f"Data dir not found: {data_path}\n"
            f"Run: bash scripts/download_l2arctic.sh {accent or ''}"
        )
    wavs = list(data_path.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAV files in {data_path}")
    log.info("%d WAV files, data_dir=%s", len(wavs), data_path)

    if run_name is None:
        run_name = f"{accent}_ft"

    log.info("run=%s steps=%d batch=%d precision=%s train_ar=%s",
             run_name, steps, batch_size, mixed_precision, train_ar)

    _inject_seed_vc()
    from train_v2 import Trainer  # type: ignore[import]

    trainer = Trainer(
        config_path=SEED_VC_TRAIN_CONFIG,
        pretrained_cfm_ckpt_path=pretrained_cfm,
        pretrained_ar_ckpt_path=pretrained_ar,
        data_dir=str(data_path),
        run_name=run_name,
        batch_size=batch_size,
        num_workers=num_workers,
        steps=steps,
        save_interval=save_every,
        train_cfm=True,
        train_ar=train_ar,
        mixed_precision=mixed_precision,
    )
    trainer.train()


if __name__ == "__main__":
    main()
