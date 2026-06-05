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
import threading
from pathlib import Path
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_style")


def _tg_send(token: str, chat_id: str, text: str) -> None:
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


def _tg_watcher(token: str, chat_id: str, runs_dir: Path,
                run_name: str, stop: threading.Event, interval: int = 600) -> None:
    seen: set = set()
    ping = 0
    while not stop.wait(interval):
        ckpts = set(runs_dir.glob(f"{run_name}/*.pth"))
        new = ckpts - seen
        if new:
            names = ", ".join(p.name for p in sorted(new))
            _tg_send(token, chat_id, f"AccentShift checkpoint saved:\n{names}")
            seen.update(new)
        else:
            ping += 1
            _tg_send(token, chat_id, f"AccentShift training running... (heartbeat #{ping}, no new checkpoint yet)")

ROOT = Path(__file__).resolve().parents[1]
SEED_VC = ROOT / "third_party" / "seed-vc"

# Seed-VC V2 training config (Hydra/OmegaConf, relative to seed-vc repo root)
SEED_VC_TRAIN_CONFIG = "configs/v2/vc_wrapper.yaml"

# Accent -> L2-Arctic speaker mapping (verified against L2-Arctic v5.0 README)
ACCENT_SPEAKERS = {
    "indian_english":     ["ASI", "RRBI", "SVBI", "TNI"],    # Hindi L1
    "chinese_english":    ["BWC", "LXC", "NCC", "TXHC"],     # Mandarin L1
    "korean_english":     ["HJK", "HKK", "YDCK", "YKWK"],   # Korean L1
    "vietnamese_english": ["HQTV", "PNV", "THV", "TLV"],    # Vietnamese L1
    "spanish_english":    ["EBVS", "ERMS", "MBMPS", "NJS"], # Spanish L1
    "arabic_english":     ["ABA", "SKA", "YBAA", "ZHAA"],   # Arabic L1
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
@click.option("--telegram-token", default=lambda: os.environ.get("TELEGRAM_TOKEN", ""),
              help="Telegram bot token (or set TELEGRAM_TOKEN env var)")
@click.option("--telegram-chat-id", default=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""),
              help="Telegram chat/user ID (or set TELEGRAM_CHAT_ID env var)")
@click.option("--notify-every", default=600, show_default=True,
              help="Telegram heartbeat interval in seconds (default 10 min)")
def main(accent, data_dir, run_name, steps, batch_size, save_every, num_workers,
         mixed_precision, train_ar, pretrained_cfm, pretrained_ar,
         telegram_token, telegram_chat_id, notify_every):
    _check_prereqs()

    # In multi-GPU (accelerate launch), LOCAL_RANK is set by the launcher.
    # Only rank 0 sends Telegram and manages checkpoints to avoid duplicate messages.
    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0

    tg = bool(telegram_token and telegram_chat_id and is_main)
    if is_main:
        if tg:
            log.info("Telegram notifications enabled (chat_id=%s)", telegram_chat_id)
        else:
            log.info("Telegram not configured — set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID or pass flags")

    ft_root = ROOT / "data" / "finetune"

    if data_dir is not None:
        data_path = Path(data_dir)
    elif accent is None:
        raise SystemExit(
            "Provide --accent <accent_key> or --data-dir <path>.\n"
            "Use --accent all to merge all accents under data/finetune/."
        )
    elif accent == "all":
        import shutil, subprocess as _sp, time
        merged = ROOT / "data" / "finetune_all"
        sentinel = ROOT / "data" / "finetune_all.ready"
        if is_main:
            # Only rank 0 sets up the merged dir; other ranks wait for sentinel
            sentinel.unlink(missing_ok=True)
            if merged.exists():
                _sp.run(["rm", "-rf", str(merged)], check=True)
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
            sentinel.touch()
        else:
            log.info("Rank %d waiting for rank 0 to stage WAVs...", int(os.environ.get("LOCAL_RANK", "1")))
            while not sentinel.exists():
                time.sleep(0.5)
            log.info("Rank %d: WAV staging done, proceeding", int(os.environ.get("LOCAL_RANK", "1")))
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
    log.info("Found %d WAV files in %s", len(wavs), data_path)

    if run_name is None:
        run_name = f"{accent}_ft"

    log.info("=" * 60)
    log.info("run_name   : %s", run_name)
    log.info("steps      : %d", steps)
    log.info("batch_size : %d", batch_size)
    log.info("precision  : %s", mixed_precision)
    log.info("train_ar   : %s", train_ar)
    log.info("save_every : %d", save_every)
    log.info("output dir : runs/%s/", run_name)
    log.info("=" * 60)

    if tg:
        _tg_send(telegram_token, telegram_chat_id,
                 f"AccentShift training STARTED\n"
                 f"run: {run_name}\nsteps: {steps} | batch: {batch_size} | "
                 f"precision: {mixed_precision} | WAVs: {len(wavs)}")

    _inject_seed_vc()
    log.info("Importing Seed-VC train_v2 (first run downloads HuBERT/CAMPPlus ~2GB)...")
    from train_v2 import Trainer  # type: ignore[import]
    log.info("Seed-VC Trainer imported OK")

    stop_event = threading.Event()
    if tg:
        watcher = threading.Thread(
            target=_tg_watcher,
            args=(telegram_token, telegram_chat_id, ROOT / "runs", run_name,
                  stop_event, notify_every),
            daemon=True,
        )
        watcher.start()
        log.info("Telegram watcher started (heartbeat every %ds)", notify_every)

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

    try:
        log.info("Training started...")
        trainer.train()
        ckpts = list((ROOT / "runs" / run_name).glob("*.pth"))
        ckpt_names = [p.name for p in ckpts]
        log.info("Training complete. Checkpoints: %s", ckpt_names)
        if tg:
            _tg_send(telegram_token, telegram_chat_id,
                     f"AccentShift training DONE\nrun: {run_name}\n"
                     f"Checkpoints: {', '.join(ckpt_names) or 'none found'}")
    except Exception as exc:
        log.exception("Training crashed: %s", exc)
        if tg:
            _tg_send(telegram_token, telegram_chat_id,
                     f"AccentShift training CRASHED\nrun: {run_name}\nError: {exc}")
        raise
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
