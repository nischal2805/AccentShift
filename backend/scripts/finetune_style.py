"""Finetune Seed-VC V2 style encoder on L2-ARCTIC parallel pairs.

RUN ON A100 — DO NOT EXECUTE ON THE 8GB BOX (it cannot hold training state).
Content encoder + vocoder are frozen. Only the ~50M-param style encoder trains.

Usage (A100 box):
    python scripts/finetune_style.py --config configs/finetune_config.yaml \
        --manifest data/l2arctic_pairs/manifest.jsonl

Output checkpoint -> checkpoints/seedvc_finetuned/ ; then point `seed_vc.ar_checkpoint_path`
(and/or cfm) in configs/pipeline_config.yaml at it.

This is a GUARDED SKELETON: it loads config + manifest and lays out the training loop, but
raises before training because the exact Seed-VC trainer module imports are only knowable from
the cloned repo on the A100 box. Wire step (1) below against third_party/seed-vc/train_v2.py,
then remove the guard.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import click
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_style")
ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    log.info("manifest: %d pairs (%d train)", len(rows), len(train))
    return train


@click.command()
@click.option("--config", "config_path", default=str(ROOT / "configs" / "finetune_config.yaml"))
@click.option("--manifest", required=True)
def main(config_path, manifest):
    cfg = yaml.safe_load(Path(config_path).read_text())
    log.info("finetune config: %s", cfg)
    train_pairs = load_manifest(manifest)
    if not train_pairs:
        raise SystemExit("no training pairs in manifest")

    seed_vc = ROOT / "third_party" / "seed-vc"
    if str(seed_vc) not in sys.path:
        sys.path.insert(0, str(seed_vc))

    # ===== (1) WIRE THIS on the A100 box, against third_party/seed-vc/train_v2.py =====
    #   - Build the V2 wrapper (see inference_v2.load_v2_models) to get the style encoder.
    #   - Freeze cfg["model"]["freeze"] modules; unfreeze the style encoder.
    #   - Build a Dataset over train_pairs: source = nonnative wav, target = native wav.
    #   - optimizer: AdamW, lr cfg["optimizer"]["lr_style_encoder"] (style) /
    #       lr_ar_decoder (AR), weight_decay cfg["optimizer"]["weight_decay"].
    #   - scheduler: cosine_with_warmup (warmup_steps, total_steps from cfg).
    #   - mixed precision bf16, gradient_accumulation, gradient_checkpointing per cfg.
    #   - save to cfg["output"]["checkpoint_dir"] every N steps.
    raise SystemExit(
        "finetune_style.py is a guarded skeleton. Wire the Seed-VC trainer imports (step 1) "
        "against third_party/seed-vc/train_v2.py on the A100 box, then remove this guard. "
        "See SETUP.md section 8."
    )


if __name__ == "__main__":
    main()
