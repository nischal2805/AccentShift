"""Prepare L2-ARCTIC parallel pairs for style-encoder finetune. Run before finetune (A100 box).

Pairs each L2-ARCTIC (non-native) utterance with the matching CMU-ARCTIC (native) recording
sharing the same utterance id (filename stem, e.g. arctic_a0001). Emits a jsonl manifest with
{native, nonnative, accent, split}.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_l2arctic")


@click.command()
@click.option("--l2arctic-dir", required=True, help="root of L2-ARCTIC corpus")
@click.option("--cmu-dir", required=True, help="root of CMU-ARCTIC (native) corpus")
@click.option("--out", default="data/l2arctic_pairs/manifest.jsonl")
@click.option("--eval-frac", default=0.05, type=float)
def main(l2arctic_dir, cmu_dir, out, eval_frac):
    l2 = Path(l2arctic_dir)
    cmu = Path(cmu_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CMU native wavs keyed by utterance id (filename stem)
    native = {p.stem: p for p in cmu.rglob("*.wav")}
    pairs = []
    for spk_dir in sorted(p for p in l2.iterdir() if p.is_dir()):
        accent = spk_dir.name  # speaker id; map to accent via L2-ARCTIC README
        wav_root = spk_dir / "wav" if (spk_dir / "wav").exists() else spk_dir
        for wav in wav_root.glob("*.wav"):
            nat = native.get(wav.stem)
            if nat is not None:
                pairs.append({"native": str(nat), "nonnative": str(wav), "accent": accent})

    n_eval = int(len(pairs) * eval_frac)
    for i, p in enumerate(pairs):
        p["split"] = "eval" if i < n_eval else "train"
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    log.info("wrote %d pairs (%d eval) -> %s", len(pairs), n_eval, out_path)


if __name__ == "__main__":
    main()
