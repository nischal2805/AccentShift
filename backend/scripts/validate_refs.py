"""Validate reference accent clips before a pipeline run. CPU-only, no models.

`run_pipeline.pick_reference()` grabs the first WAV in references/<accent>/ — a bad clip
(wrong length, stereo, silent, clipped, mislabeled) silently degrades every conversion for
that accent. This script checks each accent folder and reports problems up front.

Checks per clip:
  - readable WAV
  - mono (warn if multi-channel — pipeline downmixes but the ref should already be mono)
  - duration in [min_s, max_s] (Seed-VC wants a 5-10s reference)
  - not effectively silent (peak above floor)
  - not hard-clipped (fraction of |samples| >= 0.999 below limit)

Exit code 0 = all configured accents have at least one OK clip. 1 = something failed.

    uv run python scripts/validate_refs.py
    uv run python scripts/validate_refs.py --min-s 5 --max-s 10
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import click
import numpy as np
import soundfile as sf
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("validate_refs")
ROOT = Path(__file__).resolve().parents[1]


def check_clip(path: Path, min_s: float, max_s: float,
               silence_floor: float, clip_frac_max: float) -> list[str]:
    """Return list of problem strings for one clip (empty = OK)."""
    problems: list[str] = []
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        return [f"unreadable ({e})"]
    n_channels = audio.shape[1]
    if n_channels != 1:
        problems.append(f"{n_channels} channels (want mono)")
    mono = audio.mean(axis=1)
    dur = len(mono) / sr if sr else 0.0
    if dur < min_s:
        problems.append(f"too short {dur:.1f}s (< {min_s}s)")
    elif dur > max_s:
        problems.append(f"too long {dur:.1f}s (> {max_s}s)")
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    if peak < silence_floor:
        problems.append(f"near-silent (peak {peak:.4f})")
    if mono.size:
        clip_frac = float(np.mean(np.abs(mono) >= 0.999))
        if clip_frac > clip_frac_max:
            problems.append(f"clipped ({clip_frac:.1%} of samples)")
    return problems


@click.command()
@click.option("--config", "config_path", default=str(ROOT / "configs" / "pipeline_config.yaml"))
@click.option("--min-s", default=3.0, type=float, help="min acceptable duration")
@click.option("--max-s", default=15.0, type=float, help="max acceptable duration")
@click.option("--silence-floor", default=1e-3, type=float)
@click.option("--clip-frac-max", default=0.01, type=float)
def main(config_path, min_s, max_s, silence_floor, clip_frac_max):
    cfg = yaml.safe_load(Path(config_path).read_text())
    ref_root = ROOT / cfg["paths"]["references"]
    accents = [a["key"] for a in cfg["accents"]]

    failed = False
    for accent in accents:
        folder = ref_root / accent
        wavs = sorted(folder.glob("*.wav")) if folder.exists() else []
        if not wavs:
            log.error("%s: NO clips (add a 5-10s native-accent WAV)", accent)
            failed = True
            continue
        ok_clips = 0
        for w in wavs:
            problems = check_clip(w, min_s, max_s, silence_floor, clip_frac_max)
            if problems:
                log.warning("%s/%s: %s", accent, w.name, "; ".join(problems))
            else:
                ok_clips += 1
        if ok_clips == 0:
            log.error("%s: %d clip(s) present but NONE pass — first clip is used by default",
                      accent, len(wavs))
            failed = True
        else:
            log.info("%s: OK (%d/%d clips pass)", accent, ok_clips, len(wavs))

    if failed:
        log.error("validation FAILED — fix clips above before running the pipeline")
        sys.exit(1)
    log.info("all accents have at least one valid reference clip")


if __name__ == "__main__":
    main()
