"""End-to-end CLI driver: wav in -> accent-converted wav + metrics json out.

Wires the Architecture-A pipeline:
    decode -> VAD -> per-seg[ASR/SER/prosody -> SeedVC+Vevo -> score -> emotion-correct]
    -> overlap-add -> loudness-norm -> WAV + metrics.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

# Make Python's ssl trust the OS cert store (corporate/proxy self-signed root) before any
# Hugging Face / torch.hub / speechbrain download happens.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import click
import numpy as np
import soundfile as sf
import yaml

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor, load_audio
from pipeline.feature_extractor import FeatureExtractor
from pipeline.converter import SeedVCBackend, VevoBackend
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_pipeline")
ROOT = Path(__file__).resolve().parent


def pick_reference(cfg: dict, accent: str) -> str:
    ref_dir = ROOT / cfg["paths"]["references"] / accent
    wavs = sorted(ref_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No reference clips in {ref_dir}. Add 5-10s WAVs.")
    return str(wavs[0])


def build_backends(cfg: dict, mm: ModelManager):
    backends = [SeedVCBackend(cfg, mm.device)]
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))
    return backends


@click.command()
@click.option("--input", "input_path", required=True)
@click.option("--target-accent", required=True)
@click.option("--output", "output_path", required=True)
@click.option("--metrics-out", default=None)
@click.option("--config", "config_path", default=str(ROOT / "configs" / "pipeline_config.yaml"))
def main(input_path, target_accent, output_path, metrics_out, config_path):
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    valid = {a["key"] for a in cfg["accents"]}
    if target_accent not in valid:
        raise click.BadParameter(f"target_accent must be one of {sorted(valid)}")

    mm = ModelManager.get(config_path)
    pre = Preprocessor(cfg)
    fe = FeatureExtractor(mm)
    qs = QualitySelector(cfg, fe, mm)
    ec = EmotionCorrector(cfg)
    post = Postprocessor(cfg)
    backends = build_backends(cfg, mm)
    ref = pick_reference(cfg, target_accent)

    audio = load_audio(input_path, sr=cfg["audio"]["sample_rate"])
    segments = pre.segment(audio)

    out_wavs, seg_metrics = [], []
    for i, seg in enumerate(segments):
        text, _ = fe.transcribe(seg)
        src_emotion = fe.emotion(seg)
        prosody = fe.prosody(seg, n_words=len(text.split()))
        candidates = [b.convert(seg, ref) for b in backends]
        cand, score, meta = qs.select(candidates, seg, src_emotion, text)
        corrected = ec.correct(cand.wav, cand.sr, src_emotion,
                               meta["emotion_output"], prosody)
        out_wavs.append(corrected)
        eo = meta["emotion_output"]
        seg_metrics.append({
            "segment": i, "chosen_backend": cand.name, "score": score,
            "emotion_sim": meta["emotion_sim"], "wer": meta["wer"], "mos": meta["mos"],
            "valence_source": src_emotion.valence, "valence_output": eo.valence,
            "arousal_source": src_emotion.arousal, "arousal_output": eo.arousal,
            "dominance_source": src_emotion.dominance, "dominance_output": eo.dominance,
        })

    final = post.assemble(out_wavs)
    final = post.loudness_normalize(final)
    sf.write(output_path, final, cfg["audio"]["sample_rate"], subtype="PCM_16")

    def mean(key, default):
        return float(np.mean([m[key] for m in seg_metrics])) if seg_metrics else default

    metrics = {
        "processing_time_ms": int((time.time() - t0) * 1000),
        "n_segments": len(segments),
        "emotion_similarity": mean("emotion_sim", 0.0),
        "wer": mean("wer", 1.0),
        "mos_estimate": mean("mos", 0.0),
        "segments": seg_metrics,
    }
    if metrics_out:
        Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    log.info("done in %dms -> %s", metrics["processing_time_ms"], output_path)
    click.echo(json.dumps({k: metrics[k] for k in
               ("processing_time_ms", "n_segments", "emotion_similarity", "wer", "mos_estimate")},
               indent=2))


if __name__ == "__main__":
    main()
