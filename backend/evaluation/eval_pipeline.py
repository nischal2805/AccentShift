"""Batch evaluation over a test split. Runs the full pipeline and reports metrics vs targets.

Covers emotion similarity, WER, UTMOS, and speaker agnosticism. Accent accuracy is reported
separately via accent_classifier once a classifier checkpoint exists (see --accent-clf).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import click
import numpy as np
import yaml

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor, load_audio
from pipeline.feature_extractor import FeatureExtractor
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor
from pipeline.converter import SeedVCBackend, VevoBackend, ConverterBackend
from evaluation import metrics as M
from evaluation.accent_classifier import AccentClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_pipeline")
ROOT = Path(__file__).resolve().parents[1]

TARGETS = {"accent_acc": 0.70, "emotion_sim": 0.85, "wer_rel_increase": 0.10,
           "utmos": 3.8, "speaker_cos": 0.50}


@click.command()
@click.option("--test-dir", required=True, help="dir of source WAVs")
@click.option("--target-accent", required=True)
@click.option("--out-json", default="eval_results.json")
@click.option("--accent-clf", default=None, help="path to trained accent classifier (joblib)")
@click.option("--config", "config_path", default=str(ROOT / "configs" / "pipeline_config.yaml"))
def main(test_dir, target_accent, out_json, accent_clf, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    mm = ModelManager.get(config_path)
    pre, fe = Preprocessor(cfg), FeatureExtractor(mm)
    qs, ec, post = QualitySelector(cfg, fe, mm), EmotionCorrector(cfg), Postprocessor(cfg)
    backends: list[ConverterBackend] = [SeedVCBackend(cfg, mm.device)]
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))
    ref_dir = ROOT / cfg["paths"]["references"] / target_accent
    ref_wavs = sorted(ref_dir.glob("*.wav"))
    if not ref_wavs:
        raise FileNotFoundError(f"No reference clips in {ref_dir}")
    ref = str(ref_wavs[0])

    clf = None
    if accent_clf:
        clf = AccentClassifier(mm)
        clf.load(accent_clf)

    sr = cfg["audio"]["sample_rate"]
    rows = []
    for wav_path in sorted(Path(test_dir).glob("*.wav")):
        audio = load_audio(str(wav_path), sr=sr)
        segs = pre.segment(audio)
        outs, emo_sims, src_texts, out_texts = [], [], [], []
        for seg in segs:
            text, _ = fe.transcribe(seg)
            src_emotion = fe.emotion(seg)
            prosody = fe.prosody(seg, n_words=len(text.split()))
            cands = [b.convert(seg, ref) for b in backends]
            result = qs.select(cands, seg, src_emotion, text)
            if result is None:
                log.warning("Segment produced no candidates; skipping.")
                continue
            cand, _score, meta = result
            corrected = ec.correct(cand.wav, cand.sr, src_emotion, meta["emotion_output"], prosody)
            outs.append(corrected)
            emo_sims.append(meta["emotion_sim"])
            src_texts.append(text)
            out_texts.append(meta["transcript"])
        final = post.loudness_normalize(post.assemble(outs))
        row = {
            "file": wav_path.name,
            "emotion_sim": float(np.mean(emo_sims)) if emo_sims else 0.0,
            "wer": M.word_error_rate(" ".join(src_texts), " ".join(out_texts)),
            "utmos": M.utmos(mm, final, sr),
            "speaker_cos": M.speaker_agnosticism(mm, audio, final, sr),
        }
        if clf is not None:
            row["accent_pred"] = clf.predict(final)
            row["accent_correct"] = int(row["accent_pred"] == target_accent)
        rows.append(row)
        log.info("%s: emo=%.3f wer=%.3f utmos=%.2f spk=%.3f",
                 wav_path.name, row["emotion_sim"], row["wer"], row["utmos"], row["speaker_cos"])

    def mean(key, default):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else default

    summary = {
        "n": len(rows),
        "emotion_sim_mean": mean("emotion_sim", 0.0),
        "wer_mean": mean("wer", 1.0),
        "utmos_mean": mean("utmos", 0.0),
        "speaker_cos_mean": mean("speaker_cos", 1.0),
        "targets": TARGETS,
        "rows": rows,
    }
    if clf is not None:
        summary["accent_acc"] = mean("accent_correct", 0.0)
    Path(out_json).write_text(json.dumps(summary, indent=2))
    printable = {k: summary[k] for k in
                 ("n", "emotion_sim_mean", "wer_mean", "utmos_mean", "speaker_cos_mean")}
    if "accent_acc" in summary:
        printable["accent_acc"] = summary["accent_acc"]
    click.echo(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
