"""End-to-end CLI driver: wav in → accent-converted wav + metrics json out.

Architecture A pipeline:
    decode → VAD → per-seg[ASR + prosody → SeedVC/Vevo → F0-score → F0/energy-correct]
    → overlap-add → loudness-norm → WAV + metrics.

Emotion preservation is handled via direct F0/energy restoration (EmotionCorrector),
not via SER V/A/D trajectory which saturates on near-zero logits.
Quality is measured by F0 Pearson r (prosody preservation proxy) + proxy MOS + WER.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import click
import numpy as np
import soundfile as sf
import tempfile
import yaml

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor, load_audio
from pipeline.feature_extractor import FeatureExtractor
from pipeline.converter import SeedVCBackend, VevoBackend, ConverterBackend
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor
from pipeline.types import EmotionVec, Segment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_pipeline")
ROOT = Path(__file__).resolve().parent

_NEUTRAL = EmotionVec(0.5, 0.5, 0.5)


_EMOTIONS = ("angry", "happy", "sad", "neutral")


def emotion_to_label(vec: EmotionVec) -> str:
    """Map a V/A/D emotion vector (sigmoid [0,1], neutral≈0.5) to an emotion label.

    Valence-arousal quadrants (the dimensions SER discriminates best):
      high arousal + low  valence → angry
      high arousal + high valence → happy
      low  arousal + low  valence → sad
      otherwise                   → neutral
    Used to pick a target-accent reference clip carrying the SOURCE emotion, so
    convert_style=true transfers accent + matching emotional tone (no resynthesis).
    """
    a, v = vec.arousal, vec.valence
    hi_a, lo_a = a > 0.55, a < 0.45
    hi_v, lo_v = v > 0.55, v < 0.45
    if hi_a and lo_v:
        return "angry"
    if hi_a and hi_v:
        return "happy"
    if lo_a and lo_v:
        return "sad"
    return "neutral"


def build_reference(cfg: dict, accent: str, emotion: str | None = None) -> tuple[str, bool]:
    """Pick the target-accent reference WAV matching the source emotion.

    Bank layout (emotion-matched, preferred):
        references/<accent>/<emotion>.wav   e.g. chinese_english/angry.wav
    Selection: <emotion> → neutral → largest available clip (graceful fallback,
    so a bank with only a neutral clip still works and auto-upgrades as clips are
    added). If multiple files share an emotion prefix they are concatenated.

    Returns (path, is_temp); is_temp=True means caller must delete the temp file.
    """
    ref_dir = ROOT / cfg["paths"]["references"] / accent
    all_wavs = [p for p in ref_dir.glob("*.wav") if p.stat().st_size > 1000]
    if not all_wavs:
        raise click.BadParameter(
            f"No reference WAVs found in {ref_dir}\n"
            f"Add emotion clips named <emotion>.wav (angry/happy/sad/neutral), 5-15s each,\n"
            f"of a native {accent} speaker. Minimum: neutral.wav."
        )

    # Emotion-matched selection with fallback chain.
    def by_prefix(label: str) -> list:
        return sorted([p for p in all_wavs if p.stem.lower().startswith(label)],
                      key=lambda p: p.stat().st_size, reverse=True)

    wavs, chosen = [], None
    if emotion:
        wavs = by_prefix(emotion)
        if wavs:
            chosen = emotion
    if not wavs:
        wavs = by_prefix("neutral")
        if wavs:
            chosen = "neutral"
    if not wavs:  # no emotion-named clips at all → legacy flat bank: largest clip
        wavs = sorted(all_wavs, key=lambda p: p.stat().st_size, reverse=True)
        chosen = "fallback"
    log.info("reference for %s: emotion=%s → %s (%d clip(s))",
             accent, emotion or "n/a", chosen, len(wavs))

    # Single clip: return direct path, no temp file needed
    if len(wavs) == 1:
        log.info("  → %s", wavs[0].name)
        return str(wavs[0]), False

    sr = cfg["audio"]["sample_rate"]
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    parts = []
    for p in wavs:
        clip, clip_sr = sf.read(str(p), dtype="float32")
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        if clip_sr != sr:
            import librosa
            clip = librosa.resample(clip, orig_sr=clip_sr, target_sr=sr)
        parts.append(clip)
        parts.append(silence)

    combined = np.concatenate(parts[:-1])  # drop trailing silence
    total_s = len(combined) / sr
    log.info("reference for %s: %d clips concatenated → %.1fs", accent, len(wavs), total_s)

    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, combined, sr)
    f.close()
    return f.name, True


def build_backends(cfg: dict, mm: ModelManager):
    backends: list[ConverterBackend] = [SeedVCBackend(cfg, mm.device)]
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))
    return backends


@click.command()
@click.option("--input",         "input_path",    required=True)
@click.option("--target-accent", required=True)
@click.option("--output",        "output_path",   required=True)
@click.option("--metrics-out",   default=None)
@click.option("--reference-text", default=None,
              help="Ground-truth transcript of the input audio. When provided, "
                   "WER compares this against the re-transcribed output instead "
                   "of running Whisper twice on each segment.")
@click.option("--config", "config_path",
              default=str(ROOT / "configs" / "pipeline_config.yaml"))
def main(input_path, target_accent, output_path, metrics_out, reference_text, config_path):
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    valid = {a["key"] for a in cfg["accents"]}
    if target_accent not in valid:
        raise click.BadParameter(f"target_accent must be one of {sorted(valid)}")

    mm  = ModelManager.get(config_path)
    pre = Preprocessor(cfg)
    fe  = FeatureExtractor(mm)
    qs  = QualitySelector(cfg, fe, mm)
    ec  = EmotionCorrector(cfg)
    post = Postprocessor(cfg)
    backends = build_backends(cfg, mm)

    audio    = load_audio(input_path, sr=cfg["audio"]["sample_rate"])
    segments = pre.segment(audio)

    # Detect global source emotion → pick the target-accent reference carrying that
    # same emotion, so convert_style=true transfers accent + matching emotional tone.
    emo_probe = Segment(audio=audio[: 15 * cfg["audio"]["sample_rate"]],
                        start_s=0.0, end_s=0.0, sr=cfg["audio"]["sample_rate"])
    src_emotion = fe.emotion(emo_probe)
    emo_label = emotion_to_label(src_emotion)
    log.info("source emotion: A=%.2f V=%.2f D=%.2f → %s",
             src_emotion.arousal, src_emotion.valence, src_emotion.dominance, emo_label)
    ref, ref_is_temp = build_reference(cfg, target_accent, emotion=emo_label)

    ref_text_override = reference_text.strip() if reference_text else None

    out_wavs, seg_metrics = [], []
    try:
        for i, seg in enumerate(segments):
            # Source transcript: ground-truth if provided, else Whisper on source audio
            if ref_text_override:
                text = ref_text_override
            else:
                text, _ = fe.transcribe(seg)

            # Source prosody: F0 contour + energy envelope — used for correction and scoring
            prosody = fe.prosody(seg, n_words=len(text.split()))

            # Convert with all configured backends
            candidates = [b.convert(seg, ref) for b in backends]

            # Score by F0 correlation (prosody/emotion proxy), WER, proxy MOS
            result = qs.select(candidates, seg, prosody, text)
            if result is None:
                log.warning("Segment %d produced no candidates; skipping.", i)
                continue
            cand, score, meta = result

            # Parselmouth OLA emotion correction: F0 contour transfer (source→output)
            # + per-frame energy scaling. No WORLD vocoder — OLA preserves voice quality.
            # Fires when f0_corr < threshold; energy-only when Seed-VC already preserved F0.
            corrected = ec.correct(
                cand.wav, cand.sr,
                _NEUTRAL, _NEUTRAL,
                prosody,
                f0_corr=meta["f0_corr"],
            )

            out_wavs.append(corrected)
            seg_metrics.append({
                "segment":        i,
                "chosen_backend": cand.name,
                "score":          score,
                "f0_corr":        meta["f0_corr"],
                "wer":            meta["wer"],
                "mos":            meta["mos"],
                "transcript":     meta["transcript"],
            })
    finally:
        if ref_is_temp:
            Path(ref).unlink(missing_ok=True)

    final = post.assemble(out_wavs)
    final = post.loudness_normalize(final)
    sf.write(output_path, final, cfg["audio"]["sample_rate"], subtype="PCM_16")

    def mean(key, default):
        vals = [m[key] for m in seg_metrics if isinstance(m.get(key), (int, float))]
        return float(np.mean(vals)) if vals else default

    metrics = {
        "processing_time_ms": int((time.time() - t0) * 1000),
        "n_segments":         len(segments),
        "f0_correlation":     mean("f0_corr", 0.0),
        "wer":                mean("wer",     1.0),
        "mos_estimate":       mean("mos",     0.0),
        "segments":           seg_metrics,
    }
    if metrics_out:
        Path(metrics_out).write_text(json.dumps(metrics, indent=2))

    log.info("done in %dms → %s", metrics["processing_time_ms"], output_path)
    click.echo(json.dumps(
        {k: metrics[k] for k in
         ("processing_time_ms", "n_segments", "f0_correlation", "wer", "mos_estimate")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
