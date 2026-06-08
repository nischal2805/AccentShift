"""FastAPI route handlers: /convert, /accents, /evaluate, /health.

All heavy pipeline objects live in app.state — initialised once at startup
in main.py and shared across all requests.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi import Request as FastAPIRequest

from pipeline.preprocessor import Preprocessor
from pipeline.feature_extractor import FeatureExtractor
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor
from pipeline.converter import ConverterBackend
from .schemas import (
    AccentInfo, AccentsResponse,
    ConversionMetrics, ConversionResponse,
    EvaluationRequest, EvaluationResponse,
    HealthResponse,
)

log = logging.getLogger("routes")
router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wav_to_b64(audio: np.ndarray, sr: int) -> str:
    """Encode float32 ndarray → 16-bit WAV → base64 string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_wav(b64: str, sr: int) -> np.ndarray:
    """Decode base64 WAV string → float32 ndarray at `sr`."""
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    audio, file_sr = sf.read(buf, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32)


def _decode_upload(data: bytes, sr: int) -> np.ndarray:
    """Decode uploaded audio bytes (any format librosa supports) to float32 mono at `sr`."""
    import librosa
    buf = io.BytesIO(data)
    audio, _ = librosa.load(buf, sr=sr, mono=True)
    return librosa.util.normalize(audio).astype(np.float32)


def _pick_reference(cfg: dict, accent: str) -> str:
    ref_dir = ROOT / cfg["paths"]["references"] / accent
    wavs = sorted(ref_dir.glob("*.wav"))
    if not wavs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No reference clips found in references/{accent}/. "
                   "Please add 5-10s WAV files for this accent.",
        )
    return str(wavs[0])


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["status"])
def health(request: FastAPIRequest) -> HealthResponse:
    """Return model load status. Frontend polls this on startup."""
    st = request.app.state
    cfg = st.cfg
    return HealthResponse(
        status="ok",
        models_loaded=getattr(st, "models_loaded", False),
        device=st.mm.device if getattr(st, "models_loaded", False) else "unknown",
        accents_available=[a["key"] for a in cfg["accents"]],
    )


@router.get("/accents", response_model=AccentsResponse, tags=["meta"])
def list_accents(request: FastAPIRequest) -> AccentsResponse:
    cfg = request.app.state.cfg
    return AccentsResponse(
        accents=[AccentInfo(key=a["key"], label=a["label"]) for a in cfg["accents"]]
    )


@router.post("/convert", response_model=ConversionResponse, tags=["pipeline"])
async def convert(
    request: FastAPIRequest,
    audio: UploadFile = File(..., description="Source audio file (.wav/.mp3/.flac/.m4a, max 50 MB)"),
    target_accent: str = Form(..., description="Target accent key from /accents"),
) -> ConversionResponse:
    """Run the full accent-conversion pipeline on the uploaded audio.

    Returns the converted WAV as base64 plus per-run quality metrics.
    """
    st = request.app.state
    if not getattr(st, "models_loaded", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Models still loading — retry in a few seconds.")

    cfg = st.cfg
    valid_accents = {a["key"] for a in cfg["accents"]}
    if target_accent not in valid_accents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"target_accent must be one of {sorted(valid_accents)}",
        )

    # File size guard (50 MB)
    MAX_BYTES = 50 * 1024 * 1024
    raw = await audio.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="Audio file exceeds 50 MB limit.")

    t0 = time.time()
    sr = cfg["audio"]["sample_rate"]

    try:
        src_audio = _decode_upload(raw, sr)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Could not decode audio: {exc}") from exc

    ref_path = _pick_reference(cfg, target_accent)

    pre: Preprocessor = st.pre
    fe: FeatureExtractor = st.fe
    qs: QualitySelector = st.qs
    ec: EmotionCorrector = st.ec
    post: Postprocessor = st.post
    backends: list[ConverterBackend] = st.backends

    from pipeline.types import Segment, EmotionVec
    NEUTRAL = EmotionVec(0.5, 0.5, 0.5)

    segments = pre.segment(src_audio)

    out_wavs: list[np.ndarray] = []
    seg_metrics: list[dict] = []

    for seg in segments:
        text, _ = fe.transcribe(seg)
        src_emotion = fe.emotion(seg)
        prosody = fe.prosody(seg, n_words=len(text.split()))
        candidates = [b.convert(seg, ref_path) for b in backends]
        result = qs.select(candidates, seg, prosody, text)  # select scores on prosody (F0), not emotion
        if result is None:
            log.warning("Segment produced no candidates; skipping.")
            continue
        cand, score, meta = result
        # convert_style=false preserves source prosody; energy EC restores intensity (method=energy)
        corrected = ec.correct(cand.wav, cand.sr, NEUTRAL, NEUTRAL, prosody,
                               f0_corr=meta["f0_corr"])
        out_wavs.append(corrected)

        # Measure output emotion on the corrected audio → emotion-preservation metric
        out_seg = Segment(audio=corrected, start_s=0.0,
                          end_s=len(corrected) / cand.sr, sr=cand.sr)
        out_emotion = fe.emotion(out_seg)
        sa, oa = src_emotion.to_array(), out_emotion.to_array()
        emo_sim = float(np.dot(sa, oa) / (np.linalg.norm(sa) * np.linalg.norm(oa) + 1e-9))
        seg_metrics.append({
            "chosen_backend": cand.name,
            "score": score,
            "emotion_sim": max(0.0, min(1.0, emo_sim)),
            "wer": meta["wer"],
            "mos": meta["mos"],
            "valence_source": src_emotion.valence,
            "valence_output": out_emotion.valence,
            "arousal_source": src_emotion.arousal,
            "arousal_output": out_emotion.arousal,
            "dominance_source": src_emotion.dominance,
            "dominance_output": out_emotion.dominance,
        })

    final = post.assemble(out_wavs)
    final = post.loudness_normalize(final)
    processing_ms = int((time.time() - t0) * 1000)

    def mean(key: str, default: float) -> float:
        vals = [m[key] for m in seg_metrics]
        return float(np.mean(vals)) if vals else default

    metrics = ConversionMetrics(
        emotion_similarity=mean("emotion_sim", 0.0),
        wer=mean("wer", 1.0),
        mos_estimate=mean("mos", 3.5),
        valence_source=mean("valence_source", 0.0),
        valence_output=mean("valence_output", 0.0),
        arousal_source=mean("arousal_source", 0.0),
        arousal_output=mean("arousal_output", 0.0),
        dominance_source=mean("dominance_source", 0.0),
        dominance_output=mean("dominance_output", 0.0),
        n_segments=len(segments),
        processing_time_ms=processing_ms,
        chosen_backends=[m["chosen_backend"] for m in seg_metrics],
    )

    return ConversionResponse(audio_b64=_wav_to_b64(final, sr), metrics=metrics)


@router.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
def evaluate(request: FastAPIRequest, body: EvaluationRequest) -> EvaluationResponse:
    """Compute quality metrics between a source and an already-converted audio clip.

    Useful for offline / batch evaluation without rerunning the conversion.
    """
    st = request.app.state
    if not getattr(st, "models_loaded", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Models still loading.")

    cfg = st.cfg
    sr = body.sample_rate
    fe: FeatureExtractor = st.fe
    mm = st.mm

    src = _b64_to_wav(body.source_audio_b64, sr)
    out = _b64_to_wav(body.converted_audio_b64, sr)

    from pipeline.types import Segment
    from evaluation import metrics as M

    src_seg = Segment(audio=src, start_s=0.0, end_s=len(src) / sr, sr=sr)
    out_seg = Segment(audio=out, start_s=0.0, end_s=len(out) / sr, sr=sr)

    src_emo = fe.emotion(src_seg)
    out_emo = fe.emotion(out_seg)
    emo_sim = float(np.dot(src_emo.to_array(), out_emo.to_array()) /
                    (np.linalg.norm(src_emo.to_array()) * np.linalg.norm(out_emo.to_array()) + 1e-9))

    try:
        mos = M.utmos(mm, out, sr)
    except Exception:
        mos = 3.5

    spk_cos = M.speaker_agnosticism(mm, src, out, sr)

    return EvaluationResponse(
        emotion_similarity=emo_sim,
        mos_estimate=mos,
        speaker_cosine=spk_cos,
    )
