"""Pydantic request/response models for AccentShift FastAPI routes."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ConversionMetrics(BaseModel):
    emotion_similarity: float = Field(..., ge=0.0, le=1.0,
                                      description="Cosine similarity of source vs output emotion vectors")
    wer: float = Field(..., ge=0.0, le=1.0, description="Word Error Rate (0=perfect, 1=total failure)")
    mos_estimate: float = Field(..., ge=1.0, le=5.0, description="UTMOS automated MOS estimate")
    # Per-dimension emotion values for visualisation
    valence_source: float
    valence_output: float
    arousal_source: float
    arousal_output: float
    dominance_source: float
    dominance_output: float
    # Metadata
    n_segments: int
    processing_time_ms: int
    chosen_backends: list[str] = Field(default_factory=list,
                                       description="Which backend (seed_vc/vevo) was chosen per segment")


class ConversionResponse(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded 16kHz PCM-16 WAV bytes")
    metrics: ConversionMetrics


class AccentInfo(BaseModel):
    key: str
    label: str


class AccentsResponse(BaseModel):
    accents: list[AccentInfo]


class EvaluationRequest(BaseModel):
    source_audio_b64: str = Field(..., description="Base64-encoded source WAV")
    converted_audio_b64: str = Field(..., description="Base64-encoded converted WAV")
    sample_rate: int = Field(default=16000)


class EvaluationResponse(BaseModel):
    emotion_similarity: float
    wer: Optional[float] = None
    mos_estimate: float
    speaker_cosine: float = Field(...,
                                  description="ECAPA cosine sim between source and output (target < 0.5)")


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool
    device: str
    accents_available: list[str]
