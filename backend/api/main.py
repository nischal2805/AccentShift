"""AccentShift FastAPI application entry point.

Startup sequence:
  1. Parse pipeline config.
  2. Load all models via ModelManager singleton (heavy; happens once).
  3. Build pipeline components (Preprocessor, FeatureExtractor, etc.).
  4. Build conversion backends (SeedVCBackend always; VevoBackend if enabled).
  5. Store everything in app.state so route handlers can share them.

Run (dev):
    cd backend
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Run (production):
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
    (Only 1 worker — GPU models are not fork-safe.)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Corporate proxy / self-signed CA fix — must happen before any HF/torch download.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.model_manager import ModelManager
from pipeline.preprocessor import Preprocessor
from pipeline.feature_extractor import FeatureExtractor
from pipeline.quality_selector import QualitySelector
from pipeline.emotion_corrector import EmotionCorrector
from pipeline.postprocessor import Postprocessor
from pipeline.converter import SeedVCBackend, VevoBackend, ConverterBackend
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("main")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(ROOT / "configs" / "pipeline_vevo.yaml")  # Vevo backend (stronger accent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    log.info("AccentShift — loading models (this may take 30–120 s on first run)…")
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())
    app.state.cfg = cfg
    app.state.models_loaded = False

    mm = ModelManager.get(CONFIG_PATH)
    app.state.mm = mm

    pre = Preprocessor(cfg)
    fe = FeatureExtractor(mm)
    qs = QualitySelector(cfg, fe, mm)
    ec = EmotionCorrector(cfg)
    post = Postprocessor(cfg)

    backends: list[ConverterBackend] = []
    if cfg["seed_vc"].get("enabled", True):
        backends.append(SeedVCBackend(cfg, mm.device))
    if cfg["vevo"]["enabled"]:
        backends.append(VevoBackend(cfg, mm.vevo_device))

    app.state.pre = pre
    app.state.fe = fe
    app.state.qs = qs
    app.state.ec = ec
    app.state.post = post
    app.state.backends = backends
    app.state.models_loaded = True

    log.info("All models loaded. AccentShift is ready.")
    yield
    # Cleanup (Python GC handles torch tensors; nothing explicit needed here)
    log.info("AccentShift shutting down.")


app = FastAPI(
    title="AccentShift",
    description="Emotion-preserving accent conversion API (DP6)",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the Next.js dev server (localhost:3000) and any same-host production
# origin.  Restrict in production by tightening allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
