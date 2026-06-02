"""Shared pytest fixtures. Full test cases land in the dedicated test phase."""
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def sine_wav():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


@pytest.fixture
def silence_wav():
    sr = 16000
    return np.zeros(sr, dtype=np.float32), sr


def weights_present() -> bool:
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints"
    return ckpt.exists() and any(ckpt.iterdir())


requires_weights = pytest.mark.skipif(
    not weights_present(), reason="model weights not downloaded")
