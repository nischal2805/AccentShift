"""Tests for ConverterBackend interfaces and Postprocessor.

Heavy model tests (SeedVCBackend / VevoBackend) are skipped unless weights are present
(see conftest.requires_weights). The structural / type tests always run.
"""
from __future__ import annotations
import numpy as np
import pytest
from pipeline.converter import ConverterBackend, SeedVCBackend, VevoBackend
from pipeline.postprocessor import Postprocessor
from pipeline.types import Segment

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

MINIMAL_CFG = {
    "audio": {"sample_rate": 16000, "max_segment_s": 30.0},
    "postprocess": {"crossfade_ms": 30, "target_lufs": -23.0},
    "seed_vc": {
        "repo_dir": "third_party/seed-vc",
        "convert_style": True, "anonymization_only": False,
        "diffusion_steps": 30, "length_adjust": 1.0,
        "intelligibility_cfg_rate": 0.7, "similarity_cfg_rate": 0.7,
        "top_p": 0.9, "temperature": 1.0, "repetition_penalty": 1.0,
        "ar_checkpoint_path": None, "cfm_checkpoint_path": None, "compile": False,
    },
    "vevo": {
        "repo_dir": "third_party/Amphion",
        "enabled": True, "device": "cpu",
        "flow_matching_steps": 32,
        "hf_repo": "amphion/Vevo", "cache_dir": "checkpoints/Vevo",
    },
}


def _sine_seg(duration: float = 1.0) -> Segment:
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return Segment(audio=wav, start_s=0.0, end_s=duration, sr=sr)


# ──────────────────────────────────────────────────────────────────────────────
# ConverterBackend ABC
# ──────────────────────────────────────────────────────────────────────────────

class TestConverterBackendABC:
    def test_is_abstract(self):
        """ConverterBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ConverterBackend()  # type: ignore[call-arg]

    def test_seedvc_is_subclass(self):
        assert issubclass(SeedVCBackend, ConverterBackend)

    def test_vevo_is_subclass(self):
        assert issubclass(VevoBackend, ConverterBackend)

    def test_backends_have_name(self):
        assert SeedVCBackend.name == "seed_vc"
        assert VevoBackend.name == "vevo"


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessor (no models needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestPostprocessor:
    def setup_method(self):
        self.post = Postprocessor(MINIMAL_CFG)

    def test_assemble_empty(self):
        out = self.post.assemble([])
        assert len(out) == 0

    def test_assemble_single(self):
        wav = np.ones(16000, dtype=np.float32) * 0.1
        out = self.post.assemble([wav])
        np.testing.assert_array_equal(out, wav)

    def test_assemble_two_segments(self):
        a = np.ones(16000, dtype=np.float32) * 0.1
        b = np.ones(16000, dtype=np.float32) * 0.2
        out = self.post.assemble([a, b])
        assert len(out) > 0
        assert out.dtype == np.float32

    def test_loudness_normalize_short_passthrough(self):
        """Clips shorter than 1 s pass through unchanged (too short for LUFS)."""
        wav = np.zeros(8000, dtype=np.float32)
        out = self.post.loudness_normalize(wav)
        np.testing.assert_array_equal(out, wav)

    def test_loudness_normalize_returns_float32(self):
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = self.post.loudness_normalize(wav)
        assert out.dtype == np.float32

    def test_loudness_normalize_peak_clamp(self):
        """Output peak must never exceed 1.0 even on loud input."""
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        wav = (5.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = self.post.loudness_normalize(wav)
        assert np.abs(out).max() <= 1.0 + 1e-6
