"""validate_refs.check_clip: flags bad reference clips (pure, no models)."""
import numpy as np
import soundfile as sf
import pytest
from scripts.validate_refs import check_clip

SR = 16000
ARGS = dict(min_s=3.0, max_s=15.0, silence_floor=1e-3, clip_frac_max=0.01)


def _write(tmp_path, name, audio, sr=SR):
    p = tmp_path / name
    sf.write(str(p), audio, sr)
    return p


def test_good_clip_passes(tmp_path):
    t = np.linspace(0, 6.0, 6 * SR, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    assert check_clip(_write(tmp_path, "ok.wav", audio), **ARGS) == []


def test_too_short_flagged(tmp_path):
    audio = (0.3 * np.ones(SR, dtype=np.float32))   # 1s
    problems = check_clip(_write(tmp_path, "short.wav", audio), **ARGS)
    assert any("too short" in p for p in problems)


def test_silent_flagged(tmp_path):
    audio = np.zeros(6 * SR, dtype=np.float32)
    problems = check_clip(_write(tmp_path, "silent.wav", audio), **ARGS)
    assert any("near-silent" in p for p in problems)


def test_clipped_flagged(tmp_path):
    audio = np.ones(6 * SR, dtype=np.float32)        # full-scale = clipped
    problems = check_clip(_write(tmp_path, "clip.wav", audio), **ARGS)
    assert any("clipped" in p for p in problems)


def test_stereo_flagged(tmp_path):
    t = np.linspace(0, 6.0, 6 * SR, endpoint=False)
    mono = 0.3 * np.sin(2 * np.pi * 200 * t)
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    problems = check_clip(_write(tmp_path, "stereo.wav", stereo), **ARGS)
    assert any("channels" in p for p in problems)


def test_unreadable(tmp_path):
    bad = tmp_path / "nope.wav"
    bad.write_bytes(b"not a wav")
    assert any("unreadable" in p for p in check_clip(bad, **ARGS))
