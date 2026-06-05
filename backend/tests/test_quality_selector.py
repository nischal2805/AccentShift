"""QualitySelector: formula correctness + picks the higher-scoring candidate."""
import numpy as np
from pipeline.quality_selector import QualitySelector
from pipeline.types import Segment, Candidate, EmotionVec


CFG = {"quality": {"emotion_weight": 0.4, "wer_weight": 0.4, "mos_weight": 0.2}}


class _StubMM:
    def load_utmos(self):
        return None  # selector falls back to MOS 3.5


class _StubFE:
    """Emotion fixed; transcript echoes a per-candidate label baked into wav[0]."""
    def __init__(self, src_emotion):
        self.src_emotion = src_emotion

    def emotion(self, seg):
        return self.src_emotion  # perfect emotion match -> emo_sim = 1.0

    def transcribe(self, seg):
        # wav[0] == 1.0 -> exact transcript; else garbage -> WER 1.0
        return ("hello world", []) if seg.audio[0] == 1.0 else ("zzz", [])


def _selector():
    src = EmotionVec(valence=0.7, arousal=0.6, dominance=0.5)
    return QualitySelector(CFG, _StubFE(src), _StubMM()), src


def test_score_formula():
    qs, src = _selector()
    good = Candidate(name="seed_vc", wav=np.ones(16000, dtype=np.float32), sr=16000)
    score, meta = qs.score(good, Segment(good.wav, 0, 1, 16000), src, "hello world")
    # emo_sim=1, wer=0, mos=3.5 -> 0.4*1 + 0.4*1 + 0.2*(3.5/5) = 0.94
    assert meta["wer"] == 0.0
    assert abs(score - (0.4 + 0.4 + 0.2 * 0.7)) < 1e-6


def test_select_picks_lower_wer():
    qs, src = _selector()
    good = Candidate("seed_vc", np.ones(16000, dtype=np.float32), 16000)        # exact text
    bad = Candidate("vevo", np.full(16000, 0.5, dtype=np.float32), 16000)        # garbage text
    chosen, score, meta = qs.select([bad, good], Segment(good.wav, 0, 1, 16000),
                                    src, "hello world")
    assert chosen.name == "seed_vc"
    assert meta["wer"] == 0.0
