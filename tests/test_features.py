"""Feature extraction sanity on synthetic signals (440 Hz sine wave).
Skipped automatically when audio deps (parselmouth/librosa/soundfile) are absent."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory):
    sf = pytest.importorskip("soundfile")
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t)          # clean 440 Hz tone
    path = tmp_path_factory.mktemp("audio") / "sine440.wav"
    sf.write(path, y.astype(np.float32), sr)
    return path


def test_praat_pitch(sine_wav):
    parselmouth = pytest.importorskip("parselmouth")
    from sarcastone.features.acoustic_praat import extract_summary, SUMMARY_KEYS

    feats = extract_summary(str(sine_wav))
    assert set(SUMMARY_KEYS) == set(feats.keys())
    assert abs(feats["f0_mean_hz"] - 440) < 10       # pitch recovered correctly
    assert feats["f0_range"] < 20                    # steady tone -> narrow range


def test_librosa_sequence(sine_wav):
    librosa = pytest.importorskip("librosa")
    from sarcastone.features.acoustic_librosa import extract_sequence, FEATURE_DIM

    seq = extract_sequence(str(sine_wav))
    assert seq.ndim == 2 and seq.shape[1] == FEATURE_DIM   # (T, 40)
    assert seq.shape[0] >= 1
