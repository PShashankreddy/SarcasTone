"""Split integrity: determinism, stratification, no leakage."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcastone.data.make_splits import make_splits  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    return pd.DataFrame({
        "utt_id": [f"u{i}" for i in range(200)],
        "text": [f"utterance {i}" for i in range(200)],
        "speaker": [f"spk{i % 10}" for i in range(200)],
        "show": ["test"] * 200,
        "label": [i % 2 for i in range(200)],   # perfectly balanced
    })


def test_sizes_and_disjoint(synthetic, tmp_path):
    parts = make_splits(synthetic.copy(), seed=42, out_dir=tmp_path / "splits")
    n = len(synthetic)
    assert abs(len(parts["test"]) / n - 0.15) < 0.02
    assert abs(len(parts["val"]) / n - 0.15) < 0.02
    assert abs(len(parts["train"]) / n - 0.70) < 0.02

    ids = [set(p.utt_id) for p in parts.values()]
    assert ids[0] | ids[1] | ids[2] == set(synthetic.utt_id)      # complete cover
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_determinism(synthetic, tmp_path):
    a = {k: sorted(v.utt_id)
         for k, v in make_splits(synthetic.copy(), seed=42, out_dir=tmp_path / "a").items()}
    b = {k: sorted(v.utt_id)
         for k, v in make_splits(synthetic.copy(), seed=42, out_dir=tmp_path / "b").items()}
    assert a == b


def test_stratification(synthetic, tmp_path):
    parts = make_splits(synthetic.copy(), seed=42, out_dir=tmp_path / "s")
    for part in parts.values():
        ratio = part.label.mean()
        assert 0.45 <= ratio <= 0.55
