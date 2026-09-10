"""Shared speech datasets: padded acoustic sequences + Praat summary features."""
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..utils import FEATURES_SEQ_DIR, SPLITS_DIR, SUMMARY_CSV, DATA_PROCESSED

MAX_FRAMES = 200  # ~12.5 s at 20ms hop; longer sequences get truncated


def load_norm():
    path = DATA_PROCESSED / "acoustic_norm.json"
    if not path.exists():
        raise FileNotFoundError("run `python -m sarcastone.features.build_features` first")
    return json.loads(path.read_text())


class SeqDataset(Dataset):
    """(T,40) MFCC/energy sequences -> z-scored, fixed-length (MAX_FRAMES,40)."""

    def __init__(self, df: pd.DataFrame):
        self.norm = load_norm()
        self.items = []
        for _, r in df.iterrows():
            seq = np.load(FEATURES_SEQ_DIR / f"{r.utt_id}.npy")[:MAX_FRAMES]
            mean = np.asarray(self.norm["mean"]); std = np.asarray(self.norm["std"])
            seq = ((seq - mean) / std).astype(np.float32)
            pad = np.zeros((MAX_FRAMES - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
            self.items.append((seq.T.copy(), int(r.label)))   # (D, T) for Conv1d

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return x, y


def summary_frame(split: str) -> pd.DataFrame:
    gold = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    summ = pd.read_csv(SUMMARY_CSV)
    df = gold.merge(summ, on="utt_id", how="inner")
    return df


def summary_xy(df: pd.DataFrame):
    X = df[["duration_s", "f0_mean_hz", "f0_std_hz", "f0_p05", "f0_p95", "f0_range",
            "intensity_mean_db", "intensity_std_db", "hnr_mean_db",
            "jitter_local", "shimmer_local"]].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)   # unvoiced clips
    return X, df.label.values
