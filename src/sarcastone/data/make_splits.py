"""Create the LOCKED train/val/test splits (stratified, fixed seed) from MUStARD++.

Splits are saved as CSVs (utt_id, text, speaker, label, split). All three models
(text / speech / fusion) must be evaluated on the same locked test set.

Usage: python -m sarcastone.data.make_splits
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import MUSTARD_JSON, SPLITS_DIR, LABEL2ID, SEED

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def build_dataframe() -> pd.DataFrame:
    data = json.loads(MUSTARD_JSON.read_text(encoding="utf-8"))
    rows = []
    for utt_id, v in sorted(data.items()):
        rows.append({
            "utt_id": str(utt_id),
            "text": v["utterance"].strip(),
            # preceding dialogue lines (sarcasm often depends on them)
            "context": " ".join(u.strip() for u in v.get("context", [])),
            "speaker": v["speaker"],
            "show": v["show"],
            "label": int(LABEL2ID[v["sarcasm"]]),
        })
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame | None = None, seed: int = SEED,
                out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Create locked splits. NOTE: only pass `df`/`out_dir` for experiments/tests —
    production MUStARD++ splits are written once by running this with no args."""
    out_dir = Path(out_dir) if out_dir else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = build_dataframe()

    train_df, test_df = train_test_split(
        df, test_size=TEST_FRAC, random_state=seed, stratify=df["label"]
    )
    val_frac_of_remainder = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, val_df = train_test_split(
        train_df, test_size=val_frac_of_remainder, random_state=seed, stratify=train_df["label"]
    )

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = part.reset_index(drop=True).copy()
        out["split"] = name
        out.to_csv(out_dir / f"{name}.csv", index=False)

    summary = {
        name: {
            "n": len(part),
            "sarcastic": int((part.label == 1).sum()),
            "non_sarcastic": int((part.label == 0).sum()),
        }
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    print(pd.DataFrame(summary).T.to_string())
    print(f"\nLocked splits saved -> {out_dir}  (seed={seed})")
    overlap_ids = (
        (set(train_df.utt_id) & set(test_df.utt_id))
        | (set(train_df.utt_id) & set(val_df.utt_id))
        | (set(val_df.utt_id) & set(test_df.utt_id))
    )
    print("leakage check:", "FAIL" if len(overlap_ids) else f"OK (no overlap across {len(df)} utterances)")
    return {"train": train_df, "val": val_df, "test": test_df}


if __name__ == "__main__":
    make_splits()
