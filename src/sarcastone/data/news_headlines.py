"""Text-booster corpus: Kaggle News Headlines Sarcasm v2 (~28.6k).

Put `Sarcasm_Headlines_Dataset.json` (one JSON object per line) into data/raw/,
then:  python -m sarcastone.data.news_headlines

Writes train/val/test CSVs (same schema as MUStARD splits) to
data/processed/splits_newsheadlines/. Use with:
  python -m sarcastone.models.bert_finetune --splits_dir data/processed/splits_newsheadlines
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import DATA_RAW, DATA_PROCESSED, SEED


def load_headlines() -> pd.DataFrame:
    path = DATA_RAW / "Sarcasm_Headlines_Dataset.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - download from Kaggle "
                         "(Saurabh Shahane, Sarcasm Headlines Dataset)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)[["is_sarcastic", "headline"]]
    df = df.rename(columns={"is_sarcastic": "label", "headline": "text"})
    df["utt_id"] = [f"nh{i}" for i in range(len(df))]
    df["speaker"] = "onion/huffpost"
    df["show"] = "news"
    return df[["utt_id", "text", "speaker", "show", "label"]]


def make_splits(seed: int = SEED) -> dict[str, pd.DataFrame]:
    out_dir = DATA_PROCESSED / "splits_newsheadlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_headlines()

    train_df, test_df = train_test_split(df, test_size=0.10, random_state=seed,
                                         stratify=df.label)
    train_df, val_df = train_test_split(train_df, test_size=0.1111, random_state=seed,
                                        stratify=train_df.label)
    parts = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in parts.items():
        p = part.reset_index(drop=True)
        p.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(p)} rows ({p.label.mean():.2%} sarcastic) -> {out_dir / (name + '.csv')}")
    return parts


if __name__ == "__main__":
    make_splits()
