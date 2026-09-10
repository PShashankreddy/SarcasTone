"""Light text preprocessing + merged dataset.csv with split column.

BERT consumes raw cased/punctuated text, so cleaning is intentionally minimal:
collapse whitespace and drop empty utterances.

Usage: python -m sarcastone.data.preprocess_text
"""
import re

import pandas as pd

from .make_splits import make_splits
from ..utils import DATA_PROCESSED

_WS = re.compile(r"\s+")


def clean_text(t: str) -> str:
    return _WS.sub(" ", t).strip()


def preprocess():
    parts = make_splits()
    for p in parts.values():
        p["text"] = p["text"].map(clean_text)
    full = pd.concat(parts.values()).reset_index(drop=True)
    full = full[full.text.str.len() > 0]
    out = DATA_PROCESSED / "dataset.csv"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(f"merged dataset ({len(full)} rows) -> {out}")
    return full


if __name__ == "__main__":
    preprocess()
