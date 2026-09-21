"""T3 (Option A): grow the TRAINING pool with MUStARD++ extra clips.

Locked protocol: the val/test splits are NEVER touched. This module writes a
new splits directory ``data/processed/splits_expanded/`` where:
  * val.csv  == locked val.csv  (byte-identical)
  * test.csv == locked test.csv (byte-identical)
  * train.csv = locked train.csv + the 514 new rows (train 482 -> 996)

Usage: python -m sarcastone.data.expand_splits
"""
import hashlib
from pathlib import Path

import pandas as pd

from ..utils import DATA_PROCESSED, DATA_RAW, SPLITS_DIR

EXPANSION_CSV = DATA_RAW / "mustard_pp" / "mustardpp_expansion_rows.csv"
OUT_DIR = DATA_PROCESSED / "splits_expanded"
COLS = ["utt_id", "text", "context", "speaker", "show", "label", "split"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand() -> dict[str, pd.DataFrame]:
    locked = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    exp = pd.read_csv(EXPANSION_CSV)
    exp["split"] = "train"

    assert len(exp) == 514, f"expected 514 new rows, got {len(exp)}"
    assert set(exp.columns) == set(COLS), exp.columns
    assert not (set(exp.utt_id.astype(str)) & set(locked["train"].utt_id.astype(str)))
    assert not (set(exp.utt_id.astype(str)) & set(locked["val"].utt_id.astype(str)))
    assert not (set(exp.utt_id.astype(str)) & set(locked["test"].utt_id.astype(str)))

    new_train = (pd.concat([locked["train"], exp[COLS]], ignore_index=True)
                 .sample(frac=1.0, random_state=42).reset_index(drop=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_train.to_csv(OUT_DIR / "train.csv", index=False)
    # val/test: copy bytes verbatim from the locked split
    for s in ("val", "test"):
        (OUT_DIR / f"{s}.csv").write_bytes((SPLITS_DIR / f"{s}.csv").read_bytes())

    val_ok = _sha256(OUT_DIR / "val.csv") == _sha256(SPLITS_DIR / "val.csv")
    test_ok = _sha256(OUT_DIR / "test.csv") == _sha256(SPLITS_DIR / "test.csv")
    status = "unchanged" if (val_ok and test_ok) else "CHANGED!!"
    print(f"val/test {status}; train = {len(new_train)}")
    print(f"  train class balance: {new_train.label.value_counts().to_dict()}")
    print(f"  val/test sizes: {len(locked['val'])}/{len(locked['test'])}")
    print(f"wrote -> {OUT_DIR}")
    if not (val_ok and test_ok):
        raise SystemExit("FATAL: locked val/test were modified")
    return {"train": new_train, "val": locked["val"], "test": locked["test"]}


if __name__ == "__main__":
    expand()
