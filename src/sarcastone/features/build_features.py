"""Build the unified speech dataset: Praat summaries + MFCC sequences for every
utterance that has audio; align with labels from the locked splits.

Outputs:
  data/processed/acoustic_summary.csv          (per-utterance summary stats)
  data/processed/features_seq/{utt_id}.npy     (T x 40 sequences)
  data/processed/acoustic_norm.json            (z-score stats from TRAIN only)

Usage: python -m sarcastone.features.build_features
"""
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import AUDIO_RAW, SPLITS_DIR, FEATURES_SEQ_DIR, SUMMARY_CSV, DATA_PROCESSED
from .acoustic_praat import extract_summary, SUMMARY_KEYS
from .acoustic_librosa import extract_sequence, fit_normalizer


def build():
    FEATURES_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    gold = pd.concat([pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")])
    wavs = {w.stem: w for w in AUDIO_RAW.glob("*.wav")}
    print(f"{len(wavs)} wav files | {len(gold)} labeled utterances")

    rows, seq_ids, have_audio = [], [], set()
    for _, r in tqdm(gold.iterrows(), total=len(gold)):
        utt_id = str(r.utt_id)
        if utt_id not in wavs:
            continue
        have_audio.add(utt_id)
        try:
            rows.append({"utt_id": utt_id, **extract_summary(wavs[utt_id])})
            np.save(FEATURES_SEQ_DIR / f"{utt_id}.npy", extract_sequence(wavs[utt_id]))
            seq_ids.append(utt_id)
        except Exception as e:  # bad/corrupt audio -> report, keep going
            print(f"[warn] {utt_id}: {e}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # normalizer fit on TRAIN utterances only (no test leakage)
    train_ids = set(gold[gold.split == "train"].utt_id.astype(str))
    train_seqs = [np.load(FEATURES_SEQ_DIR / f"{u}.npy") for u in sorted(train_ids & set(seq_ids))]
    norm = fit_normalizer(train_seqs)
    (DATA_PROCESSED / "acoustic_norm.json").write_text(json.dumps(norm), encoding="utf-8")

    missing = set(gold.utt_id.astype(str)) - have_audio
    print(f"\nsummary features: {len(summary_df)} | sequences: {len(seq_ids)}")
    print(f"missing audio: {len(missing)} utterances")
    print(f"saved -> {SUMMARY_CSV}, {FEATURES_SEQ_DIR}")
    return summary_df


if __name__ == "__main__":
    build()
