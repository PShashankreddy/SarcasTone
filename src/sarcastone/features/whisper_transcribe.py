"""Whisper ASR transcription + WER vs MUStARD++ gold transcripts.

Input: data/raw/audio/{utt_id}.wav (16 kHz mono; run convert_audio first).
Uses the transformers Whisper implementation (no openai-whisper install needed).

Usage: python -m sarcastone.features.whisper_transcribe --model small
"""
import argparse
from pathlib import Path

import pandas as pd

from ..utils import AUDIO_RAW, TRANSCRIPTS_CSV, SPLITS_DIR


def wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[-1] / max(len(r), 1)


def transcribe(model_name: str = "small"):
    from .asr import load_asr  # lazy import: heavy dependency

    model = load_asr(model_name)
    wavs = sorted(AUDIO_RAW.glob("*.wav"))

    # resume support: keep prior transcriptions, only process new clips
    rows = {}
    if TRANSCRIPTS_CSV.exists():
        prev = pd.read_csv(TRANSCRIPTS_CSV)
        rows = dict(zip(prev.utt_id, prev.asr_text))
        print(f"resuming: {len(rows)} already transcribed")
    todo = [w for w in wavs if w.stem not in rows]
    print(f"transcribing {len(todo)}/{len(wavs)} clips from {AUDIO_RAW}")

    for i, wav in enumerate(todo):
        result = model.transcribe(str(wav), language="english")
        rows[wav.stem] = result["text"].strip()
        if (i + 1) % 25 == 0 or i == len(todo) - 1:      # periodic checkpoint
            _save(rows)

    merged = _save(rows)
    n_bad = int((merged.wer > 1).sum())
    print(f"\nmedian WER: {merged.wer.median():.3f} | mean: {merged.wer.mean():.3f} "
          f"(skewed by {n_bad} hallucinated laugh-track clips)")


def _save(rows: dict):
    asr_df = pd.DataFrame({"utt_id": list(rows), "asr_text": list(rows.values())})
    gold_df = pd.concat(
        [pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")]
    )[["utt_id", "text"]]
    merged = gold_df.merge(asr_df, on="utt_id", how="inner")
    merged["wer"] = [wer(a, b) for a, b in zip(merged.text, merged.asr_text)]

    TRANSCRIPTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TRANSCRIPTS_CSV, index=False)
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny/base/small/medium")
    args = ap.parse_args()
    transcribe(args.model)
