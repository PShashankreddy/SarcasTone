# Phase 2 Summary — Speech-Based Sarcasm Detection

Date: 2026-08-22 · Seed: 42 · Device: CPU · Data: MUStARD++ audio (690 clips, 345/345)

## Pipeline

1. **Audio**: 690 official MUStARD clips (HuggingFace `mmsd_raw_data.zip`) → 16 kHz mono WAV
   (`data/raw/audio/`). Clip IDs match the locked Phase-1 splits exactly.
2. **Features** (all 690 extracted, zero failures):
   - Praat summaries (11 prosodic/voice-quality stats) → `data/processed/acoustic_summary.csv`
   - MFCC frame sequences, 40-dim (13 MFCC + Δ + ΔΔ + RMS) → `data/processed/features_seq/`
   - z-score normalizer fit on TRAIN split only.
3. **ASR audit**: Whisper-small transcripts vs gold text (`data/interim/whisper_transcripts.csv`).

## Results (test split, n=104)

| Model | Input | Val F1 | Test F1 | Acc |
|---|---|---|---|---|
| LogReg | 11 Praat summaries | 0.613 | 0.6527 | 0.6538 |
| **1D-CNN** ✅ | 40-dim × T MFCC frames | 0.637 | **0.7180** | 0.7212 |

CNN confusion (test): TN=43 FP=9 / FN=20 TP=32. Early stop at epoch 27 (best val F1 0.6367).
Artifacts: `checkpoints/speech_cnn/`, fusion embeddings `embeddings/speech_{split}.npz` (128-d),
`reports/phase2_{lr,cnn}_test_metrics.json`, error analyses + confusion PNGs in `reports/`.

### Baseline feature importance (|coef|)
`intensity_mean_db` (+) ≫ `f0_mean_hz` (−) > `duration_s` (+) > `hnr_mean_db` (−) >
pitch spread (f0_p05/p95/range/std). Consistent with louder, lower-pitched, drawn-out sarcastic delivery.

## ASR / audio-quality audit

Whisper-small on 690 clips: **median WER 0.200**; mean inflated to 1.001 by **31 clips (4.5%)**
where the model hallucinates on laugh-track/music segments ("No!" → "no, no, no ×16").
These 31 utt_ids are flagged in `whisper_transcripts.csv` (wer > 1) as known-noisy audio.

## Decision gate

> Gate: speech test F1 ≥ 0.45 → **PASSED decisively (0.718)** — no fallbacks needed.

## Key findings & caveats for the thesis

1. **Audio is the strongest single modality** in our setup: CNN 0.718 vs boosted-RoBERTa text arm
   0.687 (Phase 1). Sarcasm in MUStARD++ is carried substantially by vocal performance — a central,
   defensible thesis claim.
2. **Confound to document**: `intensity_mean_db` dominance partly reflects per-show/episode audio
   mastering rather than speaker intent; splits are speaker-dependent (same voices in train/test),
   which flatters acoustic models. Both belong in limitations.
3. Whisper audit enables two later ablations: (a) do CNN errors concentrate in high-WER clips?
   (b) re-run boosted RoBERTa on ASR text vs clean subtitles to quantify transcript-quality effects.
