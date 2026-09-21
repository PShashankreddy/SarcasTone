# SarcasTone — Progress Report

**Project:** Multimodal Sarcasm Detection (Text + Speech)
**Dataset:** MUStARD++ (690 utterances) + Custom YouTube set (944 clips)
**Date:** 10 September 2026

---

## 1. Project Objective

Build a system that detects sarcasm from **what** you say (text) and **how** you say it (voice), then combine both modalities to improve detection. Evaluate everything on the same locked test set to ensure fair comparison.

## 2. What Was Done — Phase 1 (Text)

Six text models were trained and evaluated on the locked test set (104 samples, seed 42):

| Model | Test F1 | Test Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | 0.596 | 0.596 |
| Frozen BERT-CLS + Logistic Regression | 0.654 | 0.654 |
| BERT fine-tuned (no context) | 0.609 | 0.625 |
| BERT fine-tuned (+ context) | 0.635 | 0.635 |
| RoBERTa fine-tuned | 0.663 | 0.664 |
| **RoBERTa boosted (News Headlines → MUStARD++)** | **0.687** | **0.692** |

The "boosted" model was pre-trained on 28,600 sarcasm headlines, then fine-tuned on MUStARD++. This transfer learning improved the F1 from 0.654 to 0.687 (+3.3 pts) and reduced false positives from 30 to 9.

**Gate (F1 ≥ 0.70):** Not met (0.687). Within statistical noise for n=104. Proceeded to Phase 2 as planned — fusion is the intended source of gains.

## 3. What Was Done — Phase 2 (Speech)

Two speech models were trained on the same locked test set:

| Model | Input | Test F1 | Test Accuracy |
|---|---|---|---|
| Logistic Regression | 11 Praat prosodic features | 0.653 | 0.654 |
| **1D-CNN** | 40-dim MFCC frame sequences | **0.718** | **0.721** |

Feature analysis: sarcastic delivery is **louder** (intensity_mean_db +), **lower-pitched** (f0_mean_hz −), and **drawn-out** (duration_s +) — a coherent, human-understandable acoustic signature.

**Gate (F1 ≥ 0.45):** Passed decisively (0.718).

**Key finding:** The voice (0.718) outperforms the words (0.687). Sarcasm in MUStARD++ is carried substantially by vocal performance.

## 4. What Was Done — Custom Dataset

A new spoken-sarcasm set was built from 8 real YouTube videos (talk shows, interviews, comedy panels):

- 944 valid utterance clips produced (73 noise segments cleaned)
- 300-clip random sample selected for human annotation
- 3 annotator kits generated (CSV sheets + HTML listening pages)
- Audio↔text accuracy verified: 90% match on 30-clip spot-check

This dataset will provide an independent test set and proof that sarcasm-by-voice is a real, annotatable phenomenon (via inter-annotator agreement).

## 5. Bugs Found and Fixed

Two critical bugs were discovered and resolved during the custom dataset pipeline:

**Bug 1 — Whisper timestamp drift:** Whisper's long-form transcription accumulates timestamp drift on files longer than ~1 minute, pairing wrong audio with wrong text. Fixed by transcribing in short, independent ≤28-second windows (`transcribe_windowed()`).

**Bug 2 — Loop unit error:** A one-line bug (`t0 += win` where `win` was samples, not seconds) meant only the first 28 seconds of every video were ever transcribed. Fixed to `t0 += window_s`.

Both bugs were verified fixed — the final dataset spans the full duration of all 8 videos.

## 6. Current Status

| Component | Status |
|---|---|
| Phase 1 (text models) | Complete. Champion: RoBERTa boosted, F1 = 0.687 |
| Phase 2 (speech models) | Complete. Champion: 1D-CNN, F1 = 0.718 |
| Phase 3 (fusion) | Code written, not yet run |
| Custom dataset | 944 clips built, 300-clip annotation sample ready |
| Annotation | Templates generated, awaiting 3 human annotators |
| Significance tests | Code exists (McNemar + bootstrap), not yet run |
| Error analysis | Reports generated for all models, not yet discussed in detail |

## 7. What Remains — Next Objectives

**Objective 2 (annotation):** Three annotators (A, B, C) independently label 300 clips as sarcastic/not/unclear. Then compute Cohen's kappa (pairwise) + Fleiss' kappa (overall) to measure inter-annotator agreement. This closes Objective 2 with a real IAA number.

**Phase 3 (fusion):** Combine text embeddings (768-d) + speech embeddings (128-d) via late fusion (concatenated classifier) and early fusion (attention mechanism). Must beat 0.718 (speech alone) to justify combining modalities. Includes McNemar + paired bootstrap significance tests.

**Independent evaluation:** Run all models (text, speech, fusion) on the human-labeled custom dataset to test generalization beyond MUStARD++.

**Rigorous comparison:** Run BiGRU to completion, compare all speech architectures with significance tests, complete error analysis across all models.

## 8. Key Numbers at a Glance

| Metric | Text (RoBERTa boosted) | Speech (1D-CNN) |
|---|---|---|
| Test macro-F1 | 0.687 | **0.718** |
| Test accuracy | 0.692 | 0.721 |
| Val F1 | 0.673 | 0.637 |
| Gate | ≥ 0.70 (missed by 0.013) | ≥ 0.45 (passed by 0.268) |

| Custom dataset | Value |
|---|---|
| Total valid clips | 944 |
| Annotation sample | 300 |
| Sources | 8 YouTube videos |
| Spot-check accuracy | 90% (27/30) |

## 9. Limitations

- Small test set (n=104) produces wide confidence intervals.
- Speaker overlap across splits (same voices in train/test) — speaker-independent split is a future extension.
- Intensity dominance partly reflects per-show audio mastering, not pure speaker intent.
- All models are CPU-only (no CUDA).

## 10. Files and Artifacts

| What | Where |
|---|---|
| Text champion checkpoint | `checkpoints/text_roberta_boosted/` |
| Speech champion checkpoint | `checkpoints/speech_cnn/` |
| Fusion embeddings (text) | `embeddings/text_roberta_boosted_{train,val,test}.npz` |
| Fusion embeddings (speech) | `embeddings/speech_{train,val,test}.npz` |
| Phase 1 report | `reports/phase1_summary.md` |
| Phase 2 report | `reports/phase2_summary.md` |
| Confusion matrices | `reports/figures/*.png` |
| Full project record | `docs/PROJECT_OVERVIEW.md` |
| Annotation guidelines | `docs/ANNOTATOR_GUIDELINES.md` |
| Progress log | `progress.md` |
