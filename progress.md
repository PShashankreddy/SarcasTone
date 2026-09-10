# SarcasTone — Progress Log

Chronological record of everything done

---

## Project setup

- Created `D:\sarcastone`, Python 3.12.3 venv (`.venv`)
- Installed core deps: numpy, pandas, scikit-learn, torch, transformers, matplotlib, seaborn
- Installed audio deps: librosa, soundfile, praat-parselmouth
- Set up package structure: `src/sarcastone/` with 7 sub-packages (data, features, models, training, evaluation, inference)
- Created `pyproject.toml`, `requirements.txt`, `.gitignore`, `AGENTS.md`
- Created `configs/` with YAML hyperparameters for all phases
- Created `tests/` — split integrity + feature extraction sanity tests
- All tests passing

## Data collection

- Downloaded MUStARD++ from GitHub (`sarcasm_data.json`) — 690 utterances, ~50/50 sarcastic/non-sarcastic
- Created locked stratified splits (seed 42): train 482 / val 104 / test 104
- Preprocessed text (whitespace collapse, merged `dataset.csv` with split column)
- Verified: no speaker leakage across splits (logged for limitations)

##  Phase 1: Text-based sarcasm detection

Ran a model ladder, each rung isolating one design decision:

| # | Model | Test F1 | Notes |
|---|---|---|---|
| 1 | TF-IDF + LogReg | 0.596 | Classical lexical baseline |
| 2 | Frozen BERT-CLS + LogReg | 0.654 | Pretrained representations help (+5.8 pts) |
| 3 | BERT fine-tuned (no context) | 0.609 | Underperforms frozen — seed sensitivity |
| 4 | BERT fine-tuned (+ context) | 0.635 | Context helps BERT (+2.6 pts) |
| 5 | RoBERTa fine-tuned (no context) | 0.663 | Stronger pretraining objective helps |
| 6 | RoBERTa fine-tuned (+ context) | 0.643 | Context doesn't help RoBERTa here |
| 7 | RoBERTa fine-tuned (no context, 5 ep) | 0.654 | Longer training helps val (0.671) but not test |
| 8 | News Headlines only | 0.919* | Sanity check — different dataset, not comparable |
| 9 | **RoBERTa boosted (NH → MUStARD++)** | **0.687** | **Final text champion** |

**Champion selection:** Rows 7 and 9 tie on val (0.671 vs 0.673). Promoted row 9 because (a) tie broken in its favour, (b) strictly more supervision (transfer learning), an a-priori advantage.

**Error profile of champion:** TN=43 FP=9 / FN=23 TP=29. Far lower false-positive rate than earlier runs (FP was 15–30 before).

**Gate verdict:** 0.687 < 0.70 → **not met** (missed by 0.013). Within noise band for n=104. Proceeded to Phase 2 as planned — fusion (Phase 3) is the intended source of gains.

## Phase 2: Speech-based sarcasm detection

### Audio collection
- Fetched 690 official MUStARD++ clips via HuggingFace `mmsd_raw_data.zip` (1.4 GB)
- Converted to uniform 16 kHz mono WAVs (`data/raw/audio/`)
- Verified: clip IDs match locked splits 1:1

### Feature extraction
- **Praat summaries:** 11 prosodic/voice-quality stats (pitch, intensity, HNR, jitter, shimmer) — 690/690 extracted, zero failures
- **MFCC sequences:** 40-dim frames (13 MFCC + Δ + Δ² + RMS) — 690/690 extracted
- z-score normalizer fitted on **train split only** (no leakage)

### Models

| Model | Input | Val F1 | Test F1 | Acc |
|---|---|---|---|---|
| Logistic Regression | 11 Praat summaries | 0.613 | 0.6527 | 0.6538 |
| **1D-CNN** | 40-dim Δ MFCC frames | 0.637 | **0.7180** | 0.7212 |

**CNN confusion (test):** TN=43 FP=9 / FN=20 TP=32. Early stop at epoch 27.

**Baseline feature importance (|coef|):** `intensity_mean_db (+)` ≫ `f0_mean_hz (−)` > `duration_s (+)` > `hnr_mean_db (−)` > pitch spread. Sarcastic delivery is louder, lower-pitched, and drawn-out.

**Whisper audit:** median WER 0.200; 31/690 clips (4.5%) flagged as hallucinated (WER > 1) due to laugh-track segments.

**Gate verdict:** 0.718 ≥ 0.45 → **passed decisively**.

**Key finding:** Audio (0.718) outperforms text (0.687) — sarcasm in MUStARD++ is carried substantially by vocal performance.

## Aug 23–Sep 5 — Custom dataset pipeline

### Objective: build a new spoken-sarcasm set from YouTube

1. Built `segment_youtube.py`: yt-dlp download → Whisper ASR → ffmpeg cut at utterance boundaries → `context.json` manifest
2. Built `asr.py`: Windows-safe Whisper wrapper via HuggingFace transformers (no triton)
3. Collected 8 real conversational YouTube videos (talk shows, interviews, comedy panels)
4. First attempt produced 876 candidate clips

### Annotation infrastructure
- Built `annotation.py`: `make-template`, `make-html`, `agreement` (Cohen's + Fleiss' kappa), `merge`
- Generated 300-clip random sample for annotation
- Created 3 annotator CSV sheets + 3 HTML listening pages
- Wrote `docs/ANNOTATOR_GUIDELINES.md`

## Sep 6–7 — Bug discovery & fix (the hard part)

### Bug 1: Whisper timestamp drift
**Symptom:** audio in annotation files didn't match the text below it for the majority of clips.

**Diagnosis:** Whisper's built-in long-form chunking accumulates timestamp drift — its 30 s windows get misaligned the deeper into a video you go, pairing wrong audio with wrong text.

**Fix:** `transcribe_windowed()` — process audio in short, independent ≤28 s windows (short inputs don't drift), stitch timestamps back to full-video time.

### Bug 2: The one-line loop error
**Symptom:** After "fixing" drift, rebuilds produced absurdly few clips (e.g. 9 from a 75-minute video).

**Diagnosis:** In `asr.py`, `t0 += win` added **samples** (448,000) to a variable counted in **seconds**. After the first 28 s window, the loop exited — we'd only ever transcribed the first 28 seconds of every video.

**Fix:** `t0 += window_s` (seconds, not samples).

### Verification
- Rebuilt all 8 videos from scratch (re-downloaded audio, re-transcribed, re-cut)
- Final dataset: **944 valid clips** across 8 sources
- Cleaned 73 noise segments (ALLCAPS LAUGHTER/APPLAUSE, repeated hallucinations, host catchphrases)
- Spot-checked 30-clip sample: **27/30 (90%) match** — 3 residual edge-slop clips dropped
- Regenerated annotation templates + HTML pages

## Sep 10 — Documentation & GitHub prep

- Rewrote `.gitignore` — clean rules excluding `data/`, `checkpoints/`, `embeddings/`
- Fixed `requirements.txt` — replaced `openai-whisper` with `transformers`, added `yt-dlp` + `imageio-ffmpeg`
- Deleted stale `src/sarcastone.egg-info/`
- Rewrote `README.md` — comprehensive one-stop file (every file explained, setup, results, reproduction)
- Created `progress.md` (this file)
- Created `docs/PROJECT_OVERVIEW.md` — complete project record covering Phase 1 + Phase 2

## Remaining

| Task | Status | Blocked by |
|---|---|---|
| Objective 2 — annotation | Templates ready, CSVs blank | 3 human annotators (A/B/C) |
| Objective 2 — IAA report | Not started | Filled annotation CSVs |
| Objective 2 — merge | Not started | IAA agreement |
| Phase 3 — fusion | Code written, not yet run | Nothing (ready to go) |
| Phase 3 — significance tests | Code written, not yet run | Phase 3 fusion results |

