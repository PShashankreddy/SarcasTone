# SarcasTone — Multimodal Sarcasm Detection

Detect sarcasm from **what** you say (text) and **how** you say it (voice), then combine
both. Evaluated on MUStARD++ with a locked test set (seed 42, no leakage).

```text
Phase 1  text baseline        BERT / RoBERTa fine-tune        gate: F1 ≥ 0.70
Phase 2  speech baseline      Praat prosody + MFCC → CNN      gate: F1 ≥ 0.45
Phase 3  fusion               early MLP / late voting         target: beat speech F1
```

## Results at a glance

| Phase | Modality | Model | Test macro-F1 | Test acc | Gate |
|---|---|---|---|---|---|
| 1 | Text | RoBERTa boosted (News Headlines → MUStARD++) | 0.687 | 0.692 | 0.70 — missed by 0.013 |
| 2 | Speech | 1D-CNN over 40-dim MFCC frames | **0.718** | 0.721 | 0.45 — passed by 0.268 |
| 3 | Fusion | *not yet run* | — | — | beat 0.718 |

Key finding: **the voice (0.718) outperforms the words (0.687)** — sarcasm in MUStARD++
is carried substantially by vocal performance.

## Setup

```powershell
cd D:\sarcastone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q          # sanity: splits + feature extraction
```

**ffmpeg** is required for audio conversion and Whisper transcription:
`winget install Gyan.FFmpeg` (reopen terminal after install).

## Project structure

```text
D:\sarcastone\
│
├── src/sarcastone/             # main package (install via pyproject.toml)
│   ├── data/                   # download, splits, preprocessing, annotation
│   ├── features/               # Praat, MFCC, Whisper ASR, YouTube segmentation
│   ├── models/                 # BERT, RoBERTa, CNN, RNN, fusion
│   ├── training/               # seed, fit loop, evaluation, embeddings
│   ├── evaluation/             # metrics, error analysis, significance tests
│   └── inference/              # single-sample predictors (text/audio/multimodal)
│
├── tests/                      # split integrity + feature sanity checks
├── configs/                    # hyperparameters per phase (YAML)
├── scripts/                    # one-command phase pipelines (PowerShell)
├── reports/                    # metrics JSONs, error analyses, confusion PNGs
├── docs/                       # project overview, plain-English guides, datasets
├── praat/                      # standalone Praat extraction script
├── checkpoints/                # trained model weights (gitignored, ~2.3 GB)
├── embeddings/                 # per-split embeddings for fusion (gitignored)
├── data/                       # all data (gitignored; regenerate via scripts)
│   ├── raw/                    #   source audio + annotations
│   ├── interim/                #   intermediate clips, transcripts, annotations
│   └── processed/              #   locked splits, features, dataset.csv
│
├── README.md                   # this file
├── progress.md                 # chronological project log
├── AGENTS.md                   # project conventions
├── pyproject.toml              # package definition + dependencies
├── requirements.txt            # pip install list
└── .gitignore                  # excludes data, checkpoints, embeddings
```

## File guide

### Source code (`src/sarcastone/`)

| File | What it contains |
|---|---|
| `utils.py` | Path constants (`DATA_RAW`, `EMB_DIR`, `CKPT_DIR`), `SEED=42`, YAML/JSON helpers |
| **data/** | |
| `data/download_mustard.py` | Downloads MUStARD++ `sarcasm_data.json` from GitHub |
| `data/make_splits.py` | Creates locked stratified train/val/test CSVs (seed 42, 70/15/15) |
| `data/preprocess_text.py` | Text cleaning + merged `dataset.csv` with split column |
| `data/news_headlines.py` | Loads Kaggle News Headlines v2, writes train/val/test splits |
| `data/annotation.py` | Annotation workflow: `make-template`, `make-html`, `agreement` (Cohen's + Fleiss' kappa), `merge` |
| **features/** | |
| `features/acoustic_praat.py` | 11 Praat prosodic features via parselmouth (pitch, intensity, HNR, jitter, shimmer) |
| `features/acoustic_librosa.py` | 40-dim MFCC+delta+delta2+RMS frame sequences; z-score normalizer |
| `features/build_features.py` | Builds `acoustic_summary.csv` + per-utterance `.npy` MFCC sequences |
| `features/convert_audio.py` | Converts raw MP4 clips → 16 kHz mono WAVs via ffmpeg |
| `features/asr.py` | Whisper ASR via HuggingFace transformers (Windows-safe, no triton); includes `transcribe_windowed()` for drift-free long-audio transcription |
| `features/whisper_transcribe.py` | Batch-transcribes all WAV clips, computes WER vs gold, saves `whisper_transcripts.csv` |
| `features/segment_youtube.py` | YouTube → utterance clips: yt-dlp download → Whisper windowed ASR → ffmpeg cut → `context.json` manifest |
| **models/** | |
| `models/text_baseline.py` | TF-IDF+LogReg and frozen-BERT-CLS+LogReg baselines |
| `models/bert_finetune.py` | Fine-tune BERT/RoBERTa on MUStARD++; dumps [CLS] embeddings for fusion |
| `models/speech_cnn.py` | 1D-CNN over MFCC sequences + LR baseline on Praat summaries |
| `models/speech_rnn.py` | BiGRU over acoustic sequences (alternative to CNN) |
| `models/speech_data.py` | Shared speech datasets: `SeqDataset` (padded MFCCs) + summary helpers |
| `models/fusion.py` | Phase 3: early fusion MLP + late fusion voting + McNemar/bootstrap significance |
| **training/** | |
| `training/seed.py` | `set_seed(42)` — random, numpy, torch seeds |
| `training/trainer.py` | Generic PyTorch fit loop (early stopping on val F1), evaluate, extract embeddings |
| **evaluation/** | |
| `evaluation/metrics.py` | Accuracy, macro P/R/F1, sarcasm-class F1, confusion matrix, heatmap PNG |
| `evaluation/error_analysis.py` | Markdown error report with FP/FN ranked by confidence gap |
| `evaluation/significance.py` | McNemar test + paired bootstrap F1 difference (2000 bootstraps, 95% CI) |
| **inference/** | |
| `inference/predict_text.py` | Single-text sarcasm predictor: loads checkpoint, returns label + P(sarcasm) |
| `inference/predict_audio.py` | Single-WAV predictor: loads CNN/RNN checkpoint, extracts features on-the-fly |
| `inference/predict_multimodal.py` | Multimodal predictor: runs text + audio, fuses with tuned late weight |

### Tests (`tests/`)

| File | What it tests |
|---|---|
| `test_splits.py` | Split integrity: correct sizes, disjoint partitions, determinism, stratification |
| `test_features.py` | Feature extraction sanity: pitch recovery on sine wave, MFCC output shape |

### Configs (`configs/`)

| File | Purpose |
|---|---|
| `text_bert.yaml` | Phase 1 BERT: bert-base-uncased, 3 epochs, lr=2e-5, max_len=128 |
| `boost_headlines.yaml` | Stage 1: roberta-base on News Headlines, 1 epoch, lr=2e-5 |
| `boost_mustard.yaml` | Stage 2: fine-tune NH checkpoint on MUStARD++, 5 epochs, lr=3e-5 |
| `tune_roberta_ctx.yaml` | Tuning run: roberta + context, 4 epochs, lr=2e-5 |
| `tune_roberta_long.yaml` | Tuning run: roberta, no context, 5 epochs, lr=3e-5 |
| `speech_cnn.yaml` | Phase 2 CNN/RNN: 60 epochs, lr=1e-3, hidden=64, dropout=0.3 |
| `fusion.yaml` | Phase 3: early MLP (hidden=256, 80 epochs), late fusion grid search |

### Scripts (`scripts/`)

| File | Pipeline steps |
|---|---|
| `run_phase1.ps1` | Download MUStARD → locked splits → TF-IDF/BERT baselines → fine-tune BERT + embeddings |
| `run_phase2.ps1` | Convert audio → build acoustic features → Whisper transcribe → LR baseline → CNN + embeddings |
| `run_phase3.ps1` | Early fusion → late fusion + comparison + significance → done |

### Reports (`reports/`)

| File | Contents |
|---|---|
| `phase1_summary.md` | Full Phase 1 report: 9-model comparison, champion selection, gate verdict |
| `phase2_summary.md` | Full Phase 2 report: pipeline, CNN vs LR, feature importance, Whisper audit |
| `phase{1,2}_*_test_metrics.json` | Per-model test metrics (accuracy, P/R/F1, confusion) |
| `phase{1,2}_*_error_analysis.md` | Per-model FP/FN error tables with sample-level details |
| `figures/*.png` | Confusion matrix heatmaps (7 models) |

### Documentation (`docs/`)

| File | Contents |
|---|---|
| `PROJECT_OVERVIEW.md` | Complete project record: every step, every number, bugs found, decisions made |
| `PHASE2_PLAIN_ENGLISH.md` | Phase 2 explained without jargon — materials, models, results, full glossary |
| `DATASETS.md` | Dataset registry: MUStARD++ primary, News Headlines booster, custom YouTube set, rejected options |
| `ANNOTATOR_GUIDELINES.md` | Annotator instruction sheet for the custom dataset labeling task |

## Reproducing results

### Phase 1 — text

```powershell
.\scripts\run_phase1.ps1          # or step by step:
python -m sarcastone.data.download_mustard
python -m sarcastone.data.make_splits
python -m sarcastone.data.preprocess_text
python -m sarcastone.models.text_baseline          # TF-IDF + frozen BERT
python -m sarcastone.models.bert_finetune          # BERT / RoBERTa fine-tune
python -m sarcastone.models.bert_finetune --config configs/boost_headlines.yaml  # stage 1
python -m sarcastone.models.bert_finetune --config configs/boost_mustard.yaml   # stage 2
```

### Phase 2 — speech

```powershell
.\scripts\run_phase2.ps1          # or step by step:
python -m sarcastone.features.convert_audio
python -m sarcastone.features.build_features
python -m sarcastone.features.whisper_transcribe
python -m sarcastone.models.speech_cnn              # CNN + LR baseline
```

### Phase 3 — fusion

```powershell
.\scripts\run_phase3.ps1          # runs early fusion → late fusion → significance
```

### Custom dataset pipeline

```powershell
# download + segment a YouTube video
python -m sarcastone.features.segment_youtube --url "https://www.youtube.com/watch?v=VIDEO_ID"

# generate annotation templates + HTML listening pages
python -m sarcastone.data.annotation make-template --n 300 --random
python -m sarcastone.data.annotation make-html

# after annotators fill in their CSVs:
python -m sarcastone.data.annotation agreement     # Cohen's + Fleiss' kappa
python -m sarcastone.data.annotation merge         # majority labels → custom_dataset.csv
```

## Inference

```powershell
python -m sarcastone.inference.predict_text "Oh great, another Monday."
python -m sarcastone.inference.predict_audio data/raw/audio/1_87.wav
python -m sarcastone.inference.predict_multimodal --text "Oh great." --audio data/raw/audio/1_87.wav
```

## Data

Decision record: [`docs/DATASETS.md`](docs/DATASETS.md).

1. **MUStARD++** (primary): 690 utterances from *Friends*, *The Golden Girls*, *The Big Bang Theory*, etc. — auto-downloaded by `download_mustard`.
2. **News Headlines v2** (text booster): ~28.6 k sarcasm headlines from Kaggle — optional pre-training corpus.
3. **Custom YouTube set** (stretch goal): 8 conversational videos → 944 utterance clips → 300-clip annotation sample with 3 annotators.

## Limitations

- Small test set (n=104) → wide confidence intervals; report both val and test.
- Speaker overlap across splits (speaker-independent split = future extension).
- `intensity_mean_db` dominance partly reflects per-show audio mastering, not pure speaker intent.
- All models are CPU-only (no CUDA).

## Project status

| Phase | Status | Key result |
|---|---|---|
| 1 — Text | Complete | RoBERTa boosted: F1 = 0.687 |
| 2 — Speech | Complete | 1D-CNN: F1 = 0.718 |
| 3 — Fusion | Code written, not yet run | Target: beat 0.718 |
| Annotation | Templates ready, awaiting 3 annotators | 300-clip sample, 90% audio↔text verified |

See [`progress.md`](progress.md) for the full chronological log.
