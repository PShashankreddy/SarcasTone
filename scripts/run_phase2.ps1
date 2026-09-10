# PHASE 2 - speech-based sarcasm detection (Weeks 4-6)
# Prereqs: Phase 1 splits exist; audio clips collected in data/raw/audio_raw/
#          and ffmpeg installed (for conversion + Whisper).
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "== [1/5] Convert clips to 16 kHz mono wav ==" -ForegroundColor Cyan
python -m sarcastone.features.convert_audio

Write-Host "== [2/5] Build acoustic features (Praat summaries + MFCC sequences) ==" -ForegroundColor Cyan
python -m sarcastone.features.build_features

Write-Host "== [3/5] Whisper transcripts (optional but recommended) ==" -ForegroundColor Cyan
python -m sarcastone.features.whisper_transcribe --model small

Write-Host "== [4/5] Week-5 baseline: Praat summary features + LogReg ==" -ForegroundColor Cyan
python -m sarcastone.models.speech_cnn --baseline

Write-Host "== [5/5] Train 1D-CNN (+ BiGRU variant) + embeddings for fusion ==" -ForegroundColor Cyan
python -m sarcastone.models.speech_cnn --extract_embeddings
# python -m sarcastone.models.speech_rnn --extract_embeddings   # optional alternative

Write-Host "Decision gate: speech test F1 >= 0.45? -> reports/phase2_*_test_metrics.json" -ForegroundColor Green
Write-Host "If < 0.45: check f0_p05 == 0 Hz outliers in data/processed/acoustic_summary.csv"
