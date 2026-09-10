# PHASE 1 - text-based sarcasm detection (Weeks 1-4)
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "== [1/4] Download MUStARD++ annotations ==" -ForegroundColor Cyan
python -m sarcastone.data.download_mustard

Write-Host "== [2/4] Locked splits (seed=42) + preprocessing ==" -ForegroundColor Cyan
python -m sarcastone.data.preprocess_text

Write-Host "== [3/4] Week-2 baselines (TF-IDF + frozen BERT-CLS, LogReg) ==" -ForegroundColor Cyan
python -m sarcastone.models.text_baseline

Write-Host "== [4/4] Fine-tune BERT + extract embeddings for fusion ==" -ForegroundColor Cyan
python -m sarcastone.models.bert_finetune --config configs/text_bert.yaml --extract_embeddings

Write-Host "Decision gate: test F1(macro) >= 0.70?  -> reports/phase1_bert_test_metrics.json" -ForegroundColor Green
