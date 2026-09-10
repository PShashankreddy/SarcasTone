# PHASE 3 - multimodal fusion + comparison (Weeks 6-8)
# Prereqs: Phase 1 + 2 completed WITH --extract_embeddings on both baselines.
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "== [1/3] Early fusion (MLP on concatenated embeddings) ==" -ForegroundColor Cyan
python -m sarcastone.models.fusion --mode early

Write-Host "== [2/3] Late fusion (soft voting) + comparison + significance ==" -ForegroundColor Cyan
python -m sarcastone.models.fusion --mode late --compare

Write-Host "== [3/3] Done. Check reports/phase3_*, phase3_comparison*.json ==" -ForegroundColor Green
Write-Host "Success criteria: fusion F1 > text F1 (target +3-5 pts), McNemar p < 0.05"
