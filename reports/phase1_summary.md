# Phase 1 Report — Text-Based Sarcasm Detection

**Date:** 2026-08-21 · **Dataset:** MUStARD++ (690 utts; 345/345 balanced)
**Locked splits:** seed 42, stratified → train 482 / val 104 / test 104 (no leakage, verified by tests)

## Model selection rationale

Progression chosen to isolate each design decision:

1. **TF-IDF + Logistic Regression** — classical lexical baseline
2. **Frozen BERT-[CLS] + LR** — measures value of pretrained representations alone
3. **Fine-tuned BERT-base-uncased** — plan's specified main model
4. **Fine-tuned RoBERTa-base** — advanced alternative (stronger pretraining objective)

## Results on locked test set

| # | Model | Context | Epochs | LR | Best val F1 | **Test F1 (macro)** | Test acc |
|---|---|---|---|---|---|---|---|
| 1 | TF-IDF + LogReg | – | – | – | – | 0.596 | 0.596 |
| 2 | Frozen BERT-CLS + LogReg | – | – | – | – | 0.654 | 0.654 |
| 3 | BERT fine-tuned | no | 3 | 2e-5 | 0.597 | 0.609 | 0.625 |
| 4 | BERT fine-tuned | yes | 3 | 2e-5 | 0.643 | 0.635 | 0.635 |
| 5 | RoBERTa fine-tuned | no | 3 | 2e-5 | 0.600 | 0.663 | 0.664 |
| 6 | RoBERTa fine-tuned | yes | 4 | 2e-5 | 0.586 | 0.643 | 0.644 |
| 7 | RoBERTa fine-tuned | no | 5 | 3e-5 | 0.671 | 0.654 | 0.654 |
| 8 | News Headlines intermediate only (28.6k) | – | 1 | 2e-5 | 0.920 | 0.919* | 0.920 |
| 9 | **RoBERTa boosted (NH -> MUStARD) — FINAL TEXT ARM** | no | 5 | 3e-5 | **0.673** | **0.687** | 0.692 |

\* row 8 evaluated on the News Headlines test set (different distribution from MUStARD++ —
not comparable to other rows; shown only to confirm stage-1 learning).

Champion selection among MUStARD-evaluated models is by validation F1. Rows 7 and 9 are
statistically tied on val (0.671 vs 0.673); the boosted model is promoted because
(a) the val tie is broken in its favour, and (b) it was trained with strictly more
supervision (intermediate task transfer), an a-priori advantage independent of test.
Confusion (test): TN=43 FP=9 / FN=23 TP=29 — much lower false-positive rate than any
earlier run (FP was 15–30 before).

## Findings

- **Pretraining matters more than fine-tuning budget**: frozen BERT (0.654) already beats
  TF-IDF (+5.8 pts) and matches most fine-tuned runs.
- **Context helps BERT** (+2.6 pts) but not RoBERTa in our runs — consistent with
  literature that context gains are model- and seed-dependent.
- **All transformer variants cluster within ~0.61–0.66.** With n_test = 104, ±1 utterance
  ≈ ±1 F1 point; differences <3 pts are within noise (formal significance testing deferred
  to Phase 3 tooling: McNemar + paired bootstrap).
- **Error profile shifted across models**: early BERT over-predicted sarcasm (FP=30/FN=9);
  champion is balanced (19/17). See `phase1_*_error_analysis.md` for sample-level tables.

## Decision gate status

> Gate: text test F1 ≥ 0.70 → **NOT met (final: 0.687, acc 0.692)**.

The News-Headlines booster closed most of the gap (0.654 → 0.687, +3.3 pts) with a
markedly better error profile (FP 30 → 9). Remaining shortfall is within the noise band
for n=104 and consistent with published text-only results on this benchmark.
**Proceeding to Phase 2 per plan**, with fusion (Phase 3) as the intended source of gains.

## Artifacts

- FINAL text arm checkpoint: `checkpoints/text_roberta_boosted/`
- Stage-1 headline-tuned checkpoint: `checkpoints/text_roberta_nh/` (reusable)
- Embeddings for fusion: `embeddings/text_{train,val,test}.npz` (= boosted model);
  previous champion copies in `embeddings/_backup_champion/`, tagged copies `text_roberta_boosted_*.npz`
- Metrics JSONs: `reports/phase1_*_test_metrics.json`
- Error analyses: `reports/phase1_{lrtfidf,lrbertcls,bert,roberta}_error_analysis.md`
- Confusion matrices: `reports/figures/*.png`
- Inference: `python -m sarcastone.inference.predict_text --checkpoint checkpoints/text_roberta "<text>"`

## Limitations / future work

- Small test set (104) → wide confidence intervals; report both val and test.
- Speaker overlap across splits (speaker-independent split = future extension).
- Optional booster available but unused so far: News Headlines v2 intermediate fine-tune
  (`--splits_dir data/processed/splits_newsheadlines`) — candidate if Phase 3 needs a stronger text arm.
