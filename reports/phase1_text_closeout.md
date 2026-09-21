# Phase 1 text — close-out experiments (2026-09-21)

Reference champion (locked single split, no context, best-val selection):
**val_F1 0.673, test macro-F1 0.6866, acc 0.692.**

## 1. Context-aware (champion recipe + `--use_context`)
| | val_F1 | test F1 | acc |
|---|---|---|---|
| champion (no ctx) | 0.673 | **0.6866** | 0.692 |
| + context | 0.703 | **0.6376** | 0.654 |

Val rose, test dropped 5 pts (FN 23 → 29). **NEGATIVE** — context as fed here
hurts; likely 128-token truncation pushes the target utterance out + fits val.
Not promoted.

## 2. Multitask with in-domain emotion labels
Shared RoBERTa (init `text_roberta_nh`) + sarcasm head + Implicit_Emotion head
(10 classes, all 1,202 rows labelled).
| config | test F1 |
|---|---|
| λ_aux = 0.5 | 0.6386 |
| λ_aux = 0.1 | 0.6259 |

**NEGATIVE** — the auxiliary head competes with the target (val F1 ~0.62 vs 0.67).
Model: `src/sarcastone/models/multitask.py`.

## 3. Overfitting-focused fine-tuning
Freeze 6/12 RoBERTa layers + layer-wise lr decay (0.85) + early stop on **val loss**
(patience 2). Weights restored at best val loss (epoch 4) → **test F1 0.6190**.
**NEGATIVE** — on this task val loss and val F1 diverge; early-stopping on loss
underfits vs the champion's best-val-F1 selection. Variance-narrowing not worth
the loss here. Model: `src/sarcastone/models/finetune_regularized.py`.

## 4. Protocol comparison vs published MUStARD++ text numbers
Our protocol: **macro-F1, one seed-42 split, locked train 482 (of 690), no context.**
Published:
- Ray et al. 2022 (MUStARD++ paper): text BART F1 ≈ **67.7–70.0** and a headline
  **70.2**, but reported as **weighted** F1.
- Bhosale et al. 2023 (Sarcasm in Sight and Sound): **macro-F1, 5-fold average,
  full 1,202, speaker-dependent** — BART text **0.677** (no ctx) / **0.692** (ctx);
  ViFiCLIP text encoder **0.716** (no ctx) / **0.719** (ctx).

Reading: our 0.6866 is **inside the published macro-F1 range** (0.677–0.716).
The nominal "0.70–0.75" figures are (a) weighted F1, and/or (b) 5-fold averaged
with context on all 1,202. The gap to the 0.70 gate is therefore **partly a
protocol difference, not purely a model deficit.**

Speaker caveat: published speaker-independent text ≈ 67.7 (only ~1.6 pts below
speaker-dependent). Our in-house speaker-independent split scored 0.5816 ≈ random
(0.5845) → our worksheet construction is harsher (effectively show-independent).
Design of that stress test must be spelled out in the final report.

## 5. Error analysis (champion test, FN = 23 / FP = 9)
Missed sarcasm concentrates in **tone-driven** forms that text cannot see:
- deadpan one-liners: *"Nooo!"*, *"Excellent hole, Joe."*, *"Almost! But first, we gotta start."*
- rhetorical questions: *"Are you still enjoying your nap?"*, *"And protected them from a tornado?"*, *"Why? Because she can sing and play guitar and do both at the same time?"*
- elaborate overstatement that reads literally off the page: *"Obviously, waitressing at The Cheese Cake Factory is a complex socio-economic activity…"*

False positives mirror this (plain sincere questions like *"Phœbe?"* flagged
sarcastic). Conclusion: a genuine, documentable limit of text-only — the
disambiguation lives in **prosody** and prior dialogue. This is exactly the
limitations analysis Objective 4 asks for, and it motivates Phase 3 fusion.

## 6. Verdict — Phase 1 text CLOSED
No further text-only lever (context, emotion multitask, regularised training)
moved the locked test beyond the champion within noise. Robust text skill:
5-fold CV 0.626 ± 0.036 (CI [0.595, 0.658]); E2 expanded 5-fold ensemble gives
the best point estimate 0.7017 (p=0.36 vs champion, not significant). Recorded
headline: champion **0.6866** macro-F1; compliments-of-protocol note in §4.