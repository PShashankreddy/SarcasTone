# Phase 1 Rigor Gap 3: Speaker-Independent Split

**Q: Does the text model generalize to speakers never seen in training?**

## Design
- Candidate held-out sets: all combinations of >=3 speakers with >=10 clips each.
- Chosen set maximised test size while keeping sarcasm rate closest to 50%.
- **Held-out speakers:** CHANDLER, HOWARD, PERSON, JOEY, RACHEL, RAJ (344 clips, sarcasm rate 50.0%).
- These speakers are **absent from train** (0 leakage). Caveat: `PERSON` is a
  generic on-screen label shared across shows, so nominal-identity disjointness
  is inherently imperfect.
- Model: RoBERTa-base, same recipe as the 5-fold CV (5 epochs, lr 3e-5, bs 16, seed 42).
- Control: identical run but with a **random** same-size split.

## Results (held-out test, n=344)
| Evaluation | F1 | Acc | P | R |
|---|---|---|---|---|
| Speaker-independent | 0.582 | 0.590 | 0.598 | 0.590 |
| Random-split control | 0.585 | 0.596 | — | — |

Confusion on held-out speakers: TN=126, FP=46, FN=95, TP=77.

## Interpretation
- Performance on never-seen speakers is essentially **identical to the random
  control** (0.582 vs 0.585 F1). No catastrophic collapse on unseen voices.
- The speaker-independent number sits inside the 5-fold CV interval
  **[0.595, 0.658]** (slightly below mean), i.e. held-out speakers cost us
  roughly 0.04 F1 vs the average random fold — a mild, not catastrophic, drop.
- This suggests text-based sarcasm cues are *partly* speaker-general, with a
  small speaker-specific component the model leans on when available.