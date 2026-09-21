# Phase 1 Objectives — Text-Only Sarcasm Detection (FORMAL, decided)

**Status:** CONFIRMED — Option A locked in on 2026-09-21. Plain English first,
technical beneath. Everything below is measurable and each objective has an
explicit acceptance criterion. Phase 3 (fusion) remains **blocked** until
Phase 1 + Phase 2 objectives are fully closed.

---

## Starting state (verified, do not re-litigate)

| Item | Value | Source |
|---|---|---|
| Champion (single split) | RoBERTa-boosted, test F1 **0.687** / acc 0.692 | `reports/phase1_*` |
| Honest skill (5-fold CV) | mean **0.626**, 95% CI [0.595, 0.658] | `reports/phase1_cv.json` |
| Significance | no text contrast reaches p<0.05 at n=104 | `reports/phase1_significance.*` |
| Speaker-independent | 0.582 ≈ random control 0.585 | `reports/phase1_speaker_independent.*` |
| Gate | ≥ 0.70 — **NOT met** | PROJECT_OVERVIEW §4.6 |
| Locked test | 104 clips (of the original 690), seed 42 | `data/processed/splits/` |

**Core principle:** the locked test stays **untouched** for the entire phase.
Growth happens on the *training side only*. Nothing tunes on test.

---

## Objectives

### T1 — Finalise the data topology (THE decision)
Pick and lock exactly how MUStARD++ (690 vs full 1,202), PodSarc, and the
custom set are used. Two options (recommendation marked):

- **Option A (recommended): grow the training pool, keep the locked test.**
  MUStARD++ **full (1,202)** is the benchmark corpus. The locked 104-clip test
  stays exactly as-is; the **~512 extra clips + transcripts** join training.
  Train pool: 482 → ~994. PodSarc = pre-training booster + its own separate
  evaluation track. Custom 300 = out-of-domain final evaluation.
  *Pros: every past number stays valid; "does more data help?" measured
  cleanly on the same test. Cons: none material.*
- **Option B: full reset to 1,202.**
  New splits (~800/200/200), retrain text + all significance from scratch.
  *Pros: bigger test (200+) → tighter statistics. Cons: every number so far
  becomes void; days of CPU retraining.*

**Acceptance:** a written decision recorded here + in `docs/DATASETS.md`.

> **DECISION (2026-09-21):** **Option A.** MUStARD++ full (1,202) = benchmark
> corpus; locked 104-clip test unchanged; ~512 extra clips + transcripts join
> training. PodSarc = pre-training booster + separate evaluation track.
> Custom 300 = out-of-domain final evaluation.

### T2 — Assemble the expanded data
Download + verify the extra MUStARD++ clips (audio) and transcripts (CSV),
podcasts/annotations if we use PodSarc, into `data/raw/` per convention.
Verify: file counts, no corruption, IDs consistent.

**Acceptance:** `data/raw/mustard_pp/` (+ `podsarc/` if used) populated and
validated; registry updated.

### T3 — Build the expanded training set
Re-run split/build with the extra clips added to **train only**; locked
val/test identical to before. Emit a verification report proving val/test
byte-identical to the existing split, train enlarged.

**Acceptance:** verification command prints "val/test unchanged, train = N".

### T4 — Honest accuracy push on the same test
- **T4a:** 5-fold ensemble of RoBERTa on the expanded pool (legitimate
  variance reduction — the *same* skill, luck averaged out).
- **T4b:** one DeBERTa-base attempt (different pretraining, +capability probe).
- **T4c (optional, if decided in T1):** show a PodSarc-pretrained booster and
  report both its separate-track number and its effect on the MUStARD test.

**Acceptance:** all numbers reported on the locked test ONLY; fresh 5-fold CV
with CI for the best architecture; no test-tuning.

### T5 — Re-run the full rigor suite on the final text model
McNemar + paired bootstrap (all pairwise), 5-fold CV w/ CI,
speaker-independent split. Update all three report files.

**Acceptance:** `reports/phase1_{significance,cv,speaker_independent}.*` updated
to final model.

### T6 — Phase 1 gate verdict (honest)
Report final text F1 vs the 0.70 gate, with CI. If missed: document clearly,
justify why Phase 3 (fusion) is the intended source of the remaining gain.
**No number-fudging, no test-fitting.**

**Acceptance:** a one-paragraph verdict paragraph in `docs/PROJECT_OVERVIEW.md`
that a supervisor can trust.

### T7 — Colab notebooks (structured, error-free)
Production-grade notebooks that clone the GitHub repo and reproduce Phase 1:
- `notebooks/00_setup.ipynb` — repo clone, env, data fetch (public sources),
  verify splits.
- `notebooks/01_text.ipynb` — full text pipeline end-to-end
  (data → model ladder → final model → significance → CV), with the locked
  protocol enforced (test touched once).

Constraints: every cell runs top-to-bottom on a fresh Colab T4; no stale
numbers; all seeds fixed; paths/data resolved from within the notebook; a
"trust audit" section stating exactly what was reproduced vs downloaded.

**Acceptance:** notebooks execute end-to-end top-to-bottom on Colab (CPU
fallback paths included) with zero manual edits; all printed numbers match
`reports/phase1_*`.

### T8 — Documentation close-out
Update `docs/PROJECT_OVERVIEW.md` (§4), `docs/DATASETS.md`, `README.md` with
final Phase 1 numbers + topology. Commit-friendly state left ready.

**Acceptance:** docs reflect final numbers; git diff shows only intended files.

---

## Execution order (one at a time, each gated)
T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8

Each objective is started only when the previous one's acceptance criterion
is met and logged. No parallel jumping ahead.

---

## Progress log (updated 2026-09-21)

| Obj | Status | Evidence |
|---|---|---|
| T1 | DONE | Option A decision above |
| T2 | DONE | `data/raw/mustard_pp/` (6041-row CSV, 514-row expansion CSV, both HF zips); 1204 WAV + 1204 MP4 |
| T3 | DONE | `data/processed/splits_expanded/`; val/test byte-identical (SHA-256), train 482→996 (498/498) |
| T4a | DONE | `reports/phase1_roberta_boosted_exp_*` (single) + `reports/phase1_t4_colab_ensemble.json` (ensemble) |
| T4b | DONE (invalid) | DeBERTa predicted one class (F1 0.3333); tokenizer fix added to notebook |
| T4c | PENDING | PodSarc booster not yet run |
| T5 | PARTIAL | significance + CV + speaker-independent exist for champion; expanded-model variants optional |
| T6 | DONE | honest gate verdict written below + in `docs/PROJECT_OVERVIEW.md` |
| T7 | CODE DONE, Colab run PENDING | `notebooks/00_setup|01_text|02_speech|03_experiments.ipynb` |
| T8 | DONE | close-out battery + protocol check + error analysis in `reports/phase1_text_closeout.md`; overview/DATASETS/README updated |

### T4a finding (honest)
Running the champion recipe (roberta headlines-boosted, 5 ep / 3e-5) on the
expanded 996-clip train pool gave test macro-F1 **0.6664** (acc 0.6827) vs the
482-clip champion's **0.6866**. The change (−0.020) is **within the CV noise
band** (5-fold std 0.036 at n=104), so the honest statement is "no improvement".
Why: the 514 added clips are **off-distribution** relative to the locked test
(later Big-Bang-Theory seasons + Silicon Valley, whereas the locked test is
drawn from the original 690: Friends / classic BBT). Adding data shifted the
decision boundary unfavourably for the in-distribution test. Conclusion:
"more data" is not a free win here; the 0.70 gate is not reached this way.

### T4 (Colab GPU ensembles) + T6 gate verdict (honest)
Run on a Colab T4 (`notebooks/03_experiments.ipynb`), full results in
`reports/phase1_t4_colab_ensemble.json`:

| Comparison | dF1 | 95% CI | p (boot / McNemar) |
|---|---|---|---|
| E2 (expanded ensemble, 0.7017) vs champion (0.6866) | +0.015 | [-0.070, +0.102] | 0.359 / 1.000 |
| E2 vs E1 (locked ensemble, 0.6506) | +0.051 | [-0.031, +0.135] | 0.128 / 0.383 |
| E1 vs champion | -0.036 | [-0.134, +0.059] | 0.764 / 0.557 |

**Verdict:** the 0.70 gate is **not met in any statistically meaningful sense**.
No text configuration significantly beats another on the locked 104-clip test -
every paired-bootstrap 95% CI crosses zero. E2 (expanded 5-fold ensemble)
nominalises 0.7017, but it is statistically indistinguishable from the 0.6866
champion (p=0.36). E1's fold-mean 0.6239 reproduces the earlier 5-fold CV mean
0.626, so the honest text skill is **~0.62-0.65**. The bottleneck at this test
size is data/annotation noise, not backbone or training recipe. The intended
source of further gain is **multimodal fusion (Phase 3)**, not text tuning.
E3 (DeBERTa) is invalid and excluded (predicted a single class).

### T7 status (honest)
The notebooks are written and their Python logic is **locally validated**
end-to-end against the real repo (champion 0.6866, CNN 0.7180, BiGRU 0.5974,
CNN-vs-BiGRU p=0.0365 all reproduce exactly; locked test SHA unchanged after a
`make_splits()` re-derive; the 5-fold/DeBERTa trainer passes a smoke run).
- `notebooks/00_setup.ipynb`, `01_text.ipynb`, `02_speech.ipynb` - reproduce the
  committed Phase 1/2 results.
- `notebooks/03_experiments.ipynb` - the **T4a/T4b GPU experiments** (5-fold
  RoBERTa ensemble on locked + expanded pools, DeBERTa-v3-base probe). These are
  ~2 h on this CPU but ~10-15 min on a Colab T4, so they are meant to run there.

They have **not yet been executed on a Colab GPU**; that run is still pending
before T7 can be marked fully DONE.

### Post-T6 closing battery (2026-09-21, all NEGATIVE)
Three "no further text lever works" attempts, all on the locked 104-clip test,
before closing Phase 1 text (details: `reports/phase1_text_closeout.md`):
- **Context** (champion recipe + context): val 0.703 but test **0.638** (FN 23→29).
- **Emotion multitask** (Implicit_Emotion head, λ=0.1 / 0.5): test **0.626 / 0.639**.
- **Regularised fine-tune** (freeze 6, decay 0.85, early stop on val *loss*): test **0.619**.

Protocol check: our 0.687 macro-F1 is inside the published macro range for
text-only MUStARD++ (0.677–0.716, Bhosale et al. 2023); the headline 0.70 of the
original MUStARD++ paper is **weighted** F1. Champion error analysis (FN=23):
missed sarcasm is tone-driven (deadpan, rhetorical questions, elaborate
overstatement) — text-only cannot see it; fusion can. Phase 1 text is closed.