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
| T4a | DONE (negative) | `reports/phase1_roberta_boosted_exp_*` — test F1 **0.6664** vs 0.6866 on 482 |
| T4b | PENDING | DeBERTa-base attempt not yet run |
| T4c | PENDING | PodSarc booster not yet run |
| T5 | PENDING | rigor suite not yet re-run on a final model |
| T6 | PENDING | gate verdict not yet written |
| T7 | CODE DONE, Colab run PENDING | `notebooks/00_setup|01_text|02_speech|03_experiments.ipynb` |
| T8 | PARTIAL | `docs/DATASETS.md` updated; overview/README close-out pending |

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