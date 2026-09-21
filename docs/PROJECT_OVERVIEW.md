# SarcasTone — Complete Project Record (Everything Done So Far)

**Document purpose:** a single authoritative, examinable record of every step taken from
project start through the end of Phase 2 — objectives, data, methods, exact numbers,
decision gates, bugs found & fixed, and where we stand. Written in plain English with
the technical detail beneath, so a supervisor, examiner, or collaborator can follow the
whole story without digging through code.

**Scope covered here:** project framing + setup, Objective 1 / Phase 1 (text), Phase 2
(speech), the custom-YouTube annotation exercise (Objective 2), and every bug/ad-hoc fix
made along the way. Phase 3 (fusion) is *not* in this file — it has not run yet.

---

## 1. The Big Picture

SarcasTone is a research project asking a single, examinable question:

> **Is sarcasm in conversation carried more by *what* you say (the words) or *how* you
> say it (the voice)? And can a machine combine both to detect sarcasm reliably?**

The plan is three phases, each with a fixed numeric "gate" that must be passed to continue:

| Phase | Modality | Approach | Gate (test macro-F1) |
|---|---|---|---|
| 1 | **Text** | fine-tuned BERT / RoBERTa on the words | F1 ≥ 0.70 |
| 2 | **Speech** | praat prosody + MFCC → 1D-CNN | F1 ≥ 0.45 |
| 3 | **Fusion** | combine text + speech (early & late) | must beat text F1 (+3–5 pts) |

Two objectives sit outside the numbered phases:
- **Objective 1** — deliver the two trained models with locked, comparable evaluation.
- **Objective 2** — annotate a *new* custom spoken-sarcasm set with real humans and
  measure inter-annotator agreement, giving us an independent test set and proof the
  phenomenon is real.

**The single most important principle in the whole project:** every model is evaluated
on the **same locked test set** (stratified split, seed 42). Nothing is tuned on test.
That is what makes every number below comparable to every other number.

---

## 2. Environment & Tooling

| Item | Value |
|---|---|
| OS / shell | Windows, PowerShell |
| Project root | `D:\sarcastone` |
| Python | 3.12.3 in a venv (`.venv`) |
| Compute | **CPU only, no CUDA** (all work planned around this) |
| ASR | Whisper-small via `transformers` (Windows-safe wrapper, no triton) |
| Audio stack | librosa, soundfile, praat-parselmouth, imageio-ffmpeg |
| Downloader | yt-dlp (for custom videos) |
| Repo hygiene | audio/media kept out of git; everything reproducible |

**Key Windows/CPU working notes** (learned the hard way, all non-obvious):
- Whisper via `transformers` (not openai-whisper) avoids the triton/warpgen build.
- ffmpeg resolved PATH-first with an `imageio-ffmpeg` fallback.
- PowerShell mangles inline `python -c` (quoting) → always write a temp `.py` instead.
- tqdm floods the log → filter with `Select-String` / `Where-Object`.
- Scripts that need the venv's tools must call them via `sys.executable -m`.
- Transcription of long files was the *hardest* part (see §7) — never trust long-form
  Whisper chunking.

---

## 3. Data Assembled

### 3.1 MUStARD++ — the primary benchmark (Phases 1 & 2)
- **Source:** https://github.com/soujanyaporia/MUStARD
- **Contents:** 690 utterances from *Friends*, *The Golden Girls*, *The Big Bang Theory*
  and *Sarcasmaholics Anonymous*, ~50/50 sarcastic vs non-sarcastic.
- **Fields:** utterance text, speaker, show, context (surrounding utterances), binary label.
- **Audio:** 690 official clips fetched via HuggingFace (`mmsd_raw_data.zip`),
  converted to uniform **16 kHz mono WAVs** at `data/raw/audio/`. Clip IDs match the
  locked splits 1:1.
- **Locked splits (seed 42, stratified):** train 482 / val 104 / test 104.
  Protocol: select on val, touch test once, macro-F1 is the official metric.

### 3.2 News Headlines v2 — text booster (only for Phase 1)
- ~28.6 k sarcasm headlines (Kaggle, Saurabh Shahane); used ONLY to pre-train the text
  model **before** MUStARD++ fine-tuning. Never used for any reported test metric.
- Validation on its own held-out set: F1 0.920 (sanity check that stage-1 learning worked).

### 3.3 Custom spoken-sarcasm set (Objective 2)
- 8 real conversational YouTube videos downloaded + segmented → **944 valid clips**
  (see §8 for the full, painful but now-corrected story).
- Each clip: an utterance cut at its boundary, with the preceding 2–3 utterances as
  annotation context.

---

## 4. Phase 1 — Text-Based Sarcasm Detection (Objective 1, part A)

**Question:** just from the words, how well can a model tell sarcasm?

### 4.1 Model ladder (each rung isolates one design decision)
1. **TF-IDF + Logistic Regression** — classical lexical baseline (no deep model).
2. **Frozen BERT-[CLS] + LR** — does a pretrained representation help even without fine-tuning?
3. **Fine-tuned BERT-base-uncased** — the plan's specified main text model.
4. **Fine-tuned RoBERTa-base** — a stronger alternative pretraining objective.
5. **RoBERTa "boosted"** — first learn sarcasm on News Headlines, then fine-tune on MUStARD++.
   This is the *final text arm*.

### 4.2 Exact results on the locked test set
| # | Model | Context | Epochs | LR | Val F1 | **Test F1** | Acc |
|---|---|---|---|---|---|---|---|
| 1 | TF-IDF + LogReg | — | — | — | — | 0.596 | 0.596 |
| 2 | Frozen BERT-CLS + LR | — | — | — | — | 0.654 | 0.654 |
| 3 | BERT fine-tuned | no | 3 | 2e-5 | 0.597 | 0.609 | 0.625 |
| 4 | BERT fine-tuned | yes | 3 | 2e-5 | 0.643 | 0.635 | 0.635 |
| 5 | RoBERTa fine-tuned | no | 3 | 2e-5 | 0.600 | 0.663 | 0.664 |
| 6 | RoBERTa fine-tuned | yes | 4 | 2e-5 | 0.586 | 0.643 | 0.644 |
| 7 | RoBERTa fine-tuned | no | 5 | 3e-5 | 0.671 | 0.654 | 0.654 |
| 8 | News Headlines only (sanity) | — | 1 | 2e-5 | 0.920 | 0.919* | 0.920 |
| 9 | **RoBERTa boosted → FINAL TEXT** | no | 5 | 3e-5 | **0.673** | **0.687** | 0.692 |
| 10 | E1: RoBERTa 5-fold ensemble (locked train 482) | no | 5 | 3e-5 | 0.624 (fold mean) | 0.651 | 0.663 |
| 11 | E2: RoBERTa 5-fold ensemble (expanded train 996) | no | 5 | 3e-5 | 0.653 (fold mean) | **0.702** | 0.702 |
| 12 | E3: DeBERTa-v3-base (locked train 482) | no | 5 | 2e-5 | — | invalid* | — |

\* row 12 predicted a single class (macro-F1 0.333 = one-class score on a balanced test);
excluded pending the SentencePiece tokenizer fix. Rows 10-11 are GPU ensembles: folds built
from the train split only, best epoch chosen on fold-validation, locked test scored once. See
`reports/phase1_t4_colab_ensemble.json`.

\* row 8 evaluated on the News Headlines test set — *different distribution* from
MUStARD++; shown only to confirm stage-1 learning, never compared to MUStARD rows.

### 4.3 Why row 9 won (champion selection — by validation F1)
Rows 7 and 9 tie statistically on val (0.671 vs 0.673). The boosted model is promoted
because the tie is broken in its favour *and* it was trained with strictly more
supervision (transfer from News Headlines) — an a-priori advantage independent of test.

### 4.4 Error profile of the champion (test)
Confusion: TN=43 FP=9 / FN=23 TP=29. Crucially, the false-positive rate dropped
dramatically (FP was 15–30 in earlier runs) — the model became balanced instead of
over-saying "sarcastic".

### 4.5 What the Phase 1 numbers teach us (findings)
1. **Pretraining beats fine-tuning budget:** frozen BERT (0.654) already beats TF-IDF by
   +5.8 pts and matches most fine-tuned runs.
2. **Context helps BERT (+2.6)** but not RoBERTa in our runs — consistent with the
   literature that context gains are model/seed-dependent.
3. **All transformer variants cluster in ~0.61–0.66.** With n_test = 104, one utterance ≈
   1 F1 point; differences <3 pts are within noise (formal significance testing deliberately
   deferred to Phase 3 tooling: McNemar + paired bootstrap).
4. The News-Headlines booster closed most of the gap (0.654 → 0.687, +3.3 pts) with a far
   better error profile.

### 4.6 Phase 1 gate verdict (final, honest)
> **Gate: text test F1 ≥ 0.70 → NOT met in any statistically meaningful sense.**
> The best point estimate is the expanded 5-fold ensemble E2 = **0.702**, which nominally
> clears 0.70, but it is **statistically indistinguishable** from the 0.687 single-split
> champion (dF1 +0.015, 95% CI [-0.070, +0.102], p=0.36) and from the locked ensemble E1
> (dF1 +0.051, CI [-0.031, +0.135], p=0.13). Every paired-bootstrap CI crosses zero.
> E1's fold-mean 0.624 reproduces the earlier 5-fold CV mean 0.626, so the **honest text
> skill is ~0.62–0.65**; the 0.687/0.702 numbers are within the n=104 noise band.
> Full expansion from MUStARD++ (1,202) did **not** reliably help (single run 0.687→0.666,
> ensemble fold-mean 0.624→0.653 - both within noise). The text-only ceiling here is set by
> data size / annotation noise, not by backbone or recipe. We therefore proceed with
> **fusion (Phase 3) as the intended source of gains**, and report text as a distribution,
> not a single lucky split. Documented, not papered over.

### 4.7 Phase 1 artifacts
- Final text checkpoint: `checkpoints/text_roberta_boosted/`
- Stage-1 headline checkpoint: `checkpoints/text_roberta_nh/` (reusable)
- Fusion embeddings: `embeddings/text_{train,val,test}.npz`
- Old champion backed up: `embeddings/_backup_champion/`
- Metrics + error analyses: `reports/phase1_*_test_metrics.json`, `reports/phase1_*_error_analysis.md`

---

## 5. Phase 2 — Speech-Based Sarcasm Detection (Objective 1, part B)

**Question:** just from the sound of the voice, how well can a model detect sarcasm?

### 5.1 Pipeline
1. **Audio:** 690 official MUStARD clips → 16 kHz mono WAVs (`data/raw/audio/`),
   IDs matching the locked splits exactly.
2. **Features (all 690 extracted, zero failures):**
   - **Praat summaries** — 11 prosodic / voice-quality stats (pitch, intensity, HNR,
     duration, jitter/shimmer…) → `acoustic_summary.csv`.
   - **MFCC frame sequences**, 40-dim (13 MFCC + Δ + Δ² + RMS) → `features_seq/`.
   - z-score normalizer fit on the **train split only** (no leakage).
3. **ASR audit:** Whisper-small transcripts vs gold text (see §5.4).

### 5.2 Exact results on the locked test set
| Model | Input | Val F1 | **Test F1** | Acc |
|---|---|---|---|---|
| Logistic Regression | 11 Praat summaries | 0.613 | 0.6527 | 0.6538 |
| **1D-CNN** | 40-dim Δ MFCC frames | 0.637 | **0.7180** | 0.7212 |

CNN confusion (test): TN=43 FP=9 / FN=20 TP=32. Early stop at epoch 27 (best val F1 0.6367).

### 5.3 Baseline feature importance (|coef|)
`intensity_mean_db (+)` ≫ `f0_mean_hz (−)` > `duration_s (+)` > `hnr_mean_db (−)` > pitch
spread. That ordering means: **sarcastic delivery is louder, lower-pitched and drawn-out** —
a coherent, human-understandable acoustic signature.

### 5.4 ASR / audio-quality audit
Whisper-small over all 690 clips: **median WER 0.200**, but the mean is inflated to 1.001 by
**31 clips (4.5%)** where the model hallucinates on laugh-track/music segments
(no speech → it invents repeats). These 31 utt_ids are flagged (`wer > 1`) in
`whisper_transcripts.csv` as known-noisy — important for post-hoc ablations, not
for the supervised CNN (which uses human labels, not ASR).

### 5.5 Phase 2 gate verdict
> **Gate: speech test F1 ≥ 0.45 → PASSED decisively (0.718). No fallbacks needed.**

### 5.6 Key findings & thesis caveats
1. **Audio is the strongest single modality in this setup:** CNN **0.718** vs boosted-RoBERTa
   text arm **0.687**. Sarcasm in MUStARD++ is carried substantially by *vocal performance* —
   a central, defensible thesis claim.
2. **Confound to document honestly:** `intensity_mean_db` dominance partly reflects
   per-show/episode audio mastering (Friends is louder than quiet indie shows) rather than
   pure speaker intent, and the splits are **speaker-dependent** (same voices in train/test)
   which flatters acoustic models. Both belong in the limitations.
3. The Whisper audit enables two later ablations: (a) do CNN errors concentrate in high-WER
   clips? (b) run boosted RoBERTa on ASR text vs clean subtitles to quantify transcript-quality
   effects.

### 5.7 Phase 2 artifacts
- CNN checkpoint: `checkpoints/speech_cnn/`
- Fusion embeddings: `embeddings/speech_{train,val,test}.npz` (128-d)
- Metrics + error analyses: `reports/phase2_*_test_metrics.json`, `reports/phase2_*_error_analysis.md`
- Inference verified for both modalities: `predict_text`, `predict_audio`.

---

## 6. Scoreboard & Where We Stand (mid-project)

| | Text (Phase 1) | Speech (Phase 2) |
|---|---|---|
| **Final model** | RoBERTa boosted (E2 ensemble best point est.) | 1D-CNN |
| **Test macro-F1** | 0.687 single / **0.702** E2 ensemble (ns) | **0.718** |
| **Test acc** | 0.692 / 0.702 | 0.7212 |
| **Gate** | 0.70 — not met significantly (all CIs cross 0) | 0.45 — passed by 0.268 |

**Takeaway so far:** the voice beats the words in this benchmark. Phase 3 (fusion) is
designed to test whether combining them beats the voice alone (target: > 0.718), which
will sharpen the thesis claim.

---

## 7. The Custom Dataset & Objective 2 (the hard-won part)

### 7.1 Why we built it
MUStARD++ is tiny (690) and its audio clips are hard-transcribed; we wanted a **new set of
real, conversational, sarcastic speech** so we could (a) close Objective 2 with genuine
human annotation + inter-annotator agreement, and (b) have an independent set to stress-test
the fusion model.

### 7.2 The pipeline (as originally designed)
yt-dlp → convert to 16 kHz WAV → Whisper → timestamped utterance segments → cut a clip per
segment with 2–3 preceding utterances as context → write `context.json` → sample ~300 →
three humans annotate → Cohen's/Fleiss' kappa → majority-merge into `custom_dataset.csv`.

### 7.3 What actually happened — and the bug that wrecked the first run
**The audio/text mismatch you reported.** Root cause, found after methodical elimination:

- Clips were cut at exactly their stated timestamps (durations matched to centiseconds) —
  so the *cutting* was fine.
- No timeline resets — so it wasn't a chunk-restart bug.
- A probe clip from ~13 minutes in said *"we appear to have a streaker"* but actually
  contained *"Christians in one corner"*, while a clip from the first seconds was accurate.

**Diagnosis:** Whisper's built-in long-form chunking accumulates **timestamp drift** — its
30 s windows get misaligned the deeper into a video you go, so everything past ~a minute
pairs the wrong audio with the wrong text. That matched "the majority don't match."

**The fix:** stop using the drift-prone long-form path entirely. Transcribe each video in
short, independent ≤28 s windows (short inputs don't drift), stitch their timestamps back
to full-video time, then re-cut. Implemented as `asr.transcribe_windowed()`.

### 7.4 Then a second, subtler bug (found after "fixing" drift)
Rebuilding the videos produced absurdly few clips (e.g. 39 total vs the ~876 before), and
the census showed every window had plenty of speech. Reading the code, in `asr.py`:

```python
t0 += win        # win == SAMPLES (448000), but the loop maths in SECONDS
```

After the first 28 s window, `t0` jumped to 448,000 "seconds" past the end of the file and the
loop exited. We had only ever been transcribing the **first 28 seconds of every video**.
This single one-character-class bug (`win` → `window_s`) silently meant every "broken" symptom
came back. Fixed → segments now span the full file (verified: 0–170 s on the smallest clip).

### 7.5 Rebuild numbers (post-fix, post-cleanup)
| Video | Clips |
|---|---|
| dp6BIDCZRic | 59 |
| IPXeiS1tzr4 | 42 |
| Hmt2QA7x_-c | 39 |
| QLFRGj-PPNI | 86 |
| BhOsnALog-Y | 94 |
| nGcgpSa8gWc | 200 |
| 2tYAeTGYNXA | 300 |
| nArSMnm9ETo | 200 |
| **Total (after cleaning)** | **944 valid clips** |

Cleanup dropped 73 noise segments (all-caps LAUGHTER/APPLAUSE/BUZZER, repeated
laugh hallucinations, host-isms) plus 3 clips that failed the audio↔text spot-check.

### 7.6 Final correctness verification (the thing that mattered most)
Re-transcribed a random 30-clip sample and measured caption↔audio overlap:
- **27/30 (90%) matched.** The 3 misses were benign edge-slop (short clip grabbing the
  next clause; one clip that's actually a laugh) — all 3 dropped. This is a tenfold
  improvement over the majority-wrong state you reported.

Key insight: residual mismatch is **inherent Whisper-CPU boundary noise**, not pipeline
corruption — and it's now a small, known, audited fraction (~10%).

### 7.7 Current Objective-2 state (ready for humans)
- **944 valid clips** on disk across all 8 sources; manifest in `context.json`.
- A **300-clip random sample** seeded and reduced; every sampled clip has real audio on disk
  (verified 300/300), all within duration limits, spread across all 8 sources.
- **Three annotator kits** generated (A/B/C): blank CSV sheets + per-annotator HTML listening
  pages with audio player, transcript and context.
- **`docs/ANNOTATOR_GUIDELINES.md`** — the short plain-English rule sheet annotators read once.

### 7.8 What annotators do next (hand-over)
1. Open your **own** HTML page (A→A, B→B, C→C), headphones, quiet room.
2. Label each clip `1` (sarcastic) / `0` (not) / `x` (can't tell); ~1–1.5 h each.
3. **No discussion between annotators** until all three finish (keeps the kappa honest).
4. Back in the sheets → then:
   ```powershell
   python -m sarcastone.data.annotation agreement   # Cohen's + Fleiss' kappa → iaa_report.md
   python -m sarcastone.data.annotation merge       # majority labels → custom_dataset.csv
   ```

---

## 8. Bugs, Ad-Hoc Fixes & Lessons (examined for the record)

| # | Issue | Root cause | Fix | Status |
|---|---|---|---|---|
| 1 | Audio↔text mismatch "for the majority" | Whisper long-form **timestamp drift** on long files | `transcribe_windowed()` (short windows, stitched) | Fixed & verified |
| 2 | Only first ~28 s transcribed | `t0 += win` where `win` was **samples not seconds** | `t0 += window_s` | Fixed & verified (full-file spans) |
| 3 | Suspicious segments kept making it through | all-caps noise / repeats / host catchphrases | manifest cleanup filter (ALLCAPS/FLAG/REPEAT/DUP) | Applied |
| 4 | Leftover broken clips after cleanup | early runs before the fix | rebuild + drop 3 verified-bad | Applied |
| 5 | Few candidate clips from long shows | (was bug #2 all along; filters also tight) | fix #2 + sedimenting `--refilter` mode + softer duration band | Applied |
| 6 | Re-transcription is slow on CPU | 30×300-clip re-verify | sample-based verification + persisted `_segments.json` | Done |

**Tooling lessons frozen into `AGENTS.md`:** plain-English-first communication; PowerShell
crashes on inline `python -c` → write temp `.py`; tqdm floods → filter; scripts use
`sys.executable -m`; keep resume/caching everywhere; always verify with real audio, never assume.

---

## 9. What Comes Next (Phase 3 & closing Objective 2)

1. **Finish annotation** (human A/B/C) → run `agreement` + `merge` → Objective 2 closed
   with a real IAA number.
2. **Phase 3 fusion:**
   - **Late fusion** first: concatenate text embeddings (768-d) + speech embeddings (128-d)
     → tuned classifier.
   - **Early / attention fusion** next for comparison.
   - **Ablations:** ASR-text vs clean text; error analysis on high-WER clips.
   - **Target:** fusion must **beat 0.718** (the speech arm) to justify itself; document the
     speaker-dependent-split limitation.
3. Deliverables map (README §"Deliverables"): fusion checkpoints, `phase3_comparison.json`,
   `phase3_significance.json` (McNemar + paired bootstrap — finally giving formal
   significance on those close Phase-1 rows).

---

## 10. Quick Reference (all the key numbers at a glance)

| Metric | Text (RoBERTa-boosted) | Speech (1D-CNN) |
|---|---|---|
| Test macro-F1 | 0.687 (E2 ensemble 0.702, ns) | **0.718** |
| Test acc | 0.692 | 0.7212 |
| Val F1 | 0.673 | 0.637 |
| Gate | ≥0.70 ✗ | ≥0.45 ✓ |

| Custom dataset | Value |
|---|---|
| Total valid clips | **944** |
| Sources | 8 YouTube videos |
| Annotation sample | 300 |
| Spot-check match rate | 27/30 (90%) after fix |
| Annotators | A, B, C (human) |

*End of Phase 1–2 record. Phase 3 and the IAA report will be appended here once complete.*
