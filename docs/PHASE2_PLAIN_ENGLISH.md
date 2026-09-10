# Phase 2, Explained in Plain English

A study companion for the speech-based sarcasm detection phase.
Read this before writing your thesis chapter or facing questions about Phase 2.

---

## The whole phase in one breath

> We turned 690 TV clips into voice measurements, let two computer "students" learn
> from them (a simple one scored 65%, a smarter pattern-spotter scored 72% on clips
> it had never seen), and double-checked the recordings themselves were clean.
> Conclusion: **you can hear sarcasm without reading the words — the voice alone
> scores higher than the words alone (72% vs 69%).**

---

## 1. What were we trying to do?

Teach a computer to guess whether someone is being sarcastic **from their voice alone**
— not from the words they say, but from *how* they say them.

## 2. The materials

- 690 short recordings of TV actors delivering one line each (*Friends*, *Big Bang
  Theory*, *Golden Girls*, *Sarcasmaholics*).
- Each clip has a known correct answer ("sarcastic" / "not sarcastic"), decided by
  human annotators when the dataset was built. Think: 690 labeled flashcards.
- Split: 482 for studying (train), 104 practice exams (val), 104 final exam (test).

## 3. Step 1 — Turning sound into numbers

Computers cannot "listen". They need numbers. For every clip we measured things a
voice coach would notice:

| Measurement (jargon) | Plain meaning | Why it matters for sarcasm |
|---|---|---|
| duration | how long the clip is | Sarcastic delivery is often drawn out ("riiiight") |
| pitch (f0) mean/std/range | how high/low the voice is, and how much it wanders | Deadpan = flat; exaggerated irony = big swings |
| intensity mean/std | how loud, and how much loudness varies | Mock enthusiasm = louder, more variable |
| HNR | how clean vs breathy/croaky the voice sounds | Flat affect vs bright tone |
| jitter / shimmer | tiny wobbles in pitch/volume cycle-to-cycle | Voice tension vs relaxation |
| MFCCs | a detailed "sound fingerprint" every 20 milliseconds | Lets the computer see patterns, not just averages |

Two toolkits did the measuring:
- **Praat** (via `parselmouth`) — the software real phonetics scientists use.
- **librosa** — a Python library that produced the frame-by-frame fingerprints.

Everything was standardized first: all clips converted to one audio format
(16 kHz mono WAV), and measurements were scaled consistently using statistics from
the training clips only (so no exam answers could leak into preparation).

## 4. Step 2 — Two kinds of students

| Student | Jargon name | What it gets to see | Score |
|---|---|---|---|
| Honest student | Logistic Regression (our "baseline") | Only the 11 summary measurements | 65% |
| Pattern-spotter | 1D-CNN (convolutional neural network) | The full frame-by-frame fingerprints | **72%** |

- The **baseline** exists as a sanity check: if the fancy model can't beat the simple
  one, the fancy model is worthless.
- A **CNN** is a pattern detector — same family of technology as face recognition,
  but scanning across time instead of across an image. It asks: *"did a
  sarcasm-marker pattern fire anywhere in this clip?"*
- It studied for up to 60 rounds but quit after round 27 because its practice-exam
  grade stopped improving (see "early stopping" below).

## 5. Step 3 — Quality check on the recordings themselves

**Whisper** (a famous speech-to-text AI by OpenAI) wrote down what it heard in each
clip, and we compared its transcription against the official script.

- Typical result: **80% of words correct** → the audio is good quality.
- 31 of 690 clips (4.5%) confused it badly — scenes where **laughter or music drowns
  the speech**, making it invent repetitive junk ("no, no, no, no…"). This quirk is
  called *hallucination*. Those clips are now flagged so we know which recordings
  are noisy.

## 6. Decoder — every line you saw scroll by

### Screen 1: feature extraction
```
690 wav files | 690 labeled utterances
100%|##########| 690/690 [00:52<00:00, 15.68it/s]
```
- `wav files` — the sound recordings (WAV = a sound file format, like MP3).
- `labeled` — each clip has its known correct answer attached.
- progress bar + `it/s` — just speed (clips processed per second).

### Screen 2: the simple student's report card
```
== test ==
acc=0.6538   P=0.6559   R=0.6538   F1(macro)=0.6527
```
- `test` — the final exam: 104 clips never seen during study.
- `acc` (accuracy) — % answered correctly.
- `P` (precision) — when it shouts "sarcastic!", how often is it right? (avoids crying wolf)
- `R` (recall) — of all genuinely sarcastic clips, how many did it catch? (avoids missing them)
- `F1` — one combined grade from P and R, between 0 and 1. **This is our official score.**
- `macro` — both categories count equally; being good at "not sarcastic" alone can't inflate it.

```
feature importance (|coef|):
intensity_mean_db     1.014
f0_mean_hz           -0.454
duration_s            0.355
```
- `coef` (coefficient) — each measurement's vote weight. Big number = mattered a lot;
  minus sign = larger values pushed toward "NOT sarcastic".
- Reading: sarcastic clips skew louder (+), lower-pitched (−), longer (+).

### Screen 3: CNN studying
```
[cnn] epoch 27 ...
[cnn] early stop at epoch 27 (best val F1=0.6367)
```
- `epoch` — one full round of studying all 482 practice clips.
- `train_loss` — how wrong it was while studying (0 = perfect). We watch it fall as proof of learning.
- `val_F1` — grade on the practice exam (the other 104 clips). Used only to decide when to stop.
- `early stop` — quitting when practice grades stop improving; more rounds would just
  memorize answers instead of learning the skill (**overfitting**).

### Screen 4: CNN's final report card
```
== TEST (speech CNN) ==
acc=0.7212  P=0.7315  R=0.7212  F1(macro)=0.7180
confusion:
[[43  9]
 [20 32]]
speech embeddings [train] (482, 128)
```

The confusing box is just a right/wrong tally:

```
                        truth: NOT sarcastic   truth: sarcastic
guessed "NOT sarcastic"       43 ✅ right        20 ❌ missed
guessed "sarcastic"            9 ❌ false alarm   32 ✅ caught
```
- The jargon names: TN=43 (true negatives), FP=9 (false positives),
  FN=20 (false negatives), TP=32 (true positives).
- Interpretation: it rarely cries wolf (9 false alarms), but misses 20 genuine ones —
  slightly too conservative. Normal, and worth one thesis sentence.
- `speech embeddings (482, 128)` — each clip distilled into 128 numbers, like
  compressing each recording into a short personality summary. Saved so Phase 3 can
  combine them with the text model's summaries.

### Screen 5: Whisper audit
```
transcribing 690 clips
median WER: 0.200 ... skewed by 31 hallucinated laugh-track clips
```
- `transcribing` — typing out what it hears.
- `WER` (word error rate) — % of words transcribed wrong. 0.2 = 80% right.
- `median` vs `mean` — the middle value vs the average. We report median because the
  31 broken clips would drag the average unfairly.

## 7. Final results

| Model | Plain description | Official score (F1) |
|---|---|---|
| Random guessing | coin flip | 50% |
| Simple student (11 measurements) | Logistic Regression baseline | 65% |
| **Pattern-spotter (CNN)** | scans sound fingerprints | **72%** |

Cross-phase scoreboard (same final exam):

| What the computer used | Score |
|---|---|
| Only the WORDS (Phase 1) | 69% |
| **Only the VOICE (Phase 2)** | **72%** |

Headline finding: **sarcasm lives more in the delivery than in the text.**

## 8. Honest caveats (you WILL be asked these)

1. Some of the 72% may come from recognizing *which actor's voice* it is, since the
   same voices appear in both study and exam sets (jargon: speaker-dependent split).
2. Loudness partly reflects how each episode was mastered in the studio, not just
   the actor's delivery.
3. Models were trained on sitcom audio (with laugh tracks); they may behave
   differently on clean microphone recordings.

## 9. Quick glossary (every term that appeared)

| Term | Plain meaning |
|---|---|
| accuracy | % answered correctly |
| baseline | deliberately simple reference model; the bar to beat |
| batch | small group of examples studied together in one go |
| CNN | pattern-detector network; scans for telltale shapes anywhere in the input |
| coefficient | a measurement's vote weight in a decision |
| confusion matrix | the right/wrong tally table |
| dropout | during study, randomly hide some internal hints so the model can't over-rely on any single one |
| early stopping | quit studying when practice-exam grades stop improving |
| embedding | complex thing compressed into a short list of numbers capturing its essence |
| epoch | one full pass through all study material |
| F1 score | combined precision+recall grade, 0-to-1; our official metric |
| feature | one measured property of the input (e.g., pitch) |
| hallucination | AI confidently inventing content when it can't actually perceive (Whisper on laughter) |
| HNR | harmonics-to-noise ratio: clean vs breathy voice |
| jitter/shimmer | micro-wobbles in pitch/volume |
| learning rate | how big each study step is |
| loss | "how wrong" score that training tries to shrink |
| MFCC | standard sound fingerprint computed every 20 ms |
| overfitting | memorizing the answer key instead of learning the skill |
| Praat / librosa | measurement toolkits (phonetics software / audio library) |
| precision | of everything flagged sarcastic, how much really was |
| recall | of everything truly sarcastic, how much was caught |
| seed | fixed randomness so results are exactly repeatable |
| test set | final exam, touched once |
| train set | study material |
| val set | practice exam used for tuning decisions |
| WER | word error rate of transcription |
| Whisper | OpenAI's speech-to-text AI |
| z-score scaling | rescaling numbers so they're comparable (average 0, spread 1) |

## 10. What's next (Phase 3 preview)

We now have two experts:
- a **reader** who scores 69% (text model),
- a **listener** who scores 72% (voice model).

Phase 3 makes them **compare notes and vote together** (jargon: fusion). The goal:
combined score above 72%. If combining can't beat listening alone, we honestly
report that too — negative results are still valid research.

Each expert hands over their summary (their "embeddings": 768 numbers from reading,
128 from listening); a third small model learns when to trust whom.
