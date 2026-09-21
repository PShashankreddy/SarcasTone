# Dataset Registry (decided)

Strategy per plan: **text baseline first, then speech, then fusion.**
Primary multimodal benchmark = **MUStARD++ full (1,202)** — the locked 104-clip
test set every model reports on is a subset of the original 690 and is
**never changed**.
Secondary training corpus = **PodSarc** (booster + its own separate track).
Extra text corpus = News Headlines v2 (bigger, cleaner supervision for the text model).

**Data topology (locked 2026-09-21 — Option A):**

| Portion | Role |
|---|---|
| MUStARD++ locked 482/104/104 (from the original 690) | official benchmark; **test untouched forever** |
| MUStARD++ extra 514 clips (`1_S09E…`/`1_S10E…`/`1_S11E…`/`1_S12E…`/`SV`) | **training only** (train 482 → 996) |
| PodSarc (11,024) | pre-training booster + its own separate evaluation track (never in MUStARD test) |
| Custom annotated sample (~300) | out-of-domain independent final evaluation (+ IAA) |

## 1. Primary: MUStARD++ full (text + audio + labels)

| | |
|---|---|
| Source | https://github.com/cfiltnlp/MUStARD_Plus_Plus (LREC-2022) |
| Text | `mustard++_text.csv` → `data/raw/mustard_pp/mustardpp_text.csv` (1,202 targets + 4,839 context rows) |
| Size | **1,202 utterances (601/601)**; original 690 subset + 514 new (BBT S9–12, Silicon Valley) |
| Fields per utt | utterance text, speaker, show, context rows (`SCENE_c_NN`), binary sarcasm label |
| Audio (690) | `data/raw/audio_raw/{utt_id}.mp4` + `data/raw/audio/{utt_id}.wav` (16 kHz mono PCM_16) |
| Audio (514 new) | `data/raw/audio/{1_S..}.wav`; source = official Drive folder (see below) |
| Role | ALL phases; the locked test set every model reports on |

**Expansion build artifacts**

- `data/raw/mustard_pp/mustardpp_expansion_rows.csv` — 514 new rows
  (`utt_id, text, context, speaker, show, label`), balanced 257/257,
  shows BBT=436 / SV=78. Zero ID collision with the locked gold.

**Known caveats (documented honestly for the report)**

- The 514 new rows' `context` is reconstructed by joining the CSV `SCENE_c_NN`
  rows in order. On the 688 overlapping scenes this matches our original
  `sarcasm_data.json` context ~95.5% of the time; the rest differ only by a few
  words (different context window). New rows therefore use the ++ window;
  the locked 690 rows are **not** altered.
- Audio provenance: the official Drive folder
  (`1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW`) throttles after bulk downloads. The 514
  utterance clips were mirrored from a public HF upload
  (`sifan077/MUStARD_plus_plus`) and **verified by filename/count against the
  official folder listing** (1,203 `*_u.mp4`). Cite the official source in the
  report; mirror is only a transport workaround.
- 2 original FRIENDS clips (`2_167`, `2_99`) exist in our 690 but not in the
  1,202 set; they stay where they already are (locked split unaffected).

## 2. Text booster: News Headlines Sarcasm v2

| | |
|---|---|
| Source | Kaggle "Sarcasm Headlines Dataset" (Saurabh Shahane) — `Sarcasm_Headlines_Dataset.json` |
| Size | ~28,600 headlines, balanced 0/1 |
| Role | optional pre-training / auxiliary fine-tuning of BERT **before** MUStARD++ fine-tune; never used for test metrics |

## 2b. Secondary corpus: PodSarc (audio + text + labels)

| | |
|---|---|
| Source | https://github.com/Abel1802/PodSarc (CC-BY-NC-4.0) |
| Size | 11,024 utterances, 29.42 h podcast speech, 8 speakers (Overly Sarcastic Podcast) |
| Labels | LLM-generated (GPT-4o / LLaMA3) + human-check on disagreements; 4,026 sarcastic / 6,998 not (~36.5%) |
| Audio | `audios.zip` (~422 MB) from GDrive folder `1VZbhgonsEImFPeZBPPl336nXJAD9Xb3M`; annotations JSON id `1rOSU0YhINRQQdUEFYibq6hC0H41a0PqH` (top-level **list**) |
| Role | pre-training booster for text/audio **and** its own separate evaluation track; **never** mixed into the MUStARD test |

> Caveats: different domain (podcast vs sitcom) and weaker (LLM) labels — report
> its number separately, never as a headline.

## 2c. Backup multimodal benchmark: MUStARD++ Balanced / House MD

| | |
|---|---|
| Source | https://github.com/spbanonymo/Sarcasm-Sight-and-Sound (arXiv 2310.01430) |
| Size | ~250+ House MD clips, audio+video+text, documented IAA, CC-BY-NC-4.0 |
| Role | reserve expansion / robustness check if needed |

## 3. Custom spoken sarcasm set (YouTube pipeline) — stretch goal

Pipeline (implemented in `sarcastone.features.segment_youtube`):

1. `yt-dlp` download conversational videos (talk shows, sitcom scenes, interviews)
2. Whisper → timestamped utterance segments
3. Cut clips at utterance boundaries (NOT fixed 7–12 s chunks)
4. Save ±context: 2–4 preceding utterances per clip (`custom_clips/context.json`)
5. Annotate ≥300 clips with ≥3 annotators using the generated template
6. Compute Cohen's kappa (pairwise) + Fleiss' kappa (overall); keep majority-agreed clips
7. Merge survivors as an extra eval set for modality comparison

> Research-use note: keep downloaded media out of git (already in `.gitignore`);
> do not redistribute clips.

## Rejected / backup options (documented for the report)

- **SARC (Reddit)** ~2M comments — huge but noisy self-labels; overkill here
- **iSarcasm** — author-labelled tweets; useful future text-only benchmark
- **MELD** — emotion, not sarcasm labels (could mine sarcastic-looking anger clips if needed)
- **UR-FUNNY / DOPE** — TED-talk humor/punchline, close cousin task, not sarcasm

## Annotation workflow

```powershell
# after building custom clips:
python -m sarcastone.data.annotation make-template   # blank sheet per annotator
# ... annotators fill data/interim/annotations/annotator_{A,B,C}.csv ...
python -m sarcastone.data.annotation agreement       # kappas + disagreement list
python -m sarcastone.data.annotation merge           # final labeled custom set
```
