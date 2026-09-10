# Dataset Registry (decided)

Strategy per plan: **text baseline first, then speech, then fusion.**
Primary multimodal benchmark = MUStARD++ (same locked test set for all three models).
Extra text corpus = News Headlines v2 (bigger, cleaner supervision for the text model).

## 1. Primary: MUStARD++ (text + audio + labels)

| | |
|---|---|
| Source | https://github.com/soujanyaporia/MUStARD |
| Size | 690 utterances (~50/50 sarcastic/non-sarcastic), from *Friends*, *The Golden Girls*, etc. |
| Fields per utt | utterance text, speaker, show, context utterances (before/after), binary sarcasm label |
| Audio | collected separately — clips hosted per-show; place in `data/raw/audio_raw/{utt_id}.mp4` |
| Role | ALL phases; the locked test set every model reports on |

## 2. Text booster: News Headlines Sarcasm v2

| | |
|---|---|
| Source | Kaggle "Sarcasm Headlines Dataset" (Saurabh Shahane) — `Sarcasm_Headlines_Dataset.json` |
| Size | ~28,600 headlines, balanced 0/1 |
| Role | optional pre-training / auxiliary fine-tuning of BERT **before** MUStARD++ fine-tune; never used for test metrics |

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
