# Project: SarcasTone — multimodal sarcasm detection (text + speech)

## Communication style (important)

The user is new to machine-learning terminology. For every technical step:

1. **Plain English first** — explain what we are doing and why using everyday
   analogies (students, exams, flashcards), before any jargon.
2. Then show the **technical detail underneath**, clearly separated.
3. When showing terminal output or results, decode the terms that appear
   (see docs/PHASE2_PLAIN_ENGLISH.md for the established format).

## Project conventions

- Python venv: `.venv` — run as `& .\.venv\Scripts\python.exe -W ignore -m ...`
- Locked splits: seed 42; select models on val, touch test once.
- Media/audio files must never be committed to git.
- Reports go to `reports/`; study-friendly explainers to `docs/`.
