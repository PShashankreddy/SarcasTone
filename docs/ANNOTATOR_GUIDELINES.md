# Annotator Guidelines — Sarcasm Listening Task

Read this ONCE before you start. It takes 3 minutes and makes everyone's labels
comparable.

## Your job

You will listen to short clips of people talking. For each one you decide:

| Label | Meaning |
|---|---|
| **1** | The speaker is being sarcastic |
| **0** | The speaker is NOT being sarcastic |
| **x** | You genuinely cannot tell |

## What counts as sarcastic?

The speaker **means something different from (usually opposite to) what the words
literally say**, typically to mock, complain, or joke. Trust your instinct — if you
would need air-quotes for a line, it's probably a 1.

**Examples of "1":**
- *"Oh great, my car broke down again. Just wonderful."* (clearly not delighted)
- *"Yeah, because THAT worked so well last time."*
- Praise so exaggerated it's obviously mockery ("Brilliant. Just brilliant.")
- Deadpan absurdity delivered as if serious

**Examples of "0" (NOT sarcastic):**
- Ordinary statements and sincere compliments
- Jokes that are literal/funny without meaning the opposite ("I ate so much cake")
- Genuine surprise or excitement
- Simple questions or narration

**Use "x" sparingly**: only when the clip is too short/noisy to judge, or the tone
is truly ambiguous. Don't agonize — first instinct is usually right.

## Rules

1. **Work alone.** No discussing clips with other annotators until everyone finishes.
2. Use **headphones** in a reasonably quiet place.
3. Listen up to **twice** per clip, then decide.
4. Judge the **speaker in the clip**, not whether the clip is funny overall.
5. Fill the label column next to the matching `utt_id` in your CSV sheet
   (`annotator_A/B/C.csv`), then save the file.
6. There is no time limit, but try to finish within a few days while your
   internal standard stays consistent.

## Why we're doing this

Your agreement with the other annotators is measured statistically (kappa).
High agreement proves sarcasm-by-voice is a real, recognizable phenomenon —
that's the foundation of the whole research project. Thank you!
