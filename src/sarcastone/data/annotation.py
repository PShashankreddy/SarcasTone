"""Annotation workflow for the custom spoken-sarcasm set.

Three commands:
  make-template  blank per-annotator sheets from the clip manifest
  agreement      Cohen's (pairwise) + Fleiss' (overall) kappa -> IAA report
  merge          majority-agreed labels -> data/processed/custom_dataset.csv

Usage:
  python -m sarcastone.data.annotation make-template [--n 400]
  python -m sarcastone.data.annotation agreement
  python -m sarcastone.data.annotation merge
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..utils import DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR

ANN_DIR = DATA_INTERIM / "annotations"
CLIPS_MANIFEST = DATA_INTERIM / "custom_clips" / "context.json"
ANNOTATORS = ["A", "B", "C"]


def make_template(n: int | None = None, sample: bool = False):
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(CLIPS_MANIFEST.read_text(encoding="utf-8"))
    if n:
        if sample:      # seeded random draw -> unbiased across source videos
            import random

            items = random.Random(42).sample(items, min(n, len(items)))
        else:
            items = items[:n]
    df = pd.DataFrame([{
        "utt_id": c["utt_id"],
        "clip_path": c["clip_path"],
        "text": c["text"],
        "context_before": " || ".join(c.get("context_before", [])),
        "label": "",        # annotator fills: 1 = sarcastic, 0 = not, x = unclear
        "notes": "",
    } for c in items])
    for who in ANNOTATORS:
        out = ANN_DIR / f"annotator_{who}.csv"
        if out.exists():
            print(f"[skip] {out.name} exists")
            continue
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} items)")


def _load_sheets() -> dict[str, pd.DataFrame]:
    sheets = {}
    for who in ANNOTATORS:
        p = ANN_DIR / f"annotator_{who}.csv"
        if p.exists():
            s = pd.read_csv(p)
            s["label"] = s.label.astype(str).str.strip().str.lower()
            sheets[who] = s[s.label.isin(["0", "1"])]      # drop blanks/'x'
    return sheets


def fleiss_kappa(table: np.ndarray) -> float:
    """table: (n_items, n_categories) counts of annotator votes."""
    n, k = table.shape
    N = table.sum(axis=1)[0]                      # annotators per item (constant assumed)
    p_i = ((table ** 2).sum(axis=1) - N) / (N * (N - 1))
    P_bar = p_i.mean()
    p_j = table.sum(axis=0) / (n * N)
    P_e = (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def agreement():
    sheets = _load_sheets()
    if len(sheets) < 2:
        raise SystemExit("need at least two filled annotator sheets")
    print("== pairwise Cohen's kappa ==")
    pair_kappas = []
    for a, b in itertools.combinations(sheets, 2):
        m = sheets[a][["utt_id", "label"]].merge(
            sheets[b][["utt_id", "label"]], on="utt_id", suffixes=(f"_{a}", f"_{b}"))
        k = cohen_kappa_score(m[f"label_{a}"], m[f"label_{b}"])
        pair_kappas.append(k)
        print(f"  {a}-{b}: {k:.3f}  ({len(m)} shared items)")

    # Fleiss: items labelled by everyone
    common = set.intersection(*(set(s.utt_id) for s in sheets.values()))
    votes = np.zeros((len(common), 2), dtype=int)
    ids = sorted(common)
    for i, u in enumerate(ids):
        for s in sheets.values():
            votes[i, int(float(s.loc[s.utt_id == u, "label"].iloc[0]))] += 1
    fk = fleiss_kappa(votes)

    maj = (votes[:, 1] > votes.sum(axis=1) / 2).astype(int)
    unanimous = int((votes.max(axis=1) == votes.sum(axis=1)).sum())
    rep = [
        "# Inter-Annotator Agreement\n",
        f"- annotators: {', '.join(sheets)}",
        f"- items labelled by all: {len(common)}",
        f"- pairwise Cohen's kappa: {[round(k, 3) for k in pair_kappas]}",
        f"- overall Fleiss' kappa: **{fk:.3f}** "
        f"({'substantial' if fk >= .61 else 'moderate' if fk >= .41 else 'slight'} agreement)",
        f"- unanimous items: {unanimous}/{len(common)}\n",
        "> Landis & Koch: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, "
        "0.61-0.80 substantial, >0.80 almost perfect.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "iaa_report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"\nsaved -> {REPORTS_DIR / 'iaa_report.md'}")

    agree_df = pd.DataFrame({"utt_id": ids,
                             "votes_sarcasm": votes[:, 1],
                             "majority_label": maj})
    agree_df.to_csv(ANN_DIR / "agreement.csv", index=False)


def merge(min_votes: int = 2):
    path = ANN_DIR / "agreement.csv"
    if not path.exists():
        raise SystemExit("run `agreement` first")
    ag = pd.read_csv(path)
    keep = ag[ag.votes_sarcasm >= min_votes].copy()
    keep = keep.rename(columns={"majority_label": "label"})
    keep["speaker"] = "custom"
    keep["show"] = "youtube_custom"
    keep.to_csv(DATA_PROCESSED / "custom_dataset.csv", index=False)
    print(f"kept {len(keep)}/{len(ag)} clips (>= {min_votes} votes) "
          f"-> {DATA_PROCESSED / 'custom_dataset.csv'}")


def make_html():
    """One listening page per annotator: audio player + transcript + context."""
    import html as _h

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    for who in ANNOTATORS:
        p = ANN_DIR / f"annotator_{who}.csv"
        if not p.exists():
            print(f"[skip] {p.name} missing (run make-template first)")
            continue
        s = pd.read_csv(p)
        blocks = []
        for i, r in s.iterrows():
            ctx = "" if pd.isna(r.context_before) or not str(r.context_before).strip() \
                else f"<p><i>Context before:</i> {_h.escape(str(r.context_before))}</p>"
            audio = Path(str(r.clip_path)).resolve().as_posix()
            blocks.append(
                f'<div class="item"><h3>{i + 1}. utt_id {r.utt_id}</h3>'
                f'<audio controls preload="none" src="file:///{audio}"></audio>'
                f'<p><b>Said:</b> {_h.escape(str(r.text))}</p>{ctx}'
                f"<p>Label for CSV: <b>1</b> sarcastic &middot; <b>0</b> not &middot; "
                f"<b>x</b> unclear</p></div>"
            )
        doc = (
            "<!doctype html><meta charset=\"utf-8\">"
            f"<title>Annotator {who}</title>"
            "<style>body{font-family:'Segoe UI',sans-serif;max-width:780px;"
            "margin:auto;padding:1em;color:#222}"
            ".item{border:1px solid #ccc;border-radius:10px;padding:.8em 1em;margin:1em 0}"
            "audio{width:100%}h1{font-size:1.4em}</style>"
            f"<h1>Sarcasm listening task — Annotator {who}</h1>"
            "<p>Headphones on. Listen to each clip (twice if unsure), decide sarcastic or not, "
            f"and type the label next to that utt_id in <code>annotator_{who}.csv</code>. "
            "Work alone. Save the CSV when done.</p>" + "".join(blocks)
        )
        out = ANN_DIR / f"annotator_{who}.html"
        out.write_text(doc, encoding="utf-8")
        print(f"wrote {out} ({len(blocks)} items)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-template", "make-html", "agreement", "merge"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--min_votes", type=int, default=2)
    args = ap.parse_args()
    {"make-template": lambda: make_template(args.n, args.random),
     "make-html": make_html,
     "agreement": agreement,
     "merge": lambda: merge(args.min_votes)}[args.cmd]()
