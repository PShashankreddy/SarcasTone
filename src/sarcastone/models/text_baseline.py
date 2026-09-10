"""Week-2 baselines before BERT fine-tuning:
  tfidf   - TF-IDF word n-grams + LogisticRegression
  bert_cls- frozen bert-base-uncased [CLS] embeddings + LogisticRegression

Usage: python -m sarcastone.models.text_baseline --mode tfidf
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.make_splits import make_splits
from ..evaluation.error_analysis import write_error_report
from ..evaluation.metrics import compute_metrics, format_metrics, plot_confusion
from ..utils import REPORTS_DIR, save_json


def load_split_texts(parts=None, splits_dir=None):
    if parts is None:
        from pathlib import Path
        from ..utils import SPLITS_DIR
        d = Path(splits_dir) if splits_dir else SPLITS_DIR
        parts = {name: pd.read_csv(d / f"{name}.csv") for name in ("train", "val", "test")}
    out = {}
    for name, df in parts.items():
        out[name] = (df.text.tolist(), df.label.values, df.speaker.values)
    return out


def run_tfidf(data):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0)),
    ])
    pipe.fit(*data["train"][:2])
    results = {}
    for split in ("val", "test"):
        texts, y, speakers = data[split]
        pred = pipe.predict(texts)
        prob = pipe.predict_proba(texts)[:, 1]
        m = compute_metrics(y, pred)
        results[split] = m
        print(f"\n== {split} ==\n{format_metrics(m)}")
        if split == "test":
            plot_confusion(y, pred, REPORTS_DIR / "figures" / "lr_tfidf_cm.png", "TF-IDF + LogReg")
            save_json(m, REPORTS_DIR / "phase1_lrtfidf_test_metrics.json")
            write_error_report(texts, y, pred, prob, speakers,
                               REPORTS_DIR / "phase1_lrtfidf_error_analysis.md",
                               "Phase 1 baseline errors: TF-IDF + LogReg")
    return results


def run_bert_cls(data, device=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    @torch.no_grad()
    def embed(texts):
        embs = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            embs.append(encoder(**enc).last_hidden_state[:, 0].cpu().numpy())
        return np.vstack(embs)

    Xtr = embed(data["train"][0])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, data["train"][1])

    results = {}
    for split in ("val", "test"):
        texts, y, speakers = data[split]
        pred = clf.predict(embed(texts))
        prob = clf.predict_proba(embed(texts))[:, 1]
        m = compute_metrics(y, pred)
        results[split] = m
        print(f"\n== {split} ==\n{format_metrics(m)}")
        if split == "test":
            plot_confusion(y, pred, REPORTS_DIR / "figures" / "lr_bertcls_cm.png", "BERT-CLS + LogReg")
            save_json(m, REPORTS_DIR / "phase1_lrbertcls_test_metrics.json")
            write_error_report(texts, y, pred, prob, speakers,
                               REPORTS_DIR / "phase1_lrbertcls_error_analysis.md",
                               "Phase 1 baseline errors: frozen BERT-CLS + LogReg")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tfidf", "bert_cls", "both"], default="both")
    ap.add_argument("--splits_dir", default=None,
                    help="default: MUStARD splits; or data/processed/splits_newsheadlines")
    args = ap.parse_args()
    data = load_split_texts(splits_dir=args.splits_dir)
    if args.mode in ("tfidf", "both"):
        run_tfidf(data)
    if args.mode in ("bert_cls", "both"):
        run_bert_cls(data)
