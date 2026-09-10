"""PHASE 2: 1D-CNN over acoustic sequences (MFCC+deltas+RMS), plus LR baseline
on Praat summary features. Dumps penultimate embeddings for fusion.

Usage:
  python -m sarcastone.models.speech_cnn --baseline        # summary-features LogReg (Week 5)
  python -m sarcastone.models.speech_cnn                   # CNN train + test eval
  python -m sarcastone.models.speech_cnn --extract_embeddings
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.make_splits import make_splits
from ..evaluation.error_analysis import write_error_report
from ..evaluation.metrics import compute_metrics, format_metrics, plot_confusion
from ..training.seed import set_seed
from ..training.trainer import evaluate as eval_loop, extract_embeddings, fit
from ..utils import CKPT_DIR, EMB_DIR, REPORTS_DIR, save_json
from .speech_data import SeqDataset, summary_frame, summary_xy


class SarcasmCNN(nn.Module):
    """Input: (B, 40, T) z-scored feature frames -> logits over {non-sarc, sarc}."""

    def __init__(self, in_dim=40, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.MaxPool1d(4),                       # 200 -> 50
            nn.Conv1d(hidden, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Linear(128, 2)
        self.drop = nn.Dropout(dropout)

    def embed(self, x):
        return self.net(x).squeeze(-1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def run_lr_baseline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parts = {s: summary_frame(s) for s in ("train", "val", "test")}
    Xtr, ytr = summary_xy(parts["train"])
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))
    pipe.fit(Xtr, ytr)
    print("== speech baseline: Praat summary + LogReg ==")
    for split in ("val", "test"):
        X, y = summary_xy(parts[split])
        pred = pipe.predict(X); prob = pipe.predict_proba(X)[:, 1]
        m = compute_metrics(y, pred)
        print(f"\n== {split} ==\n{format_metrics(m)}")
        if split == "test":
            coefs = pd.Series(pipe.named_steps["logisticregression"].coef_[0],
                              index=["duration_s", "f0_mean_hz", "f0_std_hz", "f0_p05",
                                     "f0_p95", "f0_range", "intensity_mean_db",
                                     "intensity_std_db", "hnr_mean_db",
                                     "jitter_local", "shimmer_local"]).sort_values(key=np.abs)
            print("\nfeature importance (|coef|):\n" + coefs.to_string())
            save_json({"metrics": m,
                       "feature_coefs": coefs.to_dict()},
                      REPORTS_DIR / "phase2_lr_test_metrics.json")
            write_error_report(parts["test"].text.tolist(), y, pred, prob,
                               parts["test"].speaker.values,
                               REPORTS_DIR / "phase2_lr_error_analysis.md",
                               "Phase 2 errors: Praat features + LogReg")


def train_cnn(extract_embs: bool = False):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = make_splits()
    ds = {s: SeqDataset(splits[s][["utt_id", "label"]]) for s in splits}
    dl = {s: DataLoader(ds[s], batch_size=32, shuffle=(s == "train")) for s in ds}

    model, hist, best = fit(SarcasmCNN(), dl["train"], dl["val"],
                            epochs=60, lr=1e-3, weight_decay=1e-4, patience=8,
                            device=device, desc="cnn")
    ckpt_dir = CKPT_DIR / "speech_cnn"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    m, ys, probs = eval_loop(model, dl["test"], device)
    pred = (probs >= 0.5).astype(int)
    print(f"\n== TEST (speech CNN) ==\n{format_metrics(m)}")
    save_json(m, REPORTS_DIR / "phase2_cnn_test_metrics.json")
    plot_confusion(ys, pred, REPORTS_DIR / "figures" / "speech_cnn_cm.png", "Speech 1D-CNN")
    texts_by_id = dict(zip(splits["test"].utt_id.astype(str), splits["test"].text))
    order = [u for u in splits["test"].utt_id.astype(str)]
    write_error_report([texts_by_id[u] for u in order], ys, pred, probs,
                       splits["test"].speaker.values,
                       REPORTS_DIR / "phase2_cnn_error_analysis.md",
                       "Phase 2 errors: speech CNN")

    if extract_embs:
        for split in ("train", "val", "test"):
            emb, labels = extract_embeddings(model, dl[split], device)
            np.savez(EMB_DIR / f"speech_{split}.npz",
                     ids=splits[split].utt_id.astype(str).values.astype(object),
                     emb=emb, label=labels)
            print(f"speech embeddings [{split}] {emb.shape}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="Praat summary + LogReg only")
    ap.add_argument("--extract_embeddings", action="store_true")
    args = ap.parse_args()
    if args.baseline:
        run_lr_baseline()
    else:
        train_cnn(extract_embs=args.extract_embeddings)
