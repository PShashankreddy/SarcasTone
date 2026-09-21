"""PHASE 2 alternative: BiGRU over the same acoustic sequences as the CNN.

Usage:
  python -m sarcastone.models.speech_rnn
  python -m sarcastone.models.speech_rnn --extract_embeddings
"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.make_splits import make_splits
from ..evaluation.error_analysis import write_error_report
from ..evaluation.metrics import compute_metrics, format_metrics, plot_confusion
from ..training.seed import set_seed
from ..training.trainer import evaluate as eval_loop, extract_embeddings, fit
from ..utils import CKPT_DIR, EMB_DIR, REPORTS_DIR, save_json
from .speech_cnn import SarcasmCNN  # noqa: F401 (keeps shared imports warm)
from .speech_data import SeqDataset


class SarcasmRNN(nn.Module):
    """Input (B, D=40, T=200) -> BiGRU last-step hidden (2H) -> logits."""

    def __init__(self, in_dim=40, hidden=64, layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 2)

    def embed(self, x):                        # x: (B, D, T)
        out, _ = self.gru(x.transpose(1, 2))   # -> (B, T, 2H)
        return out[:, -1]                      # last frame summary

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def train_rnn(extract_embs: bool = False):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = make_splits()
    ds = {s: SeqDataset(splits[s][["utt_id", "label"]]) for s in splits}
    dl = {s: DataLoader(ds[s], batch_size=32, shuffle=(s == "train")) for s in ds}

    model, hist, best = fit(SarcasmRNN(), dl["train"], dl["val"],
                            epochs=60, lr=1e-3, weight_decay=1e-4, patience=8,
                            device=device, desc="rnn")
    ckpt_dir = CKPT_DIR / "speech_rnn"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    m, ys, probs = eval_loop(model, dl["test"], device)
    pred = (probs >= 0.5).astype(int)
    print(f"\n== TEST (speech BiGRU) ==\n{format_metrics(m)}")
    save_json(m, REPORTS_DIR / "phase2_rnn_test_metrics.json")
    plot_confusion(ys, pred, REPORTS_DIR / "figures" / "speech_rnn_cm.png", "Speech BiGRU")

    if extract_embs:
        for split in ("train", "val", "test"):
            emb, labels = extract_embeddings(model, dl[split], device)
            np.savez(EMB_DIR / f"speech_rnn_{split}.npz",
                     ids=splits[split].utt_id.astype(str).values.astype(object),
                     emb=emb, label=labels)
            print(f"speech embeddings [{split}] {emb.shape}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_embeddings", action="store_true")
    args = ap.parse_args()
    train_rnn(extract_embs=args.extract_embeddings)
