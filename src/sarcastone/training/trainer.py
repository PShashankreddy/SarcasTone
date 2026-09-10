"""Generic PyTorch training/eval helpers shared by CNN / RNN / fusion models.

Convention: every DataLoader yields tuples `(x..., y)` — any number of input
tensors followed by integer class labels as the last element.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device="cpu") -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        *xs, y = batch
        xs = [x.to(device) for x in xs]
        logits = model(*xs)
        ps.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        ys.extend(y.tolist())
    pred = (np.asarray(ps) >= 0.5).astype(int)
    return compute_metrics(ys, pred), np.asarray(ys), np.asarray(ps)


def fit(model, train_loader, val_loader, epochs=60, lr=1e-3, weight_decay=0.0,
        patience=8, device=None, desc="train"):
    """Train with early stopping on val macro-F1; returns model with best weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_f1, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            *xs, y = batch
            xs = [x.to(device) for x in xs]
            y = y.to(device)
            opt.zero_grad()
            loss = crit(model(*xs), y)
            loss.backward()
            opt.step()
            total += loss.item()
        m, _, _ = evaluate(model, val_loader, device)
        marker = ""
        if m["f1_macro"] > best_f1:
            best_f1, best_state, bad = m["f1_macro"], copy.deepcopy(model.state_dict()), 0
            marker = " *"
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": total / max(len(train_loader), 1), **m})
        print(f"[{desc}] ep {epoch:>2} loss={total / max(len(train_loader), 1):.4f} "
              f"val_f1={m['f1_macro']:.4f}{marker}")
        if bad >= patience:
            print(f"[{desc}] early stop at epoch {epoch} (best val F1={best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history, best_f1


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device="cpu"):
    """Model must expose `.embed(x...) -> penultimate features`. Returns (embs, labels)."""
    model.eval()
    embs, ys = [], []
    for batch in loader:
        *xs, y = batch
        xs = [x.to(device) for x in xs]
        embs.append(model.embed(*xs).cpu())
        ys.append(y)
    return torch.cat(embs).numpy(), torch.cat(ys).numpy()
