"""PHASE 1 experiment: overfitting-focused regularised fine-tuning.

Same base/recipe as the champion, with three anti-overfitting controls:
  - freeze the embeddings + bottom N encoder layers
  - layer-wise learning-rate decay (lower layers train slower)
  - early stopping on validation LOSS (not F1), with best-val-loss weight restore

Single-task sarcasm; test macro-F1 compared against the champion (0.6866).

Usage:
  python -m sarcastone.models.finetune_regularized --freeze_layers 6 --layer_lr_decay 0.85
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..evaluation.metrics import compute_metrics, format_metrics
from ..models.bert_finetune import TextDataset, collate, predict
from ..training.seed import set_seed
from ..utils import CKPT_DIR, REPORTS_DIR, SPLITS_DIR, SEED, save_json

BASE = "checkpoints/text_roberta_nh"
MAX_LEN = 128


def build_param_groups(model, base_lr, decay, freeze_layers):
    groups = []
    enc = model.base_model
    layers = enc.encoder.layer
    n = len(layers)
    frozen = 0
    for p in enc.embeddings.parameters():
        p.requires_grad = False
        frozen += p.numel()
    for i, layer in enumerate(layers):
        if i < freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False
                frozen += p.numel()
            continue
        depth = n - i                      # top layer -> 1, deeper -> larger
        lr = base_lr * (decay ** (depth - 1))
        groups.append({"params": [p for p in layer.parameters() if p.requires_grad], "lr": lr})
    head = [p for n_, p in model.named_parameters()
            if ("classifier" in n_ or "pooler" in n_) and p.requires_grad]
    groups.append({"params": head, "lr": base_lr})
    return groups, frozen


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    tot, n = 0.0, 0
    for ids, attn, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        tot += ce(logits, y.to(device)).item(); n += len(y)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--freeze_layers", type=int, default=6)
    ap.add_argument("--layer_lr_decay", type=float, default=0.85)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="roberta_reg")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[reg] device={device} freeze={args.freeze_layers} decay={args.layer_lr_decay} "
          f"lr={args.lr} epochs={args.epochs} patience={args.patience}")

    tok = AutoTokenizer.from_pretrained(args.base)
    parts = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(TextDataset(df.text.tolist(), df.label.values, tok),
                      batch_size=(args.batch_size if s == "train" else 64),
                      shuffle=(s == "train"), collate_fn=lambda b: collate(b, tok, MAX_LEN))
        for s, df in parts.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)
    groups, frozen = build_param_groups(model, args.lr, args.layer_lr_decay, args.freeze_layers)
    print(f"[reg] frozen params: {frozen/1e6:.1f}M | trainable groups: {[g['lr'] for g in groups]}")
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = len(loaders["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, attn, y in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            out = model(input_ids=ids.to(device), attention_mask=attn.to(device), labels=y.to(device))
            opt.zero_grad(); out.loss.backward(); opt.step(); sched.step()
            running += out.loss.item()
        vl = val_loss(model, loaders["val"], device)
        yv, _, pv = predict(model, loaders["val"], device)
        vf1 = compute_metrics(yv, pv)["f1_macro"]
        improved = vl < best_loss - 1e-4
        if improved:
            best_loss = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"[reg] epoch {epoch}: train_loss={running/len(loaders['train']):.4f} "
              f"val_loss={vl:.4f} val_F1={vf1:.4f}{' *best' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"[reg] early stop (no val-loss improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    yt, probs, pred = predict(model, loaders["test"], device)
    m = compute_metrics(yt, pred)
    print(f"\n== TEST (regularised {args.tag}) ==\n{format_metrics(m)}")
    save_json({**m, "base": args.base, "freeze_layers": args.freeze_layers,
               "layer_lr_decay": args.layer_lr_decay, "lr": args.lr,
               "best_val_loss": best_loss},
              REPORTS_DIR / f"phase1_reg_{args.tag}_test_metrics.json")
    ckpt = CKPT_DIR / f"text_reg_{args.tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    print(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
