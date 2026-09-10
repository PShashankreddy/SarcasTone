"""PHASE 1: fine-tune bert-base-uncased on MUStARD++ transcripts.

Defaults follow the plan: 3 epochs, lr=2e-5, batch=16, max_len=128.
Also dumps [CLS] embeddings (768-d) for all splits -> Phase 3 fusion.

Usage:
  python -m sarcastone.models.bert_finetune                 # train + test eval
  python -m sarcastone.models.bert_finetune --extract_embeddings
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..evaluation.error_analysis import write_error_report
from ..evaluation.metrics import compute_metrics, format_metrics, plot_confusion
from ..training.seed import set_seed
from ..utils import (CKPT_DIR, EMB_DIR, REPORTS_DIR, SPLITS_DIR, SEED,
                     load_config, save_json)

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128


def model_tag(model_name: str) -> str:
    """Short tag for checkpoints/artifacts: bert-base-uncased -> bert, roberta-base -> roberta."""
    return model_name.split("-")[0].split("/")[0]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, contexts=None, use_context=False,
                 max_len=MAX_LEN):
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_len
        sep = f" {tokenizer.sep_token} "
        if use_context and contexts is not None:
            self.texts = [f"{c.strip()}{sep}{t}" if c and str(c).strip() else t
                          for t, c in zip(texts, contexts)]
        else:
            self.texts = list(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i])


def collate(batch, tokenizer, max_len=MAX_LEN):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)


def load_splits(splits_dir=None):
    from ..utils import SPLITS_DIR as DEFAULT_DIR
    splits_dir = Path(splits_dir) if splits_dir else DEFAULT_DIR
    parts = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(splits_dir / f"{s}.csv")
        parts[s] = {"texts": df.text.tolist(), "labels": df.label.values,
                    "speakers": df.speaker.values, "ids": df.utt_id.astype(str).tolist(),
                    "contexts": (df.context.fillna("").tolist()
                                 if "context" in df.columns else None)}
    return parts


def collate_with(tokenizer):
    return lambda batch: collate(batch, tokenizer)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for input_ids, attn, y in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        ps.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
        ys.extend(y.tolist())
    ys, ps = np.asarray(ys), np.asarray(ps)
    return ys, ps, (ps >= 0.5).astype(int)


@torch.no_grad()
def extract_cls_embeddings(model_base, loader, device):
    model_base.eval()
    embs, ys = [], []
    for input_ids, attn, y in loader:
        out = model_base(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        embs.append(out.last_hidden_state[:, 0].cpu())
        ys.append(y)
    return torch.cat(embs).numpy(), np.concatenate(ys)


def train(cfg: dict | None = None, extract_embeddings: bool = False, splits_dir=None,
          model_name: str | None = None, use_context: bool = False,
          tag_override: str | None = None):
    cfg = cfg or {}
    epochs = int(cfg.get("epochs", 3)); lr = float(cfg.get("lr", 2e-5))
    batch_size = int(cfg.get("batch_size", 16)); seed = int(cfg.get("seed", SEED))
    max_len = int(cfg.get("max_len", MAX_LEN))
    name = model_name or cfg.get("model_name", MODEL_NAME)
    tag = tag_override or model_tag(name)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] device={device} epochs={epochs} lr={lr} bs={batch_size} "
          f"ctx={use_context} max_len={max_len}\n[{tag}] from={name}")

    tok = AutoTokenizer.from_pretrained(name)
    parts = load_splits(splits_dir)
    loaders = {
        s: DataLoader(
            TextDataset(p["texts"], p["labels"], tok,
                        contexts=p.get("contexts"), use_context=use_context),
            batch_size=(batch_size if s == "train" else 64),
            shuffle=(s == "train"), collate_fn=lambda b: collate(b, tok, max_len))
        for s, p in parts.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loaders["train"]) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}")
        for input_ids, attn, y in pbar:
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device), labels=y.to(device))
            opt.zero_grad(); out.loss.backward(); opt.step(); sched.step()
            running += out.loss.item()
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")

        ys, _, pred = predict(model, loaders["val"], device)
        m = compute_metrics(ys, pred)
        star = ""
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *best"
        print(f"[{tag}] epoch {epoch}: train_loss={running/len(loaders['train']):.4f} "
              f"val_F1={m['f1_macro']:.4f}{star}")

    model.load_state_dict(best_state)
    ckpt = CKPT_DIR / f"text_{tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)

    ys, probs, pred = predict(model, loaders["test"], device)
    m = compute_metrics(ys, pred)
    print(f"\n== TEST ({tag}, context={use_context}) ==\n{format_metrics(m)}")
    save_json({**m, "model_name": name, "use_context": use_context,
               "lr": lr, "epochs": epochs, "batch_size": batch_size},
              REPORTS_DIR / f"phase1_{tag}_test_metrics.json")
    plot_confusion(ys, pred, REPORTS_DIR / "figures" / f"{tag}_cm.png", f"Fine-tuned {tag}")
    write_error_report(parts["test"]["texts"], ys, pred, probs, parts["test"]["speakers"],
                       REPORTS_DIR / f"phase1_{tag}_error_analysis.md",
                       f"Phase 1 errors: fine-tuned {tag}")

    if extract_embeddings:
        base = model.base_model   # works for bert/roberta/distilbert wrappers
        for split in ("train", "val", "test"):
            embs, labels = extract_cls_embeddings(base, loaders[split], device)
            np.savez(EMB_DIR / f"text_{tag}_{split}.npz",
                     ids=np.array(parts[split]["ids"]), emb=embs, label=labels)
            print(f"text embeddings [{tag}/{split}] -> text_{tag}_{split}.npz {embs.shape}")
            # canonical copies used by Phase 3 fusion (last promoted run wins)
            np.savez(EMB_DIR / f"text_{split}.npz",
                     ids=np.array(parts[split]["ids"]), emb=embs, label=labels)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to configs/text_bert.yaml")
    ap.add_argument("--extract_embeddings", action="store_true")
    ap.add_argument("--splits_dir", default=None,
                    help="e.g. data/processed/splits_newsheadlines (default: MUStARD splits)")
    ap.add_argument("--model_name", default=None,
                    help="e.g. bert-base-uncased, roberta-base (overrides config)")
    ap.add_argument("--use_context", action="store_true",
                    help="prepend preceding dialogue lines to the utterance")
    ap.add_argument("--tag", default=None,
                    help="override artifact tag (checkpoint/embeddings naming)")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else {}
    train(cfg,
          extract_embeddings=args.extract_embeddings or bool(cfg.get("extract_embeddings", False)),
          splits_dir=args.splits_dir,
          model_name=args.model_name,
          use_context=args.use_context,
          tag_override=args.tag)
