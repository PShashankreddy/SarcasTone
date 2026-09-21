"""PHASE 1 experiment: multitask text model.

Shared RoBERTa encoder (init from the champion's booster checkpoint) with two heads:
  - sarcasm  (binary, the target task)
  - emotion  (Implicit_Emotion / Explicit_Emotion, in-domain auxiliary supervision)

Total loss = CE(sarcasm) + lambda * CE(emotion). Selection is on validation sarcasm F1.
Test sarcasm macro-F1 is compared against the single-task champion (0.6866).

Usage:
  python -m sarcastone.models.multitask --aux Implicit_Emotion --lambda_aux 0.5 --epochs 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from ..evaluation.metrics import compute_metrics, format_metrics
from ..training.seed import set_seed
from ..utils import CKPT_DIR, REPORTS_DIR, SPLITS_DIR, SEED, save_json

RAW_CSV = Path(__file__).resolve().parents[3] / "data" / "raw" / "mustard_pp" / "mustardpp_text.csv"
MAX_LEN = 128


def load_aux_map(aux_col):
    df = pd.read_csv(RAW_CSV)
    df = df[df.Sarcasm.notna()].copy()
    df["utt_id"] = df.KEY.astype(str).str.replace("_u$", "", regex=True)
    classes = sorted(df[aux_col].dropna().unique())
    c2i = {c: i for i, c in enumerate(classes)}
    return dict(zip(df.utt_id, df[aux_col].map(c2i))), classes


class MultiTaskDataset(Dataset):
    def __init__(self, df, aux_map):
        self.texts = df.text.tolist()
        self.sarc = df.label.values.astype(int)
        # -100 = no aux label (a couple of locked clips are absent from the 1,202 set)
        self.aux = np.array([aux_map.get(u, -100) for u in df.utt_id.astype(str)], dtype=int)

    def __len__(self):
        return len(self.sarc)

    def __getitem__(self, i):
        return self.texts[i], int(self.sarc[i]), int(self.aux[i])


def collate(batch, tokenizer, max_len=MAX_LEN):
    texts, ys, ya = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return (enc["input_ids"], enc["attention_mask"],
            torch.tensor(ys), torch.tensor(ya))


class MultiTaskModel(nn.Module):
    def __init__(self, base, n_aux, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sarc = nn.Linear(h, 2)
        self.aux = nn.Linear(h, n_aux)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.drop(out.last_hidden_state[:, 0])
        return self.sarc(pooled), self.aux(pooled)


@torch.no_grad()
def predict_sarcasm(model, loader, device):
    model.eval()
    ys, ps = [], []
    for ids, attn, y, _ in loader:
        ls, _ = model(ids.to(device), attn.to(device))
        ps.extend(torch.softmax(ls, -1)[:, 1].cpu().tolist())
        ys.extend(y.tolist())
    ys, ps = np.asarray(ys), np.asarray(ps)
    return ys, ps, (ps >= 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/text_roberta_nh")
    ap.add_argument("--aux", default="Implicit_Emotion")
    ap.add_argument("--lambda_aux", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="roberta_mtl")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aux_map, classes = load_aux_map(args.aux)
    print(f"[mtl] device={device} base={args.base} aux={args.aux} "
          f"({len(classes)} classes) lambda={args.lambda_aux}")

    tok = AutoTokenizer.from_pretrained(args.base)
    parts = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(MultiTaskDataset(df, aux_map),
                      batch_size=(args.batch_size if s == "train" else 64),
                      shuffle=(s == "train"),
                      collate_fn=lambda b: collate(b, tok))
        for s, df in parts.items()
    }

    model = MultiTaskModel(args.base, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loaders["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    ce = nn.CrossEntropyLoss()
    ce_aux = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, attn, ys, ya in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            ls, la = model(ids.to(device), attn.to(device))
            loss = ce(ls, ys.to(device)) + args.lambda_aux * ce_aux(la, ya.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item()
        yv, _, pv = predict_sarcasm(model, loaders["val"], device)
        f1 = compute_metrics(yv, pv)["f1_macro"]
        star = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *best"
        print(f"[mtl] epoch {epoch}: loss={running/len(loaders['train']):.4f} "
              f"val_F1={f1:.4f}{star}", flush=True)

    model.load_state_dict(best_state)
    yt, probs, pred = predict_sarcasm(model, loaders["test"], device)
    m = compute_metrics(yt, pred)
    print(f"\n== TEST (multitask {args.tag}, aux={args.aux}) ==\n{format_metrics(m)}")
    save_json({**m, "base": args.base, "aux": args.aux, "lambda_aux": args.lambda_aux,
               "epochs": args.epochs, "lr": args.lr, "val_f1_best": best_f1},
              REPORTS_DIR / f"phase1_multitask_{args.tag}_test_metrics.json")
    ckpt = CKPT_DIR / f"text_multitask_{args.tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "model.pt")
    print(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
