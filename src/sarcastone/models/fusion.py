"""PHASE 3: multimodal fusion of text + speech embeddings on the locked test set.

early: MLP over concat(BERT-CLS 768d, speech penultimate) trained end-to-end.
late : weighted soft voting of text & speech probabilities (weight tuned on val).

Prereqs:
  python -m sarcastone.models.bert_finetune --extract_embeddings
  python -m sarcastone.models.speech_cnn --extract_embeddings   (or speech_rnn)

Usage:
  python -m sarcastone.models.fusion --mode early
  python -m sarcastone.models.fusion --mode late --compare
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.error_analysis import write_error_report
from ..evaluation.metrics import compute_metrics, format_metrics, plot_confusion
from ..evaluation.significance import mcnemar_exact, paired_bootstrap_f1
from ..training.seed import set_seed
from ..training.trainer import fit
from ..utils import CKPT_DIR, EMB_DIR, REPORTS_DIR, SPLITS_DIR, save_json


class EarlyFusionMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 2)

    def embed(self, x):
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.embed(x))


def load_pair(split: str, speech_model: str = "cnn"):
    """Aligned text+speech embeddings for one split: X = [text_emb | speech_emb]."""
    t = np.load(EMB_DIR / f"text_{split}.npz", allow_pickle=True)
    s = np.load(EMB_DIR / f"speech_{speech_model}_{split}.npz", allow_pickle=True)
    tmap = {u: i for i, u in enumerate(t["ids"])}
    smap = {u: i for i, u in enumerate(s["ids"])}
    common = [u for u in t["ids"] if u in smap]
    ti = np.array([tmap[u] for u in common])
    si = np.array([smap[u] for u in common])
    X = np.hstack([t["emb"][ti], s["emb"][si]]).astype(np.float32)
    y = t["label"][ti].astype(np.int64)
    ids = pd.Series(common).astype(str).values
    assert len(set(ids)) == len(ids), "duplicate utterance ids"
    return X, y, ids


def probs_from_logits(model, X, device="cpu"):
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        return torch.softmax(model(xt), dim=1)[:, 1].cpu().numpy()


def run_early(device="cpu"):
    set_seed()
    Xtr, ytr, _ = load_pair("train")
    Xva, yva, _ = load_pair("val")
    Xte, yte, ids_te = load_pair("test")

    dl_tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                       batch_size=32, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=64)
    dl_te = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=64)

    model, hist, best = fit(EarlyFusionMLP(Xtr.shape[1]), dl_tr, dl_va,
                            epochs=80, lr=1e-3, weight_decay=1e-4, patience=10,
                            device=device, desc="fusion-early")
    ckpt = CKPT_DIR / "fusion_early"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "model.pt")

    probs = probs_from_logits(model, Xte, device)
    pred = (probs >= 0.5).astype(int)
    m = compute_metrics(yte, pred)
    print(f"\n== TEST (EARLY FUSION) ==\n{format_metrics(m)}")
    save_json(m, REPORTS_DIR / "phase3_early_test_metrics.json")
    plot_confusion(yte, pred, REPORTS_DIR / "figures" / "fusion_early_cm.png", "Early Fusion")

    gold = pd.read_csv(SPLITS_DIR / "test.csv").set_index("utt_id").loc[ids_te]
    write_error_report(gold.text.tolist(), yte, pred, probs, gold.speaker.values,
                       REPORTS_DIR / "phase3_early_error_analysis.md",
                       "Phase 3 errors: early fusion")
    return m, yte, probs


def _align(ids_a, arrays_a, ids_b):
    """Keep only ids present in both; return aligned copies of arrays_a + index into b."""
    mb = {u: i for i, u in enumerate(ids_b)}
    keep = [(i, u) for i, u in enumerate(ids_a) if u in mb]
    ia = np.array([k[0] for k in keep])
    ib = np.array([mb[k[1]] for k in keep])
    return np.asarray(ids_a)[ia], [a[ia] for a in arrays_a], ib


def text_probs(split: str, device: str = "cpu"):
    """P(sarcasm) per utterance from the fine-tuned BERT checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    tok = AutoTokenizer.from_pretrained(CKPT_DIR / "text_bert")
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR / "text_bert").to(device).eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            enc = tok(df.text[i:i + 64].tolist(), padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            ps.extend(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist())
            ys.extend(df.label[i:i + 64].tolist())
    return np.asarray(ps), np.asarray(ys), df.utt_id.astype(str).values


def speech_probs(split: str, arch: str = "cnn", device: str = "cpu"):
    """P(sarcasm) per utterance from the trained speech CNN/RNN checkpoint."""
    from .speech_cnn import SarcasmCNN
    from .speech_rnn import SarcasmRNN
    from .speech_data import SeqDataset

    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    dl = DataLoader(SeqDataset(df[["utt_id", "label"]]), batch_size=64)
    cls = {"cnn": SarcasmCNN, "rnn": SarcasmRNN}[arch]
    model = cls()
    state = torch.load(CKPT_DIR / f"speech_{arch}" / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    df_ids = df.utt_id.astype(str).values
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.extend(torch.softmax(model(x.to(device)), -1)[:, 1].cpu().tolist())
            ys.extend(y.tolist())
    return np.asarray(ps), np.asarray(ys), df_ids


def fuse(p_text, p_speech, w: float):
    return w * p_text + (1.0 - w) * p_speech


def run_late(arch: str = "cnn", device: str = "cpu"):
    """Late fusion = weighted soft voting; weight w tuned on val macro-F1.

    Returns test metrics plus all three probability vectors for compare().
    """
    from sklearn.metrics import f1_score

    tv, yv_t, iv = text_probs("val", device)
    sv, _, isv = speech_probs("val", arch, device)
    iv2, (tv, yv), sel_v = _align(iv, [tv, yv_t], isv)
    sv = sv[sel_v]

    best_w, best_f1 = 0.5, -1.0
    for w in np.arange(0.0, 1.0001, 0.05):
        f1 = f1_score(yv, (fuse(tv, sv, float(w)) >= .5).astype(int), average="macro")
        if f1 > best_f1:
            best_w, best_f1 = float(w), f1
    print(f"[late] tuned weight w={best_w:.2f} on val (macro F1={best_f1:.4f})")

    tt, yt_t, it = text_probs("test", device)
    stp, _, ist = speech_probs("test", arch, device)
    it2, (tt, yte), sel_t = _align(it, [tt, yt_t], ist)
    stp = stp[sel_t]

    probs = fuse(tt, stp, best_w)
    pred = (probs >= .5).astype(int)
    m = compute_metrics(yte, pred)
    print(f"\n== TEST (LATE FUSION w={best_w:.2f}) ==\n{format_metrics(m)}")
    save_json({**m, "fusion_weight": best_w}, REPORTS_DIR / "phase3_late_test_metrics.json")
    plot_confusion(yte, pred, REPORTS_DIR / "figures" / "fusion_late_cm.png",
                   f"Late Fusion (w={best_w:.2f})")

    gold = pd.read_csv(SPLITS_DIR / "test.csv").set_index("utt_id").loc[it2]
    write_error_report(gold.text.tolist(), yte, pred, probs, gold.speaker.values,
                       REPORTS_DIR / "phase3_late_error_analysis.md",
                       "Phase 3 errors: late fusion")
    return m, yte, tt, stp, probs


def compare(y_test, prob_text, prob_speech, prob_fusion, out="phase3_comparison"):
    """Comparison table + McNemar / paired-bootstrap significance vs baselines."""
    rows = {}
    for name, probs in [("text_only", prob_text), ("speech_only", prob_speech),
                        ("multimodal_fusion", prob_fusion)]:
        rows[name] = compute_metrics(y_test, (np.asarray(probs) >= 0.5).astype(int))

    table = pd.DataFrame({
        k: {"accuracy": v["accuracy"], "precision_macro": v["precision_macro"],
            "recall_macro": v["recall_macro"], "f1_macro": v["f1_macro"],
            "f1_sarcasm": v["f1_sarcasm"]}
        for k, v in rows.items()
    }).T.round(4)
    print("\n=== COMPARISON ON LOCKED TEST SET ===\n" + table.to_string())
    save_json(rows, REPORTS_DIR / f"{out}.json")

    tests = {
        "fusion_vs_text_mcnemar": mcnemar_exact(
            y_test, (np.asarray(prob_fusion) >= .5).astype(int),
            (np.asarray(prob_text) >= .5).astype(int)),
        "fusion_vs_text_bootstrap_dF1": paired_bootstrap_f1(y_test, prob_fusion, prob_text),
        "fusion_vs_speech_mcnemar": mcnemar_exact(
            y_test, (np.asarray(prob_fusion) >= .5).astype(int),
            (np.asarray(prob_speech) >= .5).astype(int)),
        "fusion_vs_speech_bootstrap_dF1": paired_bootstrap_f1(y_test, prob_fusion, prob_speech),
    }
    save_json(tests, REPORTS_DIR / f"{out}_significance.json")
    pvals = {k: round(v.get("p_value", float("nan")), 4) for k, v in tests.items() if "mcnemar" in k}
    print("\nsignificance (McNemar p-values):", pvals)
    print(f"saved -> reports/{out}.json, reports/{out}_significance.json")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["early", "late", "both"], default="both")
    ap.add_argument("--speech_model", default="cnn", choices=["cnn", "rnn"])
    ap.add_argument("--compare", action="store_true",
                    help="build comparison table + significance after fusion")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode in ("early", "both"):
        run_early(dev)
    if args.mode in ("late", "both"):
        _, yte, p_text, p_speech, p_fusion = run_late(args.speech_model, dev)
        if args.compare or args.mode == "both":
            compare(yte, p_text, p_speech, p_fusion)
