"""Metrics: accuracy, macro precision/recall/F1, sarcastic-class F1, confusion matrix."""
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_sarcasm": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # rows=true [non-sarc, sarc]
        "n": int(len(y_true)),
    }


def format_metrics(m: dict) -> str:
    cm = np.array(m["confusion_matrix"])
    return (
        f"acc={m['accuracy']:.4f}  P={m['precision_macro']:.4f}  R={m['recall_macro']:.4f}  "
        f"F1(macro)={m['f1_macro']:.4f}  F1(sarcasm)={m['f1_sarcasm']:.4f}\n"
        f"confusion [[TN FP] / [FN TP]]:\n{cm}"
    )


def plot_confusion(y_true, y_pred, out_path: Path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["non-sarc", "sarcasm"], yticklabels=["non-sarc", "sarcasm"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
