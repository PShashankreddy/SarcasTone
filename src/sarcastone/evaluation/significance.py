"""Statistical significance: exact McNemar test + paired bootstrap for dF1.

Both tests assume two models predicting on the SAME locked test set.
"""
import math

import numpy as np
from sklearn.metrics import f1_score


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact binomial McNemar on discordant pairs (b = A right/B wrong, c = A wrong/B right)."""
    y_true = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    correct_a, correct_b = a == y_true, b == y_true
    n01 = int(np.sum(correct_a & ~correct_b))   # A better here
    n10 = int(np.sum(~correct_a & correct_b))   # B better here
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    p = min(float(p), 1.0)
    return {"n01": n01, "n10": n10, "p_value": p}


def paired_bootstrap_f1(y_true, prob_a, prob_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI + p-value for F1(A) - F1(B) using macro F1 at 0.5 threshold."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pa = (np.asarray(prob_a) >= 0.5).astype(int)
    pb = (np.asarray(prob_b) >= 0.5).astype(int)
    idx = np.arange(len(y_true))

    deltas, worse_count = [], 0
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        d = f1_score(y_true[sample], pa[sample], average="macro", zero_division=0) - \
            f1_score(y_true[sample], pb[sample], average="macro", zero_division=0)
        deltas.append(d)
        worse_count += d <= 0
    deltas = np.array(deltas)
    alpha = 0.05
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_f1_point": float(f1_score(y_true, pa, average="macro") - f1_score(y_true, pb, average="macro")),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value_approx": float((worse_count + 1) / (n_boot + 1)),   # add-one H0 proportion
        "n_boot": n_boot,
    }
