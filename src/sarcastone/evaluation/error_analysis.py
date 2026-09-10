"""Error analysis: markdown report of false positives/negatives ranked by confidence."""
import pandas as pd


def write_error_report(texts, y_true, y_pred, probs, speakers, out_path, title):
    df = pd.DataFrame({
        "text": texts, "true": ["sarcasm" if t else "non-sarc" for t in y_true],
        "pred": ["sarcasm" if p else "non-sarc" for p in y_pred],
        "prob_sarcasm": [round(float(p), 4) for p in probs],
        "speaker": speakers,
        "confidence_gap": [round(abs(float(p) - 0.5), 4) for p in probs],
    })
    fp = df[(df["true"] == "non-sarc") & (df.pred == "sarcasm")].sort_values("confidence_gap", ascending=False)
    fn = df[(df["true"] == "sarcasm") & (df.pred == "non-sarc")].sort_values("confidence_gap", ascending=False)

    def md_table(d: pd.DataFrame) -> str:
        if len(d) == 0:
            return "_none_\n"
        rows = ["| text | true | pred | P(sarcasm) | speaker |",
                "|---|---|---|---|---|"]
        for _, r in d.head(20).iterrows():
            t = r.text.replace("|", "\\|")
            rows.append(f"| {t} | {r['true']} | {r.pred} | {r.prob_sarcasm} | {r.speaker} |")
        return "\n".join(rows) + "\n"

    lines = [
        f"# {title}\n",
        f"- total samples: **{len(df)}**",
        f"- false positives (predicted sarcasm, actually sincere): **{len(fp)}**",
        f"- false negatives (missed sarcasm): **{len(fn)}**\n",
        "## Top false positives (most confident mistakes)\n", md_table(fp),
        "## Top false negatives (most confident mistakes)\n", md_table(fn),
        "> Fusion hypothesis to check: do FN/FP clusters differ between text-only and\n"
        "> multimodal models? Compare sample IDs across phase reports.\n",
    ]
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"error report -> {out_path}  (FP={len(fp)}, FN={len(fn)})")
    return fp, fn
