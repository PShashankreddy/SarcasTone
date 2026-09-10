"""Multimodal inference on one (text, wav) pair using the late-fusion weight.

CLI: python -m sarcastone.inference.predict_multimodal --text "..." --audio clip.wav
"""
import argparse
import json

import pandas as pd

from ..utils import REPORTS_DIR


def get_fusion_weight(default=0.5) -> float:
    path = REPORTS_DIR / "phase3_late_test_metrics.json"
    if path.exists():
        return float(pd.read_json(path, typ="series")["fusion_weight"])
    print(f"[warn] {path.name} not found; defaulting w={default}")
    return default


def predict_multimodal(text: str, wav_path: str, w: float | None = None) -> dict:
    from .predict_audio import AudioSarcasmPredictor
    from .predict_text import TextSarcasmPredictor

    w = get_fusion_weight() if w is None else w
    t_res = TextSarcasmPredictor().predict(text)
    a_res = AudioSarcasmPredictor().predict(wav_path)
    fused = w * t_res["prob_sarcasm"] + (1 - w) * a_res["prob_sarcasm"]
    return {
        "text": t_res,
        "audio": a_res,
        "fusion_weight_w": round(w, 3),
        "prob_sarcasm_fused": round(fused, 4),
        "label": "sarcasm" if fused >= 0.5 else "non-sarcastic",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--w", type=float, default=None)
    args = ap.parse_args()
    print(json.dumps(predict_multimodal(args.text, args.audio, args.w), indent=2))
