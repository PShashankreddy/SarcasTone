"""PHASE 1 deliverable: text-only sarcasm inference.

CLI: python -m sarcastone.inference.predict_text "Oh great, another Monday."
Python:
    from sarcastone.inference.predict_text import predict_text
    predict_text("Well that went *perfectly*.")
"""
import argparse

from ..utils import CKPT_DIR


class TextSarcasmPredictor:
    def __init__(self, checkpoint_dir=None, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ckpt = str(checkpoint_dir or CKPT_DIR / "text_bert")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(self.device).eval()
        self._torch = torch

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(text, truncation=True, max_length=128,
                             return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**enc).logits, dim=-1)[0]
        p = float(probs[1])
        return {"text": text,
                "label": "sarcasm" if p >= 0.5 else "non-sarcastic",
                "prob_sarcasm": round(p, 4)}


def predict_text(text: str) -> dict:
    return TextSarcasmPredictor().predict(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--checkpoint", default=None,
                    help="e.g. checkpoints/text_roberta (default: checkpoints/text_bert)")
    args = ap.parse_args()
    pred = TextSarcasmPredictor(checkpoint_dir=args.checkpoint)
    for t in [" ".join(args.text)]:
        print(pred.predict(t))
