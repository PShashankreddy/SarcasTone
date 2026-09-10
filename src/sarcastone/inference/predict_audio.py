"""PHASE 2 deliverable: audio-only inference on a single wav file.

CLI: python -m sarcastone.inference.predict_audio data/raw/audio/1_10004.wav
Python:
    from sarcastone.inference.predict_audio import AudioSarcasmPredictor
    AudioSarcasmPredictor().predict("clip.wav")
"""
import argparse

from ..utils import CKPT_DIR

MAX_FRAMES = 200


class AudioSarcasmPredictor:
    def __init__(self, arch: str = "cnn", device=None):
        import torch
        from ..models.speech_cnn import SarcasmCNN
        from ..models.speech_rnn import SarcasmRNN
        from ..models.speech_data import load_norm
        from ..features.acoustic_librosa import apply_normalizer

        if arch not in ("cnn", "rnn"):
            raise ValueError("arch must be 'cnn' or 'rnn'")
        self.torch = torch
        self.apply_normalizer = apply_normalizer
        self.norm = load_norm()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SarcasmCNN() if arch == "cnn" else SarcasmRNN()
        state = torch.load(CKPT_DIR / f"speech_{arch}" / "model.pt",
                           map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict_proba_file(self, path) -> float:
        import numpy as np
        from ..features.acoustic_librosa import extract_sequence

        seq = extract_sequence(str(path))[:MAX_FRAMES]
        seq = self.apply_normalizer(seq, self.norm)
        padded = np.zeros((MAX_FRAMES, seq.shape[1]), dtype=np.float32)
        padded[:len(seq)] = seq
        x = self.torch.from_numpy(padded.T).unsqueeze(0)      # (1, D=40, T=200)
        return self._forward(x)

    def _forward(self, x) -> float:
        with self.torch.no_grad():
            logits = self.model(x.to(self.device))
            return float(self.torch.softmax(logits, -1)[0, 1].item())

    def predict(self, path) -> dict:
        prob = self.predict_proba_file(str(path))
        return {"label": "sarcasm" if prob >= 0.5 else "non-sarcastic",
                "prob_sarcasm": round(prob, 4), "file": str(path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="path to 16 kHz mono wav clip")
    args = ap.parse_args()
    print(AudioSarcasmPredictor().predict(args.wav))
