"""Whisper ASR via HuggingFace transformers (Windows-safe: no triton needed).

Exposes an openai-whisper-compatible interface so callers can do:

    model = load_asr("small")
    result = model.transcribe("clip.wav", language="en")
    result["text"]                 -> str
    result["segments"]             -> [{"start", "end", "text"}]
"""
import re

_HF_PREFIX = "openai/whisper-"
_VALID = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"}


def _resolve(model_name: str) -> str:
    if model_name.startswith("openai/whisper"):
        return model_name
    if model_name not in _VALID:
        raise ValueError(f"unknown whisper model '{model_name}'; choose from {sorted(_VALID)}")
    return _HF_PREFIX + model_name


class _TransformersWhisper:
    def __init__(self, model_name: str):
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=_resolve(model_name),
            device=-1,  # CPU
        )

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _load_mono(self, audio_path) -> tuple:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:
            import librosa

            mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        return np.asarray(mono, dtype=np.float32), 16000

    def transcribe(self, audio_path, language: str = "english") -> dict:
        """Full-file transcription. NOTE: long inputs drift timestamps;
        use transcribe_windowed for anything needing accurate boundaries."""
        mono, sr = self._load_mono(audio_path)
        out = self._pipe(
            {"array": mono, "sampling_rate": sr},
            return_timestamps=True,
            generate_kwargs={"language": language},
        )
        segments = [
            {
                "start": float(chunk["timestamp"][0] or 0.0),
                "end": float(chunk["timestamp"][1] or 0.0),
                "text": self._clean(chunk["text"]),
            }
            for chunk in out.get("chunks", [])
        ]
        text = self._clean(out["text"])
        return {"text": text, "segments": segments}

    def transcribe_windowed(self, audio_path, window_s: float = 28.0,
                            language: str = "english") -> dict:
        """Accurate-timestamp transcription: process the audio in short windows
        (< the pipeline's 30s chunk limit) so every segment's timestamps come from
        a short, drift-free input, then offset them to full-file time.

        This is the fix for long-form timestamp drift (see docs/).
        """
        import numpy as np

        mono, sr = self._load_mono(audio_path)
        win = int(sr * window_s)
        n = len(mono)
        t0 = 0.0
        segments, texts = [], []
        while t0 * sr < n:
            lo = int(t0 * sr)
            chunk = mono[lo: min(n, lo + win)]
            if len(chunk) < sr * 0.5:          # trivial tail: drop
                break
            out = self._pipe(
                {"array": chunk, "sampling_rate": sr},
                return_timestamps=True,
                generate_kwargs={"language": language},
            )
            for c in out.get("chunks", []):
                a, b = c["timestamp"]
                if a is None or b is None:
                    continue          # truncated/open-ended chunk: no usable span
                txt = self._clean(c["text"])
                if not txt:
                    continue
                segments.append({
                    "start": round(t0 + float(a), 2),
                    "end": round(t0 + float(b), 2),
                    "text": txt,
                })
                texts.append(txt)
            t0 += window_s   # seconds, not samples — the loop maths in seconds
        return {"text": " ".join(texts), "segments": segments}


def load_asr(model_name: str = "small") -> _TransformersWhisper:
    print(f"loading ASR model '{model_name}' (first run downloads weights)...")
    return _TransformersWhisper(model_name)
