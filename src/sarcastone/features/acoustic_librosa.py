"""Frame-level acoustic sequences for CNN/RNN: log-MFCC(13)+deltas(26)+RMS energy = 40 dims.

Output per utterance: npy array of shape (T, 40). Sequences are padded/truncated to a
fixed length by the Dataset classes, and z-scored with train-set stats in build_features.
"""
import numpy as np
import librosa

N_MFCC = 13
SR = 16000

FEATURE_DIM = 3 * N_MFCC + 1


def extract_sequence(wav_path: str) -> np.ndarray:
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=50)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    T = min(mfcc.shape[1], rms.shape[1])
    seq = np.concatenate(
        [mfcc[:, :T].T, d1[:, :T].T, d2[:, :T].T, rms[0, :T][:, None]], axis=1
    ).astype(np.float32)                     # (T, 40)
    return seq


def fit_normalizer(seqs: list[np.ndarray]):
    all_frames = np.vstack(seqs)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0) + 1e-8
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_normalizer(seq: np.ndarray, norm: dict) -> np.ndarray:
    return ((seq - np.asarray(norm["mean"])) / np.asarray(norm["std"])).astype(np.float32)
