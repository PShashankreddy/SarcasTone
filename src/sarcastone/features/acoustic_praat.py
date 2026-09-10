"""Praat acoustic summary features via parselmouth (pitch, energy/intensity, HNR,
jitter, shimmer, duration). Handles unvoiced frames (Praat pitch=0/NaN) by using
robust percentiles instead of min/max.

Usage (as library):
    from sarcastone.features.acoustic_praat import extract_summary
"""
import numpy as np
import parselmouth
from parselmouth.praat import call

SUMMARY_KEYS = [
    "duration_s", "f0_mean_hz", "f0_std_hz", "f0_p05", "f0_p95", "f0_range",
    "intensity_mean_db", "intensity_std_db",
    "hnr_mean_db",
    "jitter_local", "shimmer_local",
]


def extract_summary(wav_path: str) -> dict:
    snd = parselmouth.Sound(str(wav_path))

    pitch = call(snd, "To Pitch", 0.0, 75, 600)          # 75-600 Hz voice range
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]                                   # drop unvoiced frames
    if len(voiced) == 0:
        voiced = np.array([np.nan])

    intensity = call(snd, "To Intensity", 100, 0.0, "yes")
    ivals = intensity.values[0]

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = harmonicity.values[0]
    hnr_v = hnr[np.isfinite(hnr)]

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        "duration_s": round(snd.get_total_duration(), 4),
        "f0_mean_hz": float(np.mean(voiced)),
        "f0_std_hz": float(np.std(voiced)),
        "f0_p05": float(np.percentile(voiced, 5)) if len(voiced) > 1 else np.nan,
        "f0_p95": float(np.percentile(voiced, 95)) if len(voiced) > 1 else np.nan,
        "f0_range": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)) if len(voiced) > 1 else np.nan,
        "intensity_mean_db": float(np.mean(ivals)),
        "intensity_std_db": float(np.std(ivals)),
        "hnr_mean_db": float(np.mean(hnr_v)) if len(hnr_v) else np.nan,
        "jitter_local": float(jitter),
        "shimmer_local": float(shimmer),
    }
