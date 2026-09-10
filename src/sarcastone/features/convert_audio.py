"""Convert collected MUStARD++ clips to uniform 16 kHz mono WAV.

Input: data/raw/audio_raw/{utt_id}.* (mp4/mkv/mp3...)
Output: data/raw/audio/{utt_id}.wav

Usage: python -m sarcastone.features.convert_audio
"""
import shutil
import subprocess
from pathlib import Path

from ..utils import DATA_RAW

SRC_DIR = DATA_RAW / "audio_raw"
DST_DIR = DATA_RAW / "audio"


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: PATH first, else imageio-ffmpeg's bundled one."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit(
            "ffmpeg not on PATH; run: pip install imageio-ffmpeg (bundled binary)"
        ) from exc


def convert(sample_rate: int = 16000):
    ffmpeg = ffmpeg_exe()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".flv"}
    clips = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts] if SRC_DIR.exists() else []
    print(f"{len(clips)} source clips in {SRC_DIR}")
    for src in clips:
        dst = DST_DIR / (src.stem + ".wav")
        cmd = [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", str(sample_rate), str(dst)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if res.returncode == 0 else "FAIL"
        print(f"{status:>4}  {src.name} -> {dst.name}")


if __name__ == "__main__":
    convert()
