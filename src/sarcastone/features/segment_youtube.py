"""Custom dataset builder: YouTube videos -> utterance-aligned clips + context.

Pipeline (research use only; keep media out of git):
  1. yt-dlp download video/audio
  2. Whisper -> timestamped segments
  3. cut one wav clip per segment (utterance boundaries, not fixed windows)
  4. store 2-3 preceding utterances as annotation context
  5. write manifest data/interim/custom_clips/context.json

Usage:
  python -m sarcastone.features.segment_youtube --url <youtube_url> [--max_clips 200]
Requires: pip install yt-dlp; ffmpeg (bundled via imageio-ffmpeg or on PATH).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from ..utils import DATA_INTERIM
from .convert_audio import ffmpeg_exe

CLIP_DIR = DATA_INTERIM / "custom_clips"
MIN_DUR = 0.8     # skip micro-segments; kept loose for punchline interjections
MAX_DUR = 16.0    # rejects whole-scene monologues
PAD = 0.15        # tiny head/tail pad so ffmpeg decoder lands cleanly


def _download(url: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    out = work / "source.%(ext)s"
    # invoke via the running interpreter so the venv's yt-dlp is always found
    subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "bestaudio/best",
                    "-o", str(out), url], check=True)
    files = [p for p in work.iterdir() if p.stem == "source"]
    if not files:
        raise SystemExit("yt-dlp produced no file")
    return files[0]


def _to_wav(src: Path) -> Path:
    dst = src.with_suffix(".wav")
    subprocess.run([ffmpeg_exe(), "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    str(dst)], check=True, capture_output=True)
    return dst


def _cut_and_manifest(vid: str, wav: Path, segments: list, max_clips: int):
    """Shared clip-cutting + manifest-append used by build() and refilter()."""
    manifest, prev_texts = [], []
    segs = [s for s in segments
            if MIN_DUR <= (s["end"] - s["start"]) <= MAX_DUR][:max_clips]
    print(f"{len(segs)} candidate clips")
    for i, s in enumerate(tqdm(segs)):
        clip = CLIP_DIR / vid / f"seg{i:04d}.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        lo = max(0.0, s["start"] - PAD)
        hi = s["end"] + PAD
        subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", str(lo), "-to", str(hi),
             "-i", str(wav), "-ac", "1", "-ar", "16000", str(clip)],
            check=True, capture_output=True)
        manifest.append({
            "utt_id": f"{vid}_seg{i:04d}",
            "clip_path": str(clip),
            "text": s["text"].strip(),
            "start": round(s["start"], 2), "end": round(s["end"], 2),
            "context_before": list(prev_texts[-3:]),   # last up-to-3 utterances
        })
        prev_texts.append(s["text"].strip())

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out = CLIP_DIR / "context.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    out.write_text(json.dumps(existing + manifest, indent=1), encoding="utf-8")
    print(f"manifest: {len(manifest)} new clips appended -> {out}")
    print("next: python -m sarcastone.data.annotation make-template")


def refilter(url: str, max_clips: int = 200):
    """Re-cut clips from persisted segments (no transcription)."""
    vid = url.split("v=")[-1].split("&")[0] or "video"
    work = CLIP_DIR / f"_work_{vid}"
    segjson = work / "_segments.json"
    wav = work / "source.wav"
    if not segjson.exists():
        raise SystemExit(f"no persisted segments at {segjson}; run build first")
    segments = json.loads(segjson.read_text(encoding="utf-8"))
    clipdir = CLIP_DIR / vid
    if clipdir.exists():
        import shutil
        shutil.rmtree(clipdir)
    _cut_and_manifest(vid, wav, segments, max_clips)


def build(url: str, max_clips: int = 200, model_name: str = "small"):
    from .asr import load_asr

    vid = url.split("v=")[-1].split("&")[0] or "video"
    work = CLIP_DIR / f"_work_{vid}"
    cached = work / "source.wav"
    if cached.exists():                       # resume support
        wav = cached
        print("reusing cached audio...")
    else:
        print("downloading...")
        src = _download(url, work)
        wav = _to_wav(src)

    print(f"transcribing with whisper '{model_name}' (windowed / drift-free mode)...")
    model = load_asr(model_name)
    # Windowed transcription: short inputs give accurate timestamps; long-form
    # chunking drifts badly on 10+ minute videos (see docs/PHASE2_PLAIN_ENGLISH.md).
    result = model.transcribe_windowed(str(wav), language="english")
    # persist raw segments so clip filters can be re-tuned without re-transcription
    work.joinpath("_segments.json").write_text(
        json.dumps(result["segments"], indent=1), encoding="utf-8")

    _cut_and_manifest(vid, wav, result["segments"], max_clips)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--max_clips", type=int, default=200)
    ap.add_argument("--model", default="small")
    ap.add_argument("--refilter", action="store_true",
                    help="re-cut clips from persisted segments (no ASR rerun)")
    args = ap.parse_args()
    args.url and (None, refilter(args.url, args.max_clips) if args.refilter
                  else build(args.url, args.max_clips, args.model))
