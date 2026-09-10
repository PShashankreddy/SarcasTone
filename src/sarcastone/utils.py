"""Project paths and small shared helpers. Paths are repo-relative, CWD-independent."""
from pathlib import Path
import json
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
AUDIO_RAW = DATA_RAW / "audio"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED / "splits"
FEATURES_SEQ_DIR = DATA_PROCESSED / "features_seq"
SUMMARY_CSV = DATA_PROCESSED / "acoustic_summary.csv"
TRANSCRIPTS_CSV = DATA_INTERIM / "whisper_transcripts.csv"

EMB_DIR = PROJECT_ROOT / "embeddings"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"

MUSTARD_JSON = DATA_RAW / "sarcasm_data.json"

LABEL2ID = {False: 0, True: 1}   # MUStARD++ 'sarcasm' bool -> int
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SEED = 42


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
