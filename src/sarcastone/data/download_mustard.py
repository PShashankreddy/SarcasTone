"""Download MUStARD++ annotation file (sarcasm_data.json). Audio is collected separately.

Usage: python -m sarcastone.data.download_mustard
"""
import json
import urllib.request

from ..utils import MUSTARD_JSON, DATA_RAW

URL = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"


def download_mustard(force: bool = False):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if MUSTARD_JSON.exists() and not force:
        print(f"[skip] {MUSTARD_JSON} already exists")
    else:
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, MUSTARD_JSON)
        print(f"saved -> {MUSTARD_JSON}")
    data = json.loads(MUSTARD_JSON.read_text(encoding="utf-8"))
    n_sarc = sum(1 for v in data.values() if v["sarcasm"])
    speakers = sorted({v["speaker"] for v in data.values()})
    print(f"{len(data)} utterances | sarcastic={n_sarc} non-sarcastic={len(data)-n_sarc}")
    print(f"{len(speakers)} speakers: {', '.join(speakers)}")
    print("NOTE: collect audio clips separately into data/raw/audio/{utt_id}.wav (see README).")
    return data


if __name__ == "__main__":
    download_mustard()
