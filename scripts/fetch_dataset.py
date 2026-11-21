"""Downloads the real, public PhiUSIIL Phishing URL Dataset (UCI repository id
967, CC BY 4.0) into data_raw/ (gitignored) and verifies it against the
sha256 recorded in data/manifest.json, so a fresh checkout can reproduce the
exact file every benchmark in this repo was measured against.

Usage: python scripts/fetch_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DATA_RAW.mkdir(exist_ok=True)
    zip_path = DATA_RAW / "phiusiil.zip"
    csv_path = DATA_RAW / MANIFEST["csv_filename"]
    if csv_path.exists():
        print(f"already present: {csv_path}")
        return
    print(f"downloading {MANIFEST['download_url']}")
    urllib.request.urlretrieve(MANIFEST["download_url"], zip_path)
    digest = _sha256(zip_path)
    if digest != MANIFEST["zip_sha256"]:
        print(f"CHECKSUM MISMATCH: got {digest}, expected {MANIFEST['zip_sha256']}", file=sys.stderr)
        sys.exit(1)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_RAW)
    print(f"verified and extracted: {csv_path}")


if __name__ == "__main__":
    main()
