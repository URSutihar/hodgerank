"""Fetch pinned official IMDB-WIKI-SbS metadata without redistributing it."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import requests

from .config import Config

REVISION = "6087435b6eb61993c1169e232a827fd18b51a7c1"
BASE_URL = f"https://raw.githubusercontent.com/Toloka/IMDB-WIKI-SbS/{REVISION}/data"
FILES = {
    "gt.csv": ("47281f24209f4566f0e8d0e0615b3d249f8045e36260c8dfa4e44b80f1c31d08", {"label", "score"}),
    "crowd_labels.csv": ("2307300380f8ccdb09255970cd6b013ce0fcddf74545fb0f2ca4b0eca0a9220f", {"left", "right", "label", "performer"}),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate(path: Path, name: str) -> None:
    expected_hash, columns = FILES[name]
    if digest(path) != expected_hash:
        raise ValueError(f"{path}: checksum mismatch; expected pinned official {name}")
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
    if not columns.issubset(header):
        raise ValueError(f"{path}: missing required columns {sorted(columns)}")


def download_metadata(cfg: Config) -> None:
    for name, target in (("gt.csv", cfg.gt_csv), ("crowd_labels.csv", cfg.crowd_csv)):
        if target.exists():
            validate(target, name)
            print(f"{name}: present, checksum verified")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
            validate(temporary, name)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{name}: downloaded, checksum verified ({target.stat().st_size / 1e6:.1f} MB)")
