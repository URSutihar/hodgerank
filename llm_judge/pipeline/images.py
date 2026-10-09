"""Download the 9,150 IMDB-WIKI-SbS face images locally (~87 MB total).

Resumable: already-present files are skipped, so re-running costs nothing.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .config import Config

TIMEOUT = 30
WORKERS = 16


def image_urls(cfg: Config) -> dict[str, str]:
    """filename -> full URL, taken from gt.csv."""
    gt = pd.read_csv(cfg.gt_csv)
    return {u.split("/")[-1]: u for u in gt["label"].astype(str)}


def _fetch(name: str, url: str, dest: Path) -> tuple[str, str | None]:
    target = dest / name
    if target.exists() and target.stat().st_size > 0:
        return name, None
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        # write via temp then rename so an interrupted run never leaves a partial file
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(target)
        return name, None
    except Exception as exc:                      # noqa: BLE001
        return name, str(exc)


def download_all(cfg: Config, workers: int = WORKERS) -> dict[str, str]:
    """Fetch every image. Returns {filename: error} for whatever failed."""
    cfg.ensure_dirs()
    urls = image_urls(cfg)
    missing = {n: u for n, u in urls.items()
               if not (cfg.image_dir / n).exists()
               or (cfg.image_dir / n).stat().st_size == 0}

    print(f"{len(urls):,} images referenced, {len(missing):,} still to download")
    if not missing:
        print("all present — nothing to do")
        return {}

    errors: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_fetch, n, u, cfg.image_dir) for n, u in missing.items()]
        for f in tqdm(cf.as_completed(futs), total=len(futs), unit="img"):
            name, err = f.result()
            if err:
                errors[name] = err

    have = sum(1 for n in urls if (cfg.image_dir / n).exists())
    total_mb = sum((cfg.image_dir / n).stat().st_size
                   for n in urls if (cfg.image_dir / n).exists()) / 1e6
    print(f"\n{have:,}/{len(urls):,} images on disk ({total_mb:.1f} MB)")
    if errors:
        print(f"{len(errors)} failed; re-run to retry. First few:")
        for n, e in list(errors.items())[:5]:
            print(f"  {n}: {e}")
    return errors


def verify(cfg: Config) -> tuple[int, int]:
    """(present, expected) — cheap check before a run."""
    urls = image_urls(cfg)
    present = sum(1 for n in urls
                  if (cfg.image_dir / n).exists() and (cfg.image_dir / n).stat().st_size > 0)
    return present, len(urls)
