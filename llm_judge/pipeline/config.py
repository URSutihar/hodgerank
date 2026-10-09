"""Config loading and path resolution. Works identically on Linux, macOS, Windows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo root = parent of the llm_judge package
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path against the repo root unless already absolute."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    # model
    model_name: str = "qwen3-vl:8b-instruct-q4"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 42
    timeout_s: int = 180
    num_ctx: int = 4096
    keep_alive: str = "10m"
    max_retries: int = 3
    provider: str = "ollama"
    keychain_service: str = "deepseek-anthropic"
    keychain_account: str = "urs"

    # data
    gt_csv: Path = REPO_ROOT / "data/imdb_sbs/gt.csv"
    crowd_csv: Path = REPO_ROOT / "data/imdb_sbs/crowd_labels.csv"
    image_dir: Path = REPO_ROOT / "data/imdb_sbs/images"

    # run
    n_pairs: int | None = 2000
    stratify_by_gap: bool = True
    run_seed: int = 7
    swap_repeat: bool = True
    save_every: int = 25
    max_image_px: int = 512

    # output
    out_dir: Path = REPO_ROOT / "llm_judge/results"
    jsonl_name: str = "judgments.jsonl"
    parquet_name: str = "judgments.parquet"
    excel_name: str = "judgments.xlsx"

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        cfg_path = Path(path) if path else (REPO_ROOT / "llm_judge/config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        m, d, r, o = (raw.get(k, {}) or {} for k in ("model", "data", "run", "output"))
        n_pairs = r.get("n_pairs", 2000)
        if n_pairs in (0, None, "all"):
            n_pairs = None

        return cls(
            raw=raw,
            model_name=m.get("name", cls.model_name),
            provider=m.get("provider", "ollama"),
            keychain_service=m.get("keychain_service", "deepseek-anthropic"),
            keychain_account=m.get("keychain_account", "urs"),
            host=str(m.get("host", cls.host)).rstrip("/"),
            temperature=float(m.get("temperature", 0.0)),
            seed=int(m.get("seed", 42)),
            timeout_s=int(m.get("timeout_s", 180)),
            num_ctx=int(m.get("num_ctx", 4096)),
            keep_alive=str(m.get("keep_alive", "10m")),
            max_retries=int(m.get("max_retries", 3)),
            gt_csv=_resolve(d.get("gt_csv", "data/imdb_sbs/gt.csv")),
            crowd_csv=_resolve(d.get("crowd_csv", "data/imdb_sbs/crowd_labels.csv")),
            image_dir=_resolve(d.get("image_dir", "data/imdb_sbs/images")),
            n_pairs=n_pairs,
            stratify_by_gap=bool(r.get("stratify_by_gap", True)),
            run_seed=int(r.get("seed", 7)),
            swap_repeat=bool(r.get("swap_repeat", True)),
            save_every=int(r.get("save_every", 25)),
            max_image_px=int(r.get("max_image_px", 512)),
            out_dir=_resolve(o.get("dir", "llm_judge/results")),
            jsonl_name=o.get("jsonl", "judgments.jsonl"),
            parquet_name=o.get("parquet", "judgments.parquet"),
            excel_name=o.get("excel", "judgments.xlsx"),
        )

    # convenience paths
    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / self.jsonl_name

    @property
    def parquet_path(self) -> Path:
        return self.out_dir / self.parquet_name

    @property
    def excel_path(self) -> Path:
        return self.out_dir / self.excel_name

    @property
    def pairs_path(self) -> Path:
        return self.out_dir / "pairs.parquet"

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
