#!/usr/bin/env python3
"""Qwen3.8-27B face-age judgement pipeline.

  python cli.py setup      check environment, Ollama, model, images
  python cli.py images     download the 9,150 face images (~87 MB)
  python cli.py pairs      build/inspect the pair list
  python cli.py run        run the judgements (resumable)
  python cli.py run --mock exercise the whole pipeline with no GPU
  python cli.py sync       rebuild xlsx/parquet from the JSONL log
  python cli.py analyze    accuracy, position bias, Hodge decomposition
  python cli.py status     progress so far

Runs on Linux, macOS and Windows.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline.config import REPO_ROOT, Config   # noqa: E402


def cmd_setup(cfg: Config, args) -> None:
    print(f"python      : {sys.version.split()[0]} ({sys.platform})")
    print(f"repo root   : {REPO_ROOT}")
    print()

    missing = []
    for mod in ("pandas", "numpy", "openpyxl", "pyarrow", "requests",
                "yaml", "tqdm", "PIL", "scipy"):
        try:
            __import__(mod)
            print(f"  {mod:10s} ok")
        except ImportError:
            print(f"  {mod:10s} MISSING")
            missing.append(mod)
    if missing:
        print(f"\ninstall with:  pip install -r {HERE.name}/requirements.txt")

    print()
    for label, p in (("gt.csv", cfg.gt_csv), ("crowd_labels.csv", cfg.crowd_csv)):
        print(f"  {label:18s} {'ok' if p.exists() else 'MISSING'}  {p}")

    from pipeline.images import verify
    try:
        have, want = verify(cfg)
        print(f"  images             {have:,}/{want:,} downloaded")
        if have < want:
            print(f"                     run:  python {HERE.name}/cli.py images")
    except Exception as exc:                            # noqa: BLE001
        print(f"  images             cannot check ({exc})")

    print()
    if cfg.provider == "ollama":
        print(f"  ollama binary      {'found' if shutil.which('ollama') else 'NOT on PATH'}")
    from pipeline.model import make_judge
    try:
        ok, msg = make_judge(cfg).health()
    except RuntimeError as exc:
        ok, msg = False, str(exc)
    print(f"  {cfg.provider} service     {'ok' if ok else 'not ready'}")
    for line in msg.splitlines():
        print(f"                     {line}")
    if not ok:
        print(f"\n  (you can still validate everything with:  "
              f"python {HERE.name}/cli.py run --mock --limit 50)")


def cmd_images(cfg: Config, args) -> None:
    from pipeline.images import download_all
    download_all(cfg, workers=args.workers)


def cmd_pairs(cfg: Config, args) -> None:
    from pipeline.pairs import build_and_save
    build_and_save(cfg)


def cmd_run(cfg: Config, args) -> None:
    from pipeline.run import run
    run(cfg, mock=args.mock, limit=args.limit)


def cmd_sync(cfg: Config, args) -> None:
    from pipeline.store import rebuild_snapshots
    rebuild_snapshots(cfg)


def cmd_analyze(cfg: Config, args) -> None:
    from pipeline.analyze import analyse
    analyse(cfg)


def cmd_status(cfg: Config, args) -> None:
    from pipeline.store import JudgmentStore
    store = JudgmentStore(cfg)
    n = len(store)
    if not n:
        print("no judgements recorded yet")
        return
    df = store.to_frame()
    scored = df[df["model_correct"].notna()]
    print(f"judgements : {n:,}")
    print(f"pairs      : {df['pair_id'].nunique():,}")
    print(f"errors     : {int(df['error'].notna().sum()):,}")
    if len(scored):
        print(f"model acc  : {100*scored['model_correct'].mean():.2f}%")
    for p in (cfg.jsonl_path, cfg.parquet_path, cfg.excel_path):
        mark = f"{p.stat().st_size/1e6:.1f} MB" if p.exists() else "not written"
        print(f"  {p.name:22s} {mark}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # shared flags, accepted either before or after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=None, help="path to config.yaml")
    common.add_argument("--out", default=None,
                        help="override output.dir (keeps separate runs apart)")

    # top-level copies use distinct dests so the subparser defaults (None)
    # cannot clobber a value given before the subcommand
    ap.add_argument("-c", "--config", dest="config_top", default=None,
                    help="path to config.yaml")
    ap.add_argument("--out", dest="out_top", default=None,
                    help="override output.dir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", parents=[common],
                   help="check environment and prerequisites")
    p_img = sub.add_parser("images", parents=[common],
                           help="download face images (~136 MB)")
    p_img.add_argument("--workers", type=int, default=16)
    sub.add_parser("pairs", parents=[common], help="build the pair list")
    p_run = sub.add_parser("run", parents=[common], help="run judgements (resumable)")
    p_run.add_argument("--mock", action="store_true",
                       help="use the offline mock judge, no GPU needed")
    p_run.add_argument("--limit", type=int, default=None,
                       help="stop after N judgements this session")
    sub.add_parser("sync", parents=[common],
                   help="rebuild xlsx/parquet from the JSONL log")
    sub.add_parser("analyze", parents=[common],
                   help="accuracy, bias, Hodge decomposition")
    sub.add_parser("status", parents=[common], help="show progress")

    args = ap.parse_args()
    config = getattr(args, "config", None) or getattr(args, "config_top", None)
    out = getattr(args, "out", None) or getattr(args, "out_top", None)

    cfg = Config.load(config)
    if out:
        cfg.out_dir = Path(out) if Path(out).is_absolute() else (REPO_ROOT / out)
        cfg.ensure_dirs()
    {"setup": cmd_setup, "images": cmd_images, "pairs": cmd_pairs, "run": cmd_run,
     "sync": cmd_sync, "analyze": cmd_analyze, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
