"""Main judgement loop: resumable, interrupt-safe, progress-reporting."""

from __future__ import annotations

import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import Config
from .model import PROMPT_VERSION, make_judge
from .pairs import build_and_save
from .store import JudgmentStore


def _quant(model_name: str) -> str:
    low = model_name.lower()
    for tag in ("nvfp4", "mxfp8", "bf16", "q4", "q8", "fp8", "mlx"):
        if tag in low:
            return tag
    return "unknown"


def _tasks(pairs: pd.DataFrame, swap_repeat: bool) -> list[dict[str, Any]]:
    """One task per (pair, presentation order)."""
    out = []
    for row in pairs.itertuples(index=False):
        out.append({"pair": row, "presentation": "as_shown"})
        if swap_repeat:
            out.append({"pair": row, "presentation": "swapped"})
    return out


def run(cfg: Config, mock: bool = False, limit: int | None = None) -> JudgmentStore:
    pairs = build_and_save(cfg)

    ages: dict[str, int] = {}
    for r in pairs.itertuples(index=False):
        ages[r.left_image] = int(r.left_age)
        ages[r.right_image] = int(r.right_age)

    judge = make_judge(cfg, mock=mock, ages=ages)
    ok, msg = judge.health()
    print(msg)
    if not ok:
        print("\naborting — fix the above, then re-run "
              "(already-recorded judgements are kept).")
        sys.exit(1)

    store = JudgmentStore(cfg)
    done = store.done_keys

    tasks = _tasks(pairs, cfg.swap_repeat)
    todo = [t for t in tasks
            if f"{t['pair'].pair_id}::{t['presentation']}" not in done]
    remaining = len(todo)
    if limit:
        todo = todo[:limit]

    msg = (f"{len(tasks):,} judgements planned · {len(tasks)-remaining:,} already done "
           f"· {remaining:,} remaining")
    if limit and remaining > len(todo):
        msg += f" · running {len(todo):,} this session (--limit)"
    print(msg)
    if not todo:
        store.flush(force=True)
        store.close()
        return store

    # Ctrl-C should end the run cleanly with everything synced, not lose the tail.
    stopping = {"flag": False}

    def _stop(signum, frame):                          # noqa: ARG001
        if stopping["flag"]:
            print("\nsecond interrupt — exiting now")
            sys.exit(130)
        stopping["flag"] = True
        print("\ninterrupt received — finishing current judgement and syncing…")

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        pass                                            # not available on some platforms

    quant = "provider-managed" if cfg.provider == "deepseek" else _quant(cfg.model_name)
    bar = tqdm(todo, unit="judgement", dynamic_ncols=True)
    n_since_flush = 0

    for task in bar:
        p = task["pair"]
        swapped = task["presentation"] == "swapped"

        # image 1 / image 2 as actually sent to the model
        first_side, second_side = ("right", "left") if swapped else ("left", "right")
        first_img = p.right_image if swapped else p.left_image
        second_img = p.left_image if swapped else p.right_image

        f1 = cfg.image_dir / first_img
        f2 = cfg.image_dir / second_img
        if not mock and (not f1.exists() or not f2.exists()):
            store.append(_record(cfg, p, task, None, None, None, None, quant,
                                 error=f"missing image: "
                                       f"{first_img if not f1.exists() else second_img}"))
            continue

        res = judge.judge(f1, f2)
        model_side = None
        if res["older"] == 1:
            model_side = first_side
        elif res["older"] == 2:
            model_side = second_side

        record = _record(cfg, p, task, model_side, res["confidence"],
                         res["raw"], res["latency_s"], quant, error=res["error"])
        record["backend"] = "mock" if mock else cfg.provider
        if "usage" in res:
            record["usage"] = res["usage"]
            record["served_model"] = res.get("served_model")
        store.append(record)

        n_since_flush += 1
        if n_since_flush >= cfg.save_every:
            store.flush()
            n_since_flush = 0

        df_acc = _running_accuracy(store)
        if df_acc is not None:
            bar.set_postfix_str(f"model acc {df_acc:.1%}")

        if stopping["flag"]:
            break

    bar.close()
    store.flush(force=True)
    store.close()
    return store


def _record(cfg: Config, p, task, model_side, confidence, raw, latency, quant,
            error: str | None = None) -> dict[str, Any]:
    true_side = p.true_older_side
    model_correct = (None if (model_side is None or true_side == "tie")
                     else float(model_side == true_side))
    human_correct = (None if pd.isna(p.human_correct) else float(p.human_correct))
    annotator = None if pd.isna(p.annotator) else p.annotator
    human_side = None if (p.human_side is None or pd.isna(p.human_side)) else p.human_side
    agrees = (None if (model_side is None or human_side is None)
              else float(model_side == human_side))
    model_choice = (None if model_side is None
                    else (p.left_image if model_side == "left" else p.right_image))

    return {
        "pair_id": p.pair_id,
        "presentation": task["presentation"],
        "left_image": p.left_image,
        "right_image": p.right_image,
        "left_age": int(p.left_age),
        "right_age": int(p.right_age),
        "age_gap": int(p.age_gap),
        "true_older_side": true_side,
        "model_side": model_side,
        "model_choice": model_choice,
        "model_correct": model_correct,
        "model_confidence": confidence,
        "human_side": human_side,
        "human_choice": None if pd.isna(p.human_choice) else p.human_choice,
        "human_correct": human_correct,
        "annotator": annotator,
        "model_agrees_with_human": agrees,
        "model_name": cfg.model_name,
        "quantization": quant,
        "prompt_version": PROMPT_VERSION,
        "latency_s": latency,
        "raw_response": raw,
        "error": error,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _running_accuracy(store: JudgmentStore) -> float | None:
    vals = [r["model_correct"] for r in store._records
            if r.get("model_correct") is not None]
    return sum(vals) / len(vals) if vals else None
