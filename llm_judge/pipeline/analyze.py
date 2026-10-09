"""Analyse recorded judgements: model vs human, position bias, and a degree-1
Hodge decomposition on the same complex for both judges.

Mirrors the methodology settled on earlier in this repo — pairwise (degree-1)
rather than triangle-level, with a random-flow null so the numbers have a scale.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from .config import Config


# ── Hodge (degree 1) ────────────────────────────────────────────────────────
def _complex(pairs: list[tuple[str, str]]):
    items = sorted({n for p in pairs for n in p})
    ix = {n: i for i, n in enumerate(items)}
    rows, cols, vals = [], [], []
    for e, (u, v) in enumerate(pairs):
        rows.extend([ix[u], ix[v]])
        cols.extend([e, e])
        vals.extend([-1.0, 1.0])
    D1 = csr_matrix((vals, (rows, cols)), shape=(len(items), len(pairs)))

    nbr = defaultdict(set)
    for u, v in pairs:
        nbr[u].add(v)
        nbr[v].add(u)
    eidx = {p: i for i, p in enumerate(pairs)}
    tri_r, tri_c, tri_v = [], [], []
    ntri = 0
    for (u, v) in pairs:
        for w in nbr[u] & nbr[v]:
            if not (u < v < w):
                continue
            trio = [((u, v), 1.0), ((v, w), 1.0), ((u, w), -1.0)]
            if any(e not in eidx for e, _ in trio):
                continue
            for e, s in trio:
                tri_r.append(eidx[e]); tri_c.append(ntri); tri_v.append(s)
            ntri += 1
    D2 = csr_matrix((tri_v, (tri_r, tri_c)), shape=(len(pairs), max(ntri, 1)))
    return D1, D2, len(items), ntri


def _decompose(Y, D1, D2, ntri):
    tot = float(Y @ Y)
    if tot == 0:
        return 0.0, 0.0, 0.0
    s = lsqr(D1.T, Y, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    Yg = D1.T @ s
    g = float(Yg @ Yg) / tot
    if ntri == 0:
        return g, 0.0, max(0.0, 1.0 - g)
    phi = lsqr(D2, Y - Yg, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    Yc = D2 @ phi
    c = float(Yc @ Yc) / tot
    return g, c, max(0.0, 1.0 - g - c)


# ── main ────────────────────────────────────────────────────────────────────
def analyse(cfg: Config) -> dict:
    if not cfg.jsonl_path.exists():
        raise SystemExit(f"no judgements at {cfg.jsonl_path} — run the pipeline first")

    df = pd.read_json(cfg.jsonl_path, lines=True)
    df = df[df["model_side"].notna()].copy()
    if df.empty:
        raise SystemExit("no successful judgements to analyse")

    print(f"{len(df):,} judgements loaded "
          f"({df['pair_id'].nunique():,} distinct pairs)\n")

    out: dict = {"n_judgements": int(len(df)),
                 "n_pairs": int(df["pair_id"].nunique()),
                 "model": str(df["model_name"].iloc[-1]),
                 "quantization": str(df["quantization"].iloc[-1])}

    # ── accuracy ────────────────────────────────────────────────────────
    scored = df[df["model_correct"].notna()]
    hum = df[df["human_correct"].notna()].drop_duplicates("pair_id")
    out["model_accuracy"] = round(float(scored["model_correct"].mean()), 4)
    out["human_accuracy"] = round(float(hum["human_correct"].mean()), 4)
    out["model_human_agreement"] = round(float(df["model_agrees_with_human"].mean()), 4)
    print(f"model accuracy vs true age : {out['model_accuracy']:.4f}")
    print(f"human accuracy vs true age : {out['human_accuracy']:.4f}")
    print(f"model agrees with human    : {out['model_human_agreement']:.4f}\n")

    print("accuracy by true age gap:")
    by_gap = {}
    for lo, hi, lab in [(1,3,"1-3"),(4,7,"4-7"),(8,15,"8-15"),(16,30,"16-30"),(31,200,"31+")]:
        s = scored[(scored.age_gap >= lo) & (scored.age_gap <= hi)]
        h = hum[(hum.age_gap >= lo) & (hum.age_gap <= hi)]
        if len(s):
            by_gap[lab] = {"model": round(float(s["model_correct"].mean()), 4),
                           "human": round(float(h["human_correct"].mean()), 4) if len(h) else None,
                           "n": int(len(s))}
            hv = f"{by_gap[lab]['human']:.3f}" if by_gap[lab]["human"] is not None else "  —  "
            print(f"  gap {lab:>6} yr   model {by_gap[lab]['model']:.3f}   "
                  f"human {hv}   (n={len(s)})")
    out["by_gap"] = by_gap

    # ── position bias / self-consistency from swapped presentations ─────
    out["model_left_rate"] = round(float((df["model_side"] == "left").mean()), 4)
    print(f"\nmodel left-side rate: {out['model_left_rate']:.4f}  (0.5 = unbiased)")

    piv = df.pivot_table(index="pair_id", columns="presentation",
                         values="model_side", aggfunc="first")
    if {"as_shown", "swapped"}.issubset(piv.columns):
        both = piv.dropna()
        if len(both):
            same = float((both["as_shown"] == both["swapped"]).mean())
            out["order_consistency"] = round(same, 4)
            out["n_swap_pairs"] = int(len(both))
            print(f"order consistency   : {same:.4f} over {len(both):,} pairs "
                  f"(1.0 = choice never depends on presentation order)")
            if same < 0.75:
                print("  ! low — the model is substantially driven by position, "
                      "not by the faces")

    # ── Hodge, model vs human vs oracle vs null, same complex ───────────
    print("\nHodge decomposition (degree 1, identical complex for every row):")
    agg = df.groupby("pair_id").agg(
        left_image=("left_image", "first"), right_image=("right_image", "first"),
        left_age=("left_age", "first"), right_age=("right_age", "first"),
        human_side=("human_side", "first")).reset_index()

    # model flow = mean vote over presentations, in [-1, 1], positive = right older
    votes = (df.assign(v=np.where(df["model_side"] == "right", 1.0, -1.0))
               .groupby("pair_id")["v"].mean())
    agg["model_flow"] = agg["pair_id"].map(votes)

    keyed = {(r.left_image, r.right_image): r for r in agg.itertuples(index=False)}
    pairs, y_model, y_human, y_oracle = [], [], [], []
    for (l, r), row in keyed.items():
        u, v = (l, r) if l < r else (r, l)
        if (u, v) in {(p[0], p[1]) for p in pairs}:
            continue
        flip = 1.0 if (u, v) == (l, r) else -1.0
        pairs.append((u, v))
        y_model.append(flip * row.model_flow)
        if row.human_side in ("left", "right"):
            y_human.append(flip * (1.0 if row.human_side == "right" else -1.0))
        else:
            y_human.append(0.0)          # unjudged by humans = no information
        d = row.right_age - row.left_age
        y_oracle.append(flip * float(np.sign(d)))

    D1, D2, nV, ntri = _complex(pairs)
    print(f"  {nV:,} faces, {len(pairs):,} pairs, {ntri:,} triangles")
    n_hum = int(np.count_nonzero(y_human))
    if n_hum < len(pairs):
        print(f"  note: only {n_hum:,}/{len(pairs):,} pairs carry a human judgement, "
              f"so the HUMAN row is\n        computed on a sparse flow and is not "
              f"comparable to MODEL here.")
    if ntri == 0:
        print("  ! no triangles — pairs do not overlap, so curl/harmonic are "
              "undefined here.\n    Increase n_pairs or turn off stratification "
              "to get a connected complex.")

    rng = np.random.default_rng(0)
    y_null = rng.choice([-1.0, 1.0], size=len(pairs))

    # If even random flow carries no harmonic, the complex has no holes to hold
    # it (a complete or near-complete graph is contractible). Reporting 0.00%
    # then invites reading a structural constant as a finding, so we say n/a.
    _, _, h_null = _decompose(y_null, D1, D2, ntri)
    harmonic_defined = h_null > 0.005

    rows = {}
    for label, Y in [("MODEL", np.array(y_model)),
                     ("HUMAN", np.array(y_human)),
                     ("ORACLE (true age)", np.array(y_oracle)),
                     ("NULL (random)", y_null)]:
        g, c, h = _decompose(Y, D1, D2, ntri)
        rows[label] = {"gradient_pct": round(g*100, 2), "curl_pct": round(c*100, 2),
                       "harmonic_pct": round(h*100, 2) if harmonic_defined else None}
        hs = f"{h*100:6.2f}%" if harmonic_defined else "   n/a"
        print(f"  {label:20s} gradient {g*100:6.2f}%   curl {c*100:5.2f}%   harmonic {hs}")

    if not harmonic_defined:
        print("\n  harmonic is n/a: this complex is dense enough that every cycle "
              "bounds a\n  triangle, so H1 = 0 and harmonic is structurally zero for "
              "ANY judge --\n  it is a property of the graph, not of the data. Curl "
              "and gradient are\n  well estimated here. For a meaningful harmonic term "
              "use\n  pair_source: human_judged, or clique_keep_frac <= 0.2 (which "
              "costs triangles).")
    out["hodge"] = rows
    out["harmonic_defined"] = bool(harmonic_defined)
    out["n_triangles"] = int(ntri)

    dest = cfg.out_dir / "analysis.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {dest}")
    return out
