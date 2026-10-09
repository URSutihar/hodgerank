"""Build the pair list the model will judge.

Pairs come from the real human comparisons in crowd_labels.csv, so every model
judgement has a matching human judgement and a ground-truth age gap. That is
what makes the model/human comparison possible at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

GAP_BINS = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 200)]


def _gap_label(g: int) -> str:
    """Bucket a true-age gap. 0 means the two photos share an age (a tie)."""
    if g == 0:
        return "0 (tie)"
    for lo, hi in GAP_BINS:
        if lo <= g <= hi:
            return f"{lo}-{hi}"
    return f"{GAP_BINS[-1][0]}+"


def _basename(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split("/").str[-1]


def build_pairs(cfg: Config) -> pd.DataFrame:
    """One row per human-judged pair, with ground truth and the human's choice."""
    gt = pd.read_csv(cfg.gt_csv)
    gt["item"] = _basename(gt["label"])
    age = dict(zip(gt["item"], gt["score"].astype(int)))

    crowd = pd.read_csv(cfg.crowd_csv)
    crowd["l"] = _basename(crowd["left"])
    crowd["r"] = _basename(crowd["right"])
    crowd["w"] = _basename(crowd["label"])

    df = pd.DataFrame({
        "left_image": crowd["l"],
        "right_image": crowd["r"],
        "human_choice": crowd["w"],
        "annotator": crowd["performer"].astype(str),
    })
    df["left_age"] = df["left_image"].map(age)
    df["right_age"] = df["right_image"].map(age)
    df = df.dropna(subset=["left_age", "right_age"]).copy()
    df["left_age"] = df["left_age"].astype(int)
    df["right_age"] = df["right_age"].astype(int)
    df["age_gap"] = (df["left_age"] - df["right_age"]).abs()

    df["true_older_side"] = np.where(
        df["left_age"] == df["right_age"], "tie",
        np.where(df["left_age"] > df["right_age"], "left", "right"))
    df["human_side"] = np.where(df["human_choice"] == df["left_image"], "left", "right")
    df["human_correct"] = np.where(
        df["true_older_side"] == "tie", np.nan,
        (df["human_side"] == df["true_older_side"]).astype(float))

    # stable identity so a run can resume regardless of ordering
    df["pair_id"] = (df["left_image"] + "|" + df["right_image"])
    return df.reset_index(drop=True)


def select(cfg: Config, pairs: pd.DataFrame) -> pd.DataFrame:
    """Subsample down to cfg.n_pairs, optionally spreading across age gaps."""
    if cfg.n_pairs is None or cfg.n_pairs >= len(pairs):
        return pairs.sample(frac=1.0, random_state=cfg.run_seed).reset_index(drop=True)

    rng = np.random.default_rng(cfg.run_seed)
    if not cfg.stratify_by_gap:
        idx = rng.choice(len(pairs), size=cfg.n_pairs, replace=False)
        return pairs.iloc[idx].reset_index(drop=True)

    per_bin = cfg.n_pairs // len(GAP_BINS)
    picked = []
    for lo, hi in GAP_BINS:
        sub = pairs[(pairs["age_gap"] >= lo) & (pairs["age_gap"] <= hi)]
        take = min(per_bin, len(sub))
        if take:
            picked.append(sub.sample(n=take, random_state=cfg.run_seed))
    out = pd.concat(picked) if picked else pairs.head(0)

    # top up to the requested count from whatever is left
    if len(out) < cfg.n_pairs:
        rest = pairs.drop(out.index)
        extra = min(cfg.n_pairs - len(out), len(rest))
        if extra:
            out = pd.concat([out, rest.sample(n=extra, random_state=cfg.run_seed)])

    return out.sample(frac=1.0, random_state=cfg.run_seed).reset_index(drop=True)


def select_connected(cfg: Config, pairs: pd.DataFrame) -> pd.DataFrame:
    """Grow a dense subgraph so the pair set actually contains triangles.

    Random sampling cannot work here: drawing n pairs at random from 250,249
    over 9,150 faces yields ~(n/N)^3 * 27,268 triangles, which is under 1 for
    any n small enough to run locally. Without shared faces there are no
    triangles, so curl and harmonic are undefined and every judge scores 100%
    gradient — the exact degeneracy this project already hit once.

    Instead: snowball out from a high-degree seed face, keeping every
    human-judged pair among the accepted faces, until we have enough pairs.
    """
    rng = np.random.default_rng(cfg.run_seed)

    adj: dict[str, set[str]] = {}
    for l, r in zip(pairs["left_image"], pairs["right_image"]):
        adj.setdefault(l, set()).add(r)
        adj.setdefault(r, set()).add(l)

    # start from one of the best-connected faces to maximise early density
    seed = max(adj, key=lambda n: len(adj[n]))
    chosen: set[str] = {seed}
    frontier = list(adj[seed])
    rng.shuffle(frontier)

    def n_edges() -> int:
        return sum(1 for l, r in zip(pairs["left_image"], pairs["right_image"])
                   if l in chosen and r in chosen)

    # grow in blocks, re-counting edges only occasionally (the count is O(N))
    while frontier and len(chosen) < len(adj):
        block = [frontier.pop() for _ in range(min(40, len(frontier)))]
        for node in block:
            if node in chosen:
                continue
            chosen.add(node)
            nxt = [x for x in adj.get(node, ()) if x not in chosen]
            rng.shuffle(nxt)
            frontier.extend(nxt[:20])
        if n_edges() >= cfg.n_pairs:
            break

    mask = (pairs["left_image"].isin(chosen) & pairs["right_image"].isin(chosen))
    sub = pairs[mask]
    if len(sub) > cfg.n_pairs:
        # trim by dropping the least-connected faces first, preserving density
        deg = pd.concat([sub["left_image"], sub["right_image"]]).value_counts()
        keep = set(deg.index)
        while len(sub) > cfg.n_pairs and len(keep) > 3:
            drop = deg[deg.index.isin(keep)].idxmin()
            keep.discard(drop)
            sub = pairs[pairs["left_image"].isin(keep) & pairs["right_image"].isin(keep)]
            deg = pd.concat([sub["left_image"], sub["right_image"]]).value_counts()

    return sub.sample(frac=1.0, random_state=cfg.run_seed).reset_index(drop=True)


def count_triangles(sub: pd.DataFrame) -> int:
    """Triangles in the comparison graph — the thing Hodge needs."""
    from collections import defaultdict
    nbr = defaultdict(set)
    for l, r in zip(sub["left_image"], sub["right_image"]):
        nbr[l].add(r)
        nbr[r].add(l)
    edges = {tuple(sorted((l, r)))
             for l, r in zip(sub["left_image"], sub["right_image"])}
    n = 0
    for u, v in edges:
        n += len(nbr[u] & nbr[v])
    return n // 3


def build_clique_pairs(cfg: Config, k: int, keep_frac: float = 1.0) -> pd.DataFrame:
    """Every pair among K faces — a complete graph, so every triple is a triangle.

    The human comparison graph is triangle-poor (27,268 triangles over 250,249
    pairs), which leaves curl and harmonic weakly estimated no matter how the
    subgraph is drawn. A clique fixes that by construction: K faces give
    C(K,2) pairs and C(K,3) triangles, so K=70 buys 54,740 triangles for the
    same 2,415 judgements a sparse sample spends on ~71.

    The cost is coverage of the human data: only the ~7% of clique pairs that
    humans happened to judge carry a human column. Use `human_judged` when the
    question is model-vs-human; use `clique` when the question is whether the
    model's own judgements are internally transitive.
    """
    from itertools import combinations

    gt = pd.read_csv(cfg.gt_csv)
    gt["item"] = _basename(gt["label"])
    age = dict(zip(gt["item"], gt["score"].astype(int)))

    crowd = pd.read_csv(cfg.crowd_csv)
    cl, cr, cw = _basename(crowd["left"]), _basename(crowd["right"]), _basename(crowd["label"])
    judged: dict[tuple[str, str], tuple[str, str]] = {}
    adj: dict[str, set[str]] = {}
    for l, r, w, perf in zip(cl, cr, cw, crowd["performer"].astype(str)):
        judged[tuple(sorted((l, r)))] = (w, perf)
        adj.setdefault(l, set()).add(r)
        adj.setdefault(r, set()).add(l)

    # Greedily grow the face set, always taking whichever face is best connected
    # into what we already have. Maximises how many clique pairs humans covered.
    deg = pd.concat([cl, cr]).value_counts()
    chosen = [deg.index[0]]
    cand = set(adj[chosen[0]])
    while len(chosen) < k and cand:
        cur = set(chosen)
        nxt = max(cand, key=lambda n: len(adj.get(n, ()) & cur))
        chosen.append(nxt)
        cand |= adj.get(nxt, set())
        cand -= set(chosen)
    # top up from the highest-degree faces if the neighbourhood ran dry
    for face in deg.index:
        if len(chosen) >= k:
            break
        if face not in chosen and face in age:
            chosen.append(face)
    chosen = [c for c in chosen if c in age][:k]

    rng = np.random.default_rng(cfg.run_seed)
    all_pairs = list(combinations(sorted(chosen), 2))
    if keep_frac < 1.0:
        keep_n = max(3, int(len(all_pairs) * keep_frac))
        idx = rng.choice(len(all_pairs), size=keep_n, replace=False)
        all_pairs = [all_pairs[i] for i in sorted(idx)]

    rows = []
    for a, b in all_pairs:
        # randomise which face is shown left so the set has no built-in side bias
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        hit = judged.get(tuple(sorted((a, b))))
        rows.append({
            "left_image": left, "right_image": right,
            "left_age": age[left], "right_age": age[right],
            "human_choice": hit[0] if hit else None,
            "annotator": hit[1] if hit else None,
        })

    df = pd.DataFrame(rows)
    df["age_gap"] = (df["left_age"] - df["right_age"]).abs()
    df["true_older_side"] = np.where(
        df["left_age"] == df["right_age"], "tie",
        np.where(df["left_age"] > df["right_age"], "left", "right"))
    df["human_side"] = np.where(
        df["human_choice"].isna(), None,
        np.where(df["human_choice"] == df["left_image"], "left", "right"))
    df["human_correct"] = np.where(
        df["human_side"].isna() | (df["true_older_side"] == "tie"), np.nan,
        (df["human_side"] == df["true_older_side"]).astype(float))
    df["pair_id"] = df["left_image"] + "|" + df["right_image"]

    n_hum = int(df["human_side"].notna().sum())
    tri = count_triangles(df)
    print(f"clique of {len(chosen)} faces, keeping {100*keep_frac:.0f}% of edges "
          f"-> {len(df):,} pairs, {tri:,} triangles")
    print(f"  {n_hum:,} pairs ({100*n_hum/len(df):.1f}%) were also judged by humans")
    if keep_frac >= 1.0:
        print("  ! complete graph: harmonic will be identically 0 for every judge "
              "(no holes\n    to carry it). Set clique_keep_frac below 1.0 to "
              "measure harmonic.")
    return df.sample(frac=1.0, random_state=cfg.run_seed).reset_index(drop=True)


def build_and_save(cfg: Config) -> pd.DataFrame:
    cfg.ensure_dirs()
    if cfg.pairs_path.exists():
        sel = pd.read_parquet(cfg.pairs_path)
        print(f"reusing existing pair list: {len(sel):,} pairs "
              f"({cfg.pairs_path.name}) — delete it to resample")
        return sel

    run_cfg = cfg.raw.get("run", {}) or {}
    source = run_cfg.get("pair_source", "human_judged")

    if source == "clique":
        sel = build_clique_pairs(cfg, int(run_cfg.get("clique_faces", 70)),
                                  float(run_cfg.get("clique_keep_frac", 0.75)))
        sel.to_parquet(cfg.pairs_path, index=False)
        faces = pd.concat([sel["left_image"], sel["right_image"]]).nunique()
        print(f"  {faces:,} distinct faces · complete graph")
        return sel

    allp = build_pairs(cfg)
    mode = run_cfg.get("selection", "connected")
    if mode == "connected" and cfg.n_pairs is not None:
        sel = select_connected(cfg, allp)
    else:
        sel = select(cfg, allp)
    sel.to_parquet(cfg.pairs_path, index=False)
    print(f"{len(allp):,} human-judged pairs available; selected {len(sel):,}")
    dist = sel["age_gap"].apply(_gap_label)
    for label, n in dist.value_counts().sort_index().items():
        print(f"  gap {label:>7} yr: {n:>5,}")

    faces = pd.concat([sel["left_image"], sel["right_image"]]).nunique()
    tri = count_triangles(sel)
    print(f"  {faces:,} distinct faces · {tri:,} triangles in the comparison graph")
    if tri == 0:
        print("  ! no triangles — Hodge curl/harmonic will be undefined. "
              "Raise n_pairs or use selection: connected")
    return sel
