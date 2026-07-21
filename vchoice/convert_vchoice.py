"""
Converts variable-choice (slate -> purchased subset) session data into the
canonical odd-one-out triplet CSV format that hodgerank.py expects:

    i,j,k,odd_i_pct,odd_j_pct,odd_k_pct,agreement_score,total_count

Mapping rationale
------------------
hodgerank.py's Hodge-decomposition pipeline was built for perceptual
"odd-one-out" triplet judgments: for a triangle of 3 items, some fraction of
judges pick each vertex as the one that stands out.

Our data is revealed-preference data instead: a browsing session presents a
choice set (slate) of clicked items, of which a subset was purchased. To
generate an analogous "odd one out" signal for every triangle {a,b,c} that
co-occurred in a slate, we use purchase status as the judgment:

  - If exactly 1 of {a,b,c} was purchased -> that item is the odd one out
    (it stood out from the other two, which were both passed over).
  - If exactly 2 of {a,b,c} were purchased -> the *unpurchased* item is the
    odd one out (it stood out from the two that were both chosen).
  - If all 3 or none of {a,b,c} were purchased, the triangle carries no
    relative-preference signal, so it is skipped.

Aggregating this "odd" label across every session in which a given triangle
of items co-occurred yields odd_i_pct/odd_j_pct/odd_k_pct (summing to 100),
a total_count of decisive judgments, and an entropy-based agreement_score
(1 = every judgment agreed on the same odd item, 0 = judgments were split
uniformly 1/3-1/3-1/3, i.e. no consensus).

This lets hodgerank.py's D2 boundary operator / Hodge projection pipeline
run completely unmodified on top of e-commerce choice data, decomposing
purchase-preference signal into a globally-consistent ranking component
(gradient), locally-inconsistent triangle cycles (curl), and irreducible
global inconsistency (harmonic).
"""
import argparse
import math
from collections import Counter
from itertools import combinations
from pathlib import Path


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    slate_part, purchased_part = line.split(";")
    slate = [int(x) for x in slate_part.split()]
    purchased = set(int(x) for x in purchased_part.split())
    return slate, purchased


def convert(in_paths, out_path, max_slate_size=None, min_total_count=1):
    # odd_count[(i,j,k)][item] = number of decisive judgments naming `item` odd
    odd_count = {}
    total_count = Counter()

    n_lines = 0
    n_skipped_large_slate = 0
    n_triangles_seen = 0
    n_decisive = 0

    for path in in_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                n_lines += 1
                slate, purchased = parsed
                if len(slate) < 3:
                    continue
                if max_slate_size is not None and len(slate) > max_slate_size:
                    n_skipped_large_slate += 1
                    continue

                for a, b, c in combinations(sorted(set(slate)), 3):
                    n_triangles_seen += 1
                    p = (a in purchased) + (b in purchased) + (c in purchased)
                    if p == 1:
                        odd = a if a in purchased else (b if b in purchased else c)
                    elif p == 2:
                        odd = a if a not in purchased else (b if b not in purchased else c)
                    else:
                        continue  # p == 0 or p == 3: no signal

                    n_decisive += 1
                    key = (a, b, c)
                    total_count[key] += 1
                    d = odd_count.setdefault(key, {})
                    d[odd] = d.get(odd, 0) + 1

    # Write canonical triplet CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write("i,j,k,odd_i_pct,odd_j_pct,odd_k_pct,agreement_score,total_count\n")
        for (i, j, k), n in total_count.items():
            if n < min_total_count:
                continue
            counts = odd_count[(i, j, k)]
            ci, cj, ck = counts.get(i, 0), counts.get(j, 0), counts.get(k, 0)
            pi, pj, pk = ci / n, cj / n, ck / n

            # Entropy-based agreement score: 1 = unanimous, 0 = uniform (1/3 each)
            h = 0.0
            for p in (pi, pj, pk):
                if p > 0:
                    h -= p * math.log(p)
            h_max = math.log(3)
            agreement = 1.0 - (h / h_max)

            f.write(f"{i},{j},{k},{pi*100:.6f},{pj*100:.6f},{pk*100:.6f},{agreement:.6f},{n}\n")
            n_rows += 1

    print(f"Sessions read: {n_lines:,} (skipped {n_skipped_large_slate:,} for slate > {max_slate_size})")
    print(f"Triangles enumerated: {n_triangles_seen:,}; decisive (informative) judgments: {n_decisive:,}")
    print(f"Unique canonical triangles written: {n_rows:,} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="vchoice txt file(s)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--max-slate-size", type=int, default=None,
                     help="skip sessions with slate larger than this (combinatorial guard)")
    ap.add_argument("--min-total-count", type=int, default=1,
                     help="drop triangles observed fewer than this many times")
    args = ap.parse_args()
    convert([Path(p) for p in args.inputs], Path(args.out),
             max_slate_size=args.max_slate_size, min_total_count=args.min_total_count)
