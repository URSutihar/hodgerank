"""
Pre-compute the full HodgeRank pipeline for the THINGS dataset.

Reads the raw THINGS CSV, computes:
  - Canonical triplets with agreement scores
  - Edge similarity weights + node oddness stats
  - D2 boundary operator + LSQR Hodge projection (s1_star)
  - Tetrahedra (4-cliques)
  - 3D spring layout (node positions)

Exports output/full_graph.json for the browser visualization.

Run with:  python3.12 precompute.py
"""

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

DATA_CSV = Path("data/triplets_large_final_correctednc_correctedorder.csv")
OUT_JSON = Path("output/full_graph.json")


def parse_things_csv(path):
    """Parse raw THINGS CSV into canonical triplet counts."""
    counts = defaultdict(lambda: {"c": [0, 0, 0], "total": 0})
    rows = 0
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            a, b, c, ch = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tri = tuple(sorted([a, b, c]))
            odd = [a, b, c][ch - 1]
            idx = list(tri).index(odd)
            counts[tri]["c"][idx] += 1
            counts[tri]["total"] += 1
            rows += 1
            if rows % 1_000_000 == 0:
                print(f"  parsed {rows:,} rows ...", flush=True)
    print(f"  {rows:,} total rows, {len(counts):,} distinct triplets", flush=True)
    return counts


def build_triangles(counts):
    """Convert counts to triangle objects with agreement scores."""
    ln3 = math.log(3)
    triangles = []
    for (i, j, k), v in counts.items():
        total = v["total"]
        p = [x / total for x in v["c"]]
        H = -sum(pr * math.log(pr) for pr in p if pr > 0)
        ag = max(0.0, min(1.0, 1.0 - H / ln3))
        triangles.append({
            "i": i, "j": j, "k": k,
            "total_count": total,
            "p_i": round(p[0], 4), "p_j": round(p[1], 4), "p_k": round(p[2], 4),
            "agreement": round(ag, 4),
        })
    return triangles


def build_graph(triangles):
    """Build edges and nodes from triangles."""
    edge_agg = {}
    oddness = defaultdict(list)
    for t in triangles:
        pairs = [(t["i"], t["j"], t["p_k"]),
                 (t["i"], t["k"], t["p_j"]),
                 (t["j"], t["k"], t["p_i"])]
        for a, b, sim in pairs:
            key = (a, b)
            if key not in edge_agg:
                edge_agg[key] = {"wsum": 0.0, "tot": 0}
            edge_agg[key]["wsum"] += t["total_count"] * sim
            edge_agg[key]["tot"] += t["total_count"]
        oddness[t["i"]].append(t["p_i"])
        oddness[t["j"]].append(t["p_j"])
        oddness[t["k"]].append(t["p_k"])

    edges = []
    edge_idx = {}
    for (a, b), v in edge_agg.items():
        edge_idx[(a, b)] = len(edges)
        edges.append({"source": a, "target": b,
                       "similarity": round(v["wsum"] / v["tot"], 4)})

    nodes = []
    for nid in sorted(oddness):
        ps = np.array(oddness[nid])
        mu = float(ps.mean())
        var = float(ps.var())
        n_tri = len(ps)
        if n_tri > 1 and var > 0.1:
            cat = "context_dependent"
        elif mu > 0.5 and var < 0.05:
            cat = "consistently_odd"
        elif mu < 0.2 and var < 0.05:
            cat = "consistently_similar"
        else:
            cat = "mixed"
        nodes.append({"id": nid, "mean_oddness": round(mu, 4),
                       "var_oddness": round(var, 4), "n_triangles": n_tri,
                       "category": cat})

    return nodes, edges, edge_idx


def compute_s1_star(triangles, edges, edge_idx):
    """Hodge projection via LSQR on the full dataset."""
    nE = len(edges)
    nT = len(triangles)
    rows, cols, vals = [], [], []
    for c, t in enumerate(triangles):
        rij = edge_idx.get((t["i"], t["j"]))
        rik = edge_idx.get((t["i"], t["k"]))
        rjk = edge_idx.get((t["j"], t["k"]))
        if rij is None or rik is None or rjk is None:
            continue
        rows.extend([rij, rik, rjk])
        cols.extend([c, c, c])
        vals.extend([1, -1, 1])

    D2 = csc_matrix((vals, (rows, cols)), shape=(nE, nT))
    s2 = np.array([t["agreement"] for t in triangles], dtype=np.float64)

    print(f"  D2 shape = {D2.shape}, nnz = {D2.nnz}", flush=True)
    print("  Running LSQR ...", flush=True)
    result = lsqr(D2.T, s2)
    s1 = result[0]

    s2_hat = D2.T @ s1
    residual = s2 - s2_hat
    norm_s2 = np.linalg.norm(s2)
    frac = float(np.dot(s2_hat, s2_hat) / np.dot(s2, s2)) if norm_s2 > 0 else 0.0
    rel_res = float(np.linalg.norm(residual) / norm_s2) if norm_s2 > 0 else 0.0

    print(f"  fraction explained = {frac*100:.4f}%", flush=True)
    print(f"  ||residual||/||s2|| = {rel_res*100:.4f}%", flush=True)

    return s1, frac, rel_res


def find_tetrahedra(triangles):
    """Find all tetrahedra (4-cliques where all 4 triangular faces exist)."""
    tri_set = set()
    edge_nodes = defaultdict(set)
    for t in triangles:
        a, b, c = t["i"], t["j"], t["k"]
        tri_set.add((a, b, c))
        edge_nodes[(a, b)].add(c)
        edge_nodes[(a, c)].add(b)
        edge_nodes[(b, c)].add(a)

    tet_set = set()
    for t in triangles:
        a, b, c = t["i"], t["j"], t["k"]
        ab = edge_nodes.get((a, b), set())
        ac = edge_nodes.get((a, c), set())
        bc = edge_nodes.get((b, c), set())
        for d in ab:
            if d in (a, b, c):
                continue
            if d not in ac or d not in bc:
                continue
            q = tuple(sorted([a, b, c, d]))
            if all(tuple(sorted(f)) in tri_set for f in combinations(q, 3)):
                tet_set.add(q)

    tetras = [{"nodes": list(q)} for q in sorted(tet_set)]
    return tetras


def spring_layout_3d(nodes, edges):
    """3D spring layout using networkx (handles large edge counts efficiently)."""
    import networkx as nx
    G = nx.Graph()
    for nd in nodes:
        G.add_node(nd["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"])
    print(f"  layout: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges", flush=True)
    print("  Running spring_layout (3D) ...", flush=True)
    pos = nx.spring_layout(G, dim=3, iterations=50, seed=42, k=0.15)
    # scale to reasonable range for Three.js
    coords = np.array(list(pos.values()))
    coords = coords / np.abs(coords).max() * 300
    return {nid: coords[i].tolist() for i, nid in enumerate(pos)}


def main():
    print("=== HodgeRank full pre-computation ===", flush=True)

    print("1. Parsing THINGS CSV ...", flush=True)
    counts = parse_things_csv(DATA_CSV)

    print("2. Building triangles ...", flush=True)
    triangles = build_triangles(counts)
    print(f"  {len(triangles):,} triangles", flush=True)

    print("3. Building graph ...", flush=True)
    nodes, edges, edge_idx = build_graph(triangles)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges", flush=True)

    print("4. Computing s1_star (Hodge projection) ...", flush=True)
    s1, frac, rel_res = compute_s1_star(triangles, edges, edge_idx)
    for i, e in enumerate(edges):
        e["s1_star"] = round(float(s1[i]), 8)

    print("5. Finding tetrahedra ...", flush=True)
    tetras = find_tetrahedra(triangles)
    print(f"  {len(tetras)} tetrahedra", flush=True)

    print("6. Computing 3D layout ...", flush=True)
    positions = spring_layout_3d(nodes, edges)
    for nd in nodes:
        p = positions[nd["id"]]
        nd["x"] = round(p[0], 2)
        nd["y"] = round(p[1], 2)
        nd["z"] = round(p[2], 2)

    print("7. Exporting JSON ...", flush=True)
    Path("output").mkdir(exist_ok=True)
    # slim triangles: only keep fields the visualization needs
    slim_tris = []
    for t in triangles:
        slim_tris.append([t["i"], t["j"], t["k"],
                          t["agreement"], t["total_count"],
                          t["p_i"], t["p_j"], t["p_k"]])

    # slim edges: drop similarity to save space, keep s1_star
    slim_edges = []
    for e in edges:
        slim_edges.append([e["source"], e["target"],
                           e["similarity"], e.get("s1_star", 0)])

    payload = {
        "nodes": nodes,
        "edges": slim_edges,
        "triangles": slim_tris,
        "tetrahedra": tetras,
        "meta": {
            "source": "THINGS full dataset",
            "n_judgments": sum(t["total_count"] for t in triangles),
            "frac_explained": round(frac, 6),
            "rel_residual": round(rel_res, 6),
            "precomputed": True,
        }
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    # also write gzipped version for faster browser loading
    gz_path = Path(str(OUT_JSON) + ".gz")
    with gzip.open(gz_path, "wt", compresslevel=6) as f:
        json.dump(payload, f, separators=(",", ":"))
    gz_mb = gz_path.stat().st_size / 1048576
    print(f"  wrote {gz_path} ({gz_mb:.1f} MB)", flush=True)
    size_mb = OUT_JSON.stat().st_size / 1048576
    print(f"  wrote {OUT_JSON} ({size_mb:.1f} MB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
