"""
Pre-compute HodgeRank for the AligNET dataset.

Each .npz file from https://storage.googleapis.com/alignet/data/release_1.1/
contains three arrays with n entries:
  filenames   (n, 3) str  — ImageNet filenames, e.g. 'n02165105_10535.JPEG'
  similarities(n, 3) float— [s01, s02, s12] dot-product similarities where
                             s01 = sim(img0, img1), s02 = sim(img0, img2),
                             s12 = sim(img1, img2)  — img2 is ALWAYS the odd-one-out
  indices     (n, 3) int  — ImageNet tfds indices (not used here)

Node representation: ImageNet synset classes (up to 1000 nodes)
  Synset ID extracted from filename: 'n02165105_10535.JPEG' -> 'n02165105'
  Mapped to 0-based integer class ID using the ImageNet 1000-class index.

Soft agreement: Bradley-Terry model
  P(c is odd | triplet a,b,c) ∝ exp(sim(a,b) / T)
  This gives a continuous agreement signal derived from the model's representation.

Usage:
    # Download one or more .npz files from GCS first:
    # https://storage.googleapis.com/alignet/data/release_1.1/
    #   uncertainty_distillation/val_within_class.npz   (10K rows, 467 KB)
    #   uncertainty_distillation/val_between_class.npz  (1M rows, 45 MB)
    #   uncertainty_distillation/train_between_class.npz (1M rows, 470 MB)
    #   etc.

    python3.12 precompute_alignnet.py

    # Or specify files explicitly:
    python3.12 precompute_alignnet.py data/alignet/val_between_class.npz

    # Combine multiple files:
    python3.12 precompute_alignnet.py data/alignet/val_*.npz
"""

import csv
import gzip
import json
import math
import sys
import urllib.request
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, save_npz
from scipy.sparse.linalg import lsqr
from scipy.special import softmax as scipy_softmax

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/alignet")
LABELS_PATH  = Path("data/imagenet_class_index.json")
CONCEPTS_OUT = Path("data/alignnet_concepts.json")
OUT_JSON     = Path("output/alignnet_graph.json")
OUT_D2_NPZ   = Path("output/alignnet_D2.npz")
OUT_D2_MTX   = Path("output/alignnet_D2.mtx")
OUT_EDGE_CSV = Path("output/alignnet_edge_index.csv")
OUT_S1_CSV   = Path("output/alignnet_s1_star.csv")

DEFAULT_FILES = sorted(DATA_DIR.glob("*.npz")) if DATA_DIR.exists() else []

IMAGENET_LABELS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)


# ── ImageNet class index ────────────────────────────────────────────────────

def load_imagenet_labels():
    """Load (or download) the ImageNet class index.

    Returns: dict synset_id -> (int_idx, label)
    The official JSON maps '0'..'999' -> [synset_id, class_name].
    """
    if not LABELS_PATH.exists():
        print(f"  Downloading ImageNet class index from {IMAGENET_LABELS_URL} ...", flush=True)
        Path("data").mkdir(exist_ok=True)
        try:
            urllib.request.urlretrieve(IMAGENET_LABELS_URL, LABELS_PATH)
            print(f"  Saved to {LABELS_PATH}", flush=True)
        except Exception as e:
            print(f"  WARNING: could not download labels ({e}); synset IDs will be used as labels.",
                  flush=True)
            return None

    with open(LABELS_PATH) as f:
        raw = json.load(f)  # {"0": ["n01440764", "tench"], ...}

    synset_to_class = {}
    for idx_str, (synset, label) in raw.items():
        synset_to_class[synset] = (int(idx_str), label)
    return synset_to_class


def build_class_mapping(synset_to_class, all_synsets):
    """Map synset IDs found in the data to integer class IDs (0-based).

    Uses the official ImageNet ordering when available; falls back to
    alphabetical ordering for any unknown synsets.
    """
    if synset_to_class:
        known = {s: synset_to_class[s][0] for s in all_synsets if s in synset_to_class}
        unknown = [s for s in all_synsets if s not in synset_to_class]
        if unknown:
            print(f"  WARNING: {len(unknown)} unknown synsets — assigning arbitrary IDs",
                  flush=True)
            max_known = max(known.values(), default=-1)
            for i, s in enumerate(sorted(unknown)):
                known[s] = max_known + 1 + i
        return known  # synset -> int_id
    else:
        return {s: i for i, s in enumerate(sorted(all_synsets))}


def write_concepts(synset_to_class, class_to_id):
    """Write data/alignnet_concepts.json: list of labels indexed by class ID."""
    n_classes = max(class_to_id.values()) + 1
    labels = ["?"] * n_classes
    for synset, class_id in class_to_id.items():
        if synset_to_class and synset in synset_to_class:
            label = synset_to_class[synset][1].replace("_", " ")
        else:
            label = synset
        labels[class_id] = label
    with open(CONCEPTS_OUT, "w") as f:
        json.dump(labels, f)
    print(f"  wrote {CONCEPTS_OUT} ({n_classes} classes)", flush=True)
    return labels


# ── triplet parsing ─────────────────────────────────────────────────────────

def synset_id(filename):
    """Extract ImageNet synset ID: 'n02165105_10535.JPEG' -> 'n02165105'."""
    return filename.split("_")[0]


def load_npz_files(paths):
    """Load and concatenate all .npz files. Returns (filenames, similarities)."""
    all_fns, all_sims = [], []
    for p in paths:
        print(f"  Loading {p} ...", flush=True)
        d = dict(np.load(p, allow_pickle=True))
        all_fns.append(d["filenames"])
        all_sims.append(d["similarities"].astype(np.float64))
        print(f"    {len(d['filenames']):,} rows", flush=True)
    fns  = np.concatenate(all_fns,  axis=0)
    sims = np.concatenate(all_sims, axis=0)
    print(f"  Total: {len(fns):,} rows", flush=True)
    return fns, sims


def aggregate_triplets(filenames, similarities, class_to_id):
    """Aggregate by canonical class-level triplet using soft Bradley-Terry probs.

    For each row (img0, img1, img2) with img2 as odd-one-out:
      pair_similarities = [sim(1,2), sim(0,2), sim(0,1)]
        — these are the pair-sims *excluding* each image if it were the odd one
      softmax over pair_similarities gives P(img_i is odd) for i in {0,1,2}

    Canonical triplet: sorted((class_id_0, class_id_1, class_id_2))
    Soft counts are mapped to canonical positions and accumulated.
    """
    counts = defaultdict(lambda: {"p_i": 0.0, "p_j": 0.0, "p_k": 0.0, "n": 0})
    skipped = 0
    rows = 0

    for row_idx in range(len(filenames)):
        fn0, fn1, fn2 = filenames[row_idx]
        s01, s02, s12 = similarities[row_idx]  # sim(0,1), sim(0,2), sim(1,2)

        c0 = class_to_id.get(synset_id(fn0))
        c1 = class_to_id.get(synset_id(fn1))
        c2 = class_to_id.get(synset_id(fn2))

        if c0 is None or c1 is None or c2 is None:
            skipped += 1
            continue
        if len({c0, c1, c2}) < 3:  # degenerate: two images from same class
            skipped += 1
            continue

        # Soft P(img_i is odd) via softmax over the "excluding-i" pair similarity
        #   if img0 odd: surviving pair is (1,2) → s12
        #   if img1 odd: surviving pair is (0,2) → s02
        #   if img2 odd: surviving pair is (0,1) → s01  ← largest, so img2 most likely odd
        pair_sims = np.array([s12, s02, s01], dtype=np.float64)
        probs = scipy_softmax(pair_sims)  # [p(img0 odd), p(img1 odd), p(img2 odd)]
        p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])

        # Canonical triplet (sorted integer class IDs)
        triplet = tuple(sorted([c0, c1, c2]))
        ci, cj, ck = triplet
        id_to_p = {c0: p0, c1: p1, c2: p2}

        counts[triplet]["p_i"] += id_to_p[ci]
        counts[triplet]["p_j"] += id_to_p[cj]
        counts[triplet]["p_k"] += id_to_p[ck]
        counts[triplet]["n"]   += 1
        rows += 1

        if rows % 500_000 == 0:
            print(f"    processed {rows:,} rows ...", flush=True)

    if skipped:
        print(f"  Skipped {skipped:,} rows (unknown synset or duplicate class in triplet)",
              flush=True)
    print(f"  {rows:,} valid rows -> {len(counts):,} canonical class-triplets", flush=True)
    return counts


def build_triangles(counts):
    """Convert aggregated counts to triangle objects with agreement scores."""
    ln3 = math.log(3)
    triangles = []
    for (i, j, k), v in counts.items():
        n = v["n"]
        p_i = v["p_i"] / n
        p_j = v["p_j"] / n
        p_k = v["p_k"] / n
        # Renormalize (softmax guarantees sum≈1, but numerical drift possible)
        s = p_i + p_j + p_k
        p_i, p_j, p_k = p_i/s, p_j/s, p_k/s
        H = -sum(p * math.log(p) for p in [p_i, p_j, p_k] if p > 0)
        ag = max(0.0, min(1.0, 1.0 - H / ln3))
        triangles.append({
            "i": i, "j": j, "k": k,
            "total_count": n,
            "p_i": round(p_i, 4), "p_j": round(p_j, 4), "p_k": round(p_k, 4),
            "agreement": round(ag, 4),
        })
    return triangles


def build_graph(triangles):
    """Build edges and nodes from triangles (same as precompute.py)."""
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
            edge_agg[key]["tot"]  += t["total_count"]
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
        mu  = float(ps.mean())
        var = float(ps.var())
        n_tri = len(ps)
        if n_tri > 1 and var > 0.02:
            cat = "context_dependent"
        elif mu > 0.45 and var < 0.02:
            cat = "consistently_odd"
        elif mu < 0.28 and var < 0.02:
            cat = "consistently_similar"
        else:
            cat = "mixed"
        nodes.append({"id": nid, "mean_oddness": round(mu, 4),
                      "var_oddness": round(var, 4), "n_triangles": n_tri,
                      "category": cat})
    return nodes, edges, edge_idx


def compute_s1_star(triangles, edges, edge_idx):
    """Hodge projection via LSQR (same as precompute.py)."""
    nE, nT = len(edges), len(triangles)
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
    """Find all tetrahedra (4-cliques with all 4 triangular faces)."""
    tri_set = set()
    edge_nodes = defaultdict(set)
    for t in triangles:
        a, b, c = t["i"], t["j"], t["k"]
        tri_set.add((a, b, c))
        edge_nodes[(a, b)].add(c); edge_nodes[(a, c)].add(b); edge_nodes[(b, c)].add(a)

    tet_set = set()
    for t in triangles:
        a, b, c = t["i"], t["j"], t["k"]
        for d in edge_nodes.get((a, b), set()):
            if d in (a, b, c): continue
            if d not in edge_nodes.get((a, c), set()): continue
            if d not in edge_nodes.get((b, c), set()): continue
            q = tuple(sorted([a, b, c, d]))
            if all(tuple(sorted(f)) in tri_set for f in combinations(q, 3)):
                tet_set.add(q)
    return [{"nodes": list(q)} for q in sorted(tet_set)]


def compute_b3_curl(triangles, tetras):
    """B3 curl component (same as precompute.py)."""
    nT, nTet = len(triangles), len(tetras)
    if nTet == 0:
        return np.zeros(0), 0.0

    tri_idx = {(t["i"], t["j"], t["k"]): c for c, t in enumerate(triangles)}
    s2 = np.array([t["agreement"] for t in triangles], dtype=np.float64)
    b3_rows, b3_cols, b3_vals = [], [], []
    s3 = np.zeros(nTet, dtype=np.float64)

    for col, tet in enumerate(tetras):
        a, b, c, d = sorted(tet["nodes"])
        for face, sign in [((b,c,d),+1),((a,c,d),-1),((a,b,d),+1),((a,b,c),-1)]:
            row = tri_idx.get(face)
            if row is None: continue
            b3_rows.append(row); b3_cols.append(col); b3_vals.append(sign)
            s3[col] += sign * s2[row]

    B3 = csc_matrix((b3_vals, (b3_rows, b3_cols)), shape=(nT, nTet))
    BtB = (B3.T @ B3).toarray()
    try:
        curl_coeff = np.linalg.solve(BtB, s3)
    except np.linalg.LinAlgError:
        curl_coeff, _, _, _ = np.linalg.lstsq(BtB, s3, rcond=None)

    row_contribs = defaultdict(float)
    for ri, ci, vi in zip(b3_rows, b3_cols, b3_vals):
        row_contribs[ri] += vi * curl_coeff[ci]
    norm_curl_sq = sum(v*v for v in row_contribs.values())
    norm_s2_sq = float(np.dot(s2, s2))
    curl_frac = norm_curl_sq / norm_s2_sq if norm_s2_sq > 0 else 0.0

    for col, tet in enumerate(tetras):
        tet["s3"] = round(float(s3[col]), 8)
    return s3, curl_frac


def export_d2(triangles, edges, edge_idx, s1):
    """Export D2 boundary operator + s1_star to files (like precompute.py)."""
    nE = len(edges)
    nT = len(triangles)
    d2_rows, d2_cols, d2_vals = [], [], []
    for c, t in enumerate(triangles):
        rij = edge_idx.get((t["i"], t["j"]))
        rik = edge_idx.get((t["i"], t["k"]))
        rjk = edge_idx.get((t["j"], t["k"]))
        if rij is None or rik is None or rjk is None:
            continue
        d2_rows.extend([rij, rik, rjk])
        d2_cols.extend([c, c, c])
        d2_vals.extend([1, -1, 1])

    D2 = csc_matrix((d2_vals, (d2_rows, d2_cols)), shape=(nE, nT))
    save_npz(OUT_D2_NPZ, D2)
    mmwrite(str(OUT_D2_MTX), D2, comment=(
        "D2 boundary operator (edges x triangles). "
        "Column t = triplet (i,j,k); rows are edges per alignnet_edge_index.csv. "
        "Triangle (i<j<k): edge(i,j)->+1, edge(i,k)->-1, edge(j,k)->+1."
    ), field="integer")

    with open(OUT_EDGE_CSV, "w") as f:
        f.write("edge_id,node_a,node_b,similarity\n")
        for e_idx, e in enumerate(edges):
            f.write(f"{e_idx},{e['source']},{e['target']},{e['similarity']:.6f}\n")

    with open(OUT_S1_CSV, "w") as f:
        f.write("edge_id,node_a,node_b,s1_star\n")
        for e_idx, e in enumerate(edges):
            f.write(f"{e_idx},{e['source']},{e['target']},{s1[e_idx]:.8f}\n")

    print(f"  wrote {OUT_D2_NPZ} and {OUT_D2_MTX} ({nE}×{nT}, {D2.nnz} nnz)", flush=True)
    print(f"  wrote {OUT_EDGE_CSV} and {OUT_S1_CSV}", flush=True)


def spring_layout_3d(nodes, edges):
    """3D spring layout using networkx."""
    import networkx as nx
    G = nx.Graph()
    for nd in nodes: G.add_node(nd["id"])
    for e in edges: G.add_edge(e["source"], e["target"])
    print(f"  layout: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges", flush=True)
    print("  Running spring_layout (3D) ...", flush=True)
    pos = nx.spring_layout(G, dim=3, iterations=80, seed=42, k=0.3)
    coords = np.array(list(pos.values()))
    coords = coords / np.abs(coords).max() * 300
    return {nid: coords[i].tolist() for i, nid in enumerate(pos)}


def main():
    # which files to process
    npz_files = [Path(p) for p in sys.argv[1:]] or DEFAULT_FILES
    if not npz_files:
        print("Usage: python3.12 precompute_alignnet.py [file.npz ...]")
        print(f"Expected files in {DATA_DIR}/")
        print("Download from: https://storage.googleapis.com/alignet/data/release_1.1/")
        sys.exit(1)

    missing = [p for p in npz_files if not p.exists()]
    if missing:
        print(f"ERROR: File(s) not found: {missing}")
        sys.exit(1)

    print("=== AligNET HodgeRank pre-computation ===", flush=True)
    print(f"Input files: {[str(p) for p in npz_files]}", flush=True)

    print("\n1. Loading ImageNet class labels ...", flush=True)
    synset_to_class = load_imagenet_labels()

    print("\n2. Loading .npz files ...", flush=True)
    filenames, similarities = load_npz_files(npz_files)

    print("\n3. Building synset→class mapping ...", flush=True)
    all_synsets = set()
    for row in filenames:
        for fn in row:
            all_synsets.add(synset_id(fn))
    print(f"  Found {len(all_synsets)} unique synsets in data", flush=True)
    class_to_id = build_class_mapping(synset_to_class, all_synsets)
    print(f"  Class ID range: 0–{max(class_to_id.values())}", flush=True)

    Path("data").mkdir(exist_ok=True)
    concept_labels = write_concepts(synset_to_class, class_to_id)

    print("\n4. Aggregating canonical class-triplets (soft agreement) ...", flush=True)
    counts = aggregate_triplets(filenames, similarities, class_to_id)

    print("\n5. Building triangles ...", flush=True)
    triangles = build_triangles(counts)
    print(f"  {len(triangles):,} triangles", flush=True)

    ag_vals = [t["agreement"] for t in triangles]
    print(f"  agreement: mean={np.mean(ag_vals):.4f}, "
          f"max={np.max(ag_vals):.4f}, min={np.min(ag_vals):.4f}", flush=True)

    print("\n6. Building graph ...", flush=True)
    nodes, edges, edge_idx = build_graph(triangles)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges", flush=True)

    print("\n7. Computing s1_star (Hodge projection) ...", flush=True)
    s1, frac, rel_res = compute_s1_star(triangles, edges, edge_idx)
    for i, e in enumerate(edges):
        e["s1_star"] = round(float(s1[i]), 8)

    print("\n8. Finding tetrahedra ...", flush=True)
    tetras = find_tetrahedra(triangles)
    print(f"  {len(tetras)} tetrahedra", flush=True)

    print("\n8b. Computing B3 curl component ...", flush=True)
    _, curl_frac = compute_b3_curl(triangles, tetras)
    harmonic_frac = max(0.0, 1.0 - frac - curl_frac)
    print(f"  gradient R² = {frac*100:.4f}%", flush=True)
    print(f"  curl R²     = {curl_frac*100:.6f}%", flush=True)
    print(f"  harmonic R² = {harmonic_frac*100:.6f}%", flush=True)

    print("\n9. Computing 3D layout ...", flush=True)
    positions = spring_layout_3d(nodes, edges)
    for nd in nodes:
        p = positions[nd["id"]]
        nd["x"] = round(p[0], 2)
        nd["y"] = round(p[1], 2)
        nd["z"] = round(p[2], 2)

    print("\n9b. Exporting D2 matrix + s1_star ...", flush=True)
    export_d2(triangles, edges, edge_idx, s1)

    print("\n10. Exporting JSON ...", flush=True)
    Path("output").mkdir(exist_ok=True)

    slim_tris = [[t["i"], t["j"], t["k"],
                  t["agreement"], t["total_count"],
                  t["p_i"], t["p_j"], t["p_k"]]
                 for t in triangles]
    slim_edges = [[e["source"], e["target"], e["similarity"], e.get("s1_star", 0)]
                  for e in edges]

    payload = {
        "nodes": nodes,
        "edges": slim_edges,
        "triangles": slim_tris,
        "tetrahedra": tetras,
        "meta": {
            "source": "AligNET",
            "dataset_type": "alignnet",
            "input_files": [str(p) for p in npz_files],
            "n_image_triplets": int(len(filenames)),
            "n_class_triplets": len(triangles),
            "frac_explained": round(frac, 6),
            "rel_residual": round(rel_res, 6),
            "curl_frac": round(curl_frac, 8),
            "harmonic_frac": round(harmonic_frac, 8),
            "precomputed": True,
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    gz_path = Path(str(OUT_JSON) + ".gz")
    with gzip.open(gz_path, "wt", compresslevel=6) as f:
        json.dump(payload, f, separators=(",", ":"))
    gz_mb = gz_path.stat().st_size / 1048576
    size_mb = OUT_JSON.stat().st_size / 1048576
    print(f"  wrote {OUT_JSON} ({size_mb:.1f} MB)", flush=True)
    print(f"  wrote {gz_path} ({gz_mb:.1f} MB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
