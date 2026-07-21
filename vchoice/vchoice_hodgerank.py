"""
HodgeRank D2 pipeline for perceptual triplet data.

Reads pre-computed agreement scores from a canonical-triplet CSV
(i,j,k,odd_i_pct,odd_j_pct,odd_k_pct,agreement_score,total_count),
builds the D2 boundary operator (triangles -> edges) as a standard signed
incidence matrix, computes edge similarity weights and per-node
oddness-consistency statistics, and exports everything to data.json.

This is a faithful reproduction of URSutihar/hodgerank's hodgerank.py,
with CSV_PATH/output paths made CLI-configurable so it can be pointed at
a converted dataset instead of the bundled THINGS test set.

Run with: python3 hodgerank.py --csv output/vchoice_triplet_entropy.csv --outdir output
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz
from scipy.sparse.linalg import lsqr
from scipy.io import mmwrite

# ---------------------------------------------------------------------------
# Step 1: Read pre-computed agreement scores
# ---------------------------------------------------------------------------

def load_triplets(csv_path):
    """Load canonical triplets with oddness probabilities and agreement scores.

    Each row of the CSV is a canonical triplet (i < j < k) with the odd-one-out
    percentages for each vertex (in [0, 100]) and a precomputed agreement score.
    Returns a DataFrame with probability columns p_i, p_j, p_k added.
    """
    df = pd.read_csv(csv_path)

    df["p_i"] = df["odd_i_pct"] / 100.0
    df["p_j"] = df["odd_j_pct"] / 100.0
    df["p_k"] = df["odd_k_pct"] / 100.0

    prob_sums = df[["p_i", "p_j", "p_k"]].sum(axis=1)
    assert np.allclose(prob_sums, 1.0), "Oddness probabilities must sum to 1"
    assert (df["i"] < df["j"]).all() and (df["j"] < df["k"]).all(), \
        "Triplets must be canonical (i < j < k)"
    return df


# ---------------------------------------------------------------------------
# Step 2: Build the D2 boundary operator (edges x triangles)
# ---------------------------------------------------------------------------

def build_indices(df):
    node_ids = pd.unique(df[["i", "j", "k"]].values.ravel())
    node_ids = sorted(int(n) for n in node_ids)
    node_idx_of_id = {nid: idx for idx, nid in enumerate(node_ids)}

    edge_idx_of = {}
    edge_list = []
    for _, row in df.iterrows():
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        for a, b in ((i, j), (i, k), (j, k)):
            if (a, b) not in edge_idx_of:
                edge_idx_of[(a, b)] = len(edge_list)
                edge_list.append((a, b))
    return node_ids, node_idx_of_id, edge_list, edge_idx_of


def build_D2(df, edge_idx_of):
    n_edges = len(edge_idx_of)
    n_triangles = len(df)
    rows, cols, vals = [], [], []
    for t, (_, row) in enumerate(df.iterrows()):
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        rows.append(edge_idx_of[(i, j)]); cols.append(t); vals.append(+1)
        rows.append(edge_idx_of[(i, k)]); cols.append(t); vals.append(-1)
        rows.append(edge_idx_of[(j, k)]); cols.append(t); vals.append(+1)
    return csc_matrix((vals, (rows, cols)), shape=(n_edges, n_triangles))


def build_D1(edge_list, node_idx_of_id):
    n_nodes = len(node_idx_of_id)
    n_edges = len(edge_list)
    rows, cols, vals = [], [], []
    for e, (a, b) in enumerate(edge_list):
        rows.append(node_idx_of_id[a]); cols.append(e); vals.append(+1)
        rows.append(node_idx_of_id[b]); cols.append(e); vals.append(-1)
    return csc_matrix((vals, (rows, cols)), shape=(n_nodes, n_edges))


def validate_D2(D2, D1):
    nnz_per_col = np.diff(D2.indptr)
    assert (nnz_per_col == 3).all(), "Every triangle column must have exactly 3 edges"
    nnz_per_row = np.asarray((D2 != 0).sum(axis=1)).ravel()
    max_shared = int(nnz_per_row.max())
    assert set(np.unique(D2.data)).issubset({-1, 1}), "D2 entries must be +/-1"
    product = (D1 @ D2)
    max_abs = np.abs(product.toarray()).max() if product.nnz else 0
    assert max_abs == 0, f"D1 @ D2 must be the zero matrix (got max |entry| = {max_abs})"
    return {
        "nnz_per_col_all_3": True,
        "max_triangles_per_edge": max_shared,
        "n_shared_edges": int((nnz_per_row >= 2).sum()),
        "D1_D2_is_zero": True,
    }


# ---------------------------------------------------------------------------
# Step 3: Edge similarity weights
# ---------------------------------------------------------------------------

def compute_edge_weights(df, edge_list, edge_idx_of):
    weighted_sum = np.zeros(len(edge_list))
    total_count = np.zeros(len(edge_list))
    for _, row in df.iterrows():
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        n = row["total_count"]
        p_i, p_j, p_k = row["p_i"], row["p_j"], row["p_k"]
        for (a, b), sim in (((i, j), p_k), ((i, k), p_j), ((j, k), p_i)):
            e = edge_idx_of[(a, b)]
            weighted_sum[e] += n * sim
            total_count[e] += n
    return weighted_sum / total_count


# ---------------------------------------------------------------------------
# Step 4: Per-node oddness-consistency analysis
# ---------------------------------------------------------------------------

def compute_node_stats(df):
    oddness = {}
    for _, row in df.iterrows():
        for nid, p in ((int(row["i"]), row["p_i"]),
                       (int(row["j"]), row["p_j"]),
                       (int(row["k"]), row["p_k"])):
            oddness.setdefault(nid, []).append(p)

    stats = {}
    for nid, ps in oddness.items():
        ps = np.array(ps)
        mu = float(ps.mean())
        var = float(ps.var())
        n_tri = len(ps)
        if n_tri > 1 and var > 0.1:
            category = "context_dependent"
        elif mu > 0.5 and var < 0.05:
            category = "consistently_odd"
        elif mu < 0.2 and var < 0.05:
            category = "consistently_similar"
        else:
            category = "mixed"
        stats[nid] = {"mean_oddness": mu, "var_oddness": var, "n_triangles": n_tri, "category": category}
    return stats


# ---------------------------------------------------------------------------
# Step 5: Hodge projection s^1_* = pinv(D2 D2^T) D2 s^2
# ---------------------------------------------------------------------------

def hodge_project(D2, df):
    s2 = df["agreement_score"].values.astype(np.float64)
    result = lsqr(D2.T, s2)
    s1_star = result[0]
    s2_hat = D2.T @ s1_star
    residual = s2 - s2_hat
    norm_s2_sq = np.dot(s2, s2)
    norm_hat_sq = np.dot(s2_hat, s2_hat)
    frac_explained = norm_hat_sq / norm_s2_sq if norm_s2_sq > 0 else 0.0
    return s1_star, s2_hat, residual, frac_explained


def export_s1(s1_star, edge_list, path):
    with path.open("w", encoding="utf-8") as f:
        f.write("edge_id,node_a,node_b,s1_star\n")
        for e, (a, b) in enumerate(edge_list):
            f.write(f"{e},{a},{b},{s1_star[e]:.8f}\n")


def export_D2(D2, edge_list, similarity, df, d2_npz, d2_mtx, edge_index_csv):
    save_npz(d2_npz, D2)
    mmwrite(str(d2_mtx), D2, comment=(
        "D2 boundary operator (edges x triangles). "
        "Column t = triplet (i,j,k); rows are edges per edge_index.csv. "
        "Triangle (i<j<k): edge(i,j)->+1, edge(i,k)->-1, edge(j,k)->+1."
    ), field="integer")
    with edge_index_csv.open("w", encoding="utf-8") as f:
        f.write("edge_id,node_a,node_b,similarity\n")
        for e, (a, b) in enumerate(edge_list):
            f.write(f"{e},{a},{b},{similarity[e]:.6f}\n")


def export_json(df, node_ids, edge_list, similarity, node_stats, path):
    nodes = []
    for nid in node_ids:
        s = node_stats[nid]
        nodes.append({
            "id": nid,
            "mean_oddness": round(s["mean_oddness"], 4),
            "var_oddness": round(s["var_oddness"], 4),
            "n_triangles": s["n_triangles"],
            "category": s["category"],
        })
    edges = []
    for e, (a, b) in enumerate(edge_list):
        edges.append({"source": a, "target": b, "similarity": round(float(similarity[e]), 4)})
    triangles = []
    for _, row in df.iterrows():
        triangles.append({
            "i": int(row["i"]), "j": int(row["j"]), "k": int(row["k"]),
            "p_i": round(row["p_i"], 4), "p_j": round(row["p_j"], 4), "p_k": round(row["p_k"], 4),
            "agreement": round(float(row["agreement_score"]), 4),
            "total_count": int(row["total_count"]),
        })
    payload = {"nodes": nodes, "edges": edges, "triangles": triangles}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Step 6: B3 boundary operator and curl component
# ---------------------------------------------------------------------------

def _find_tetrahedra(triangles):
    from itertools import combinations
    tri_set = {(t["i"], t["j"], t["k"]) for t in triangles}
    edge_nodes = {}
    for t in triangles:
        for a, b, c in [(t["i"], t["j"], t["k"]), (t["i"], t["k"], t["j"]), (t["j"], t["k"], t["i"])]:
            edge_nodes.setdefault((a, b), set()).add(c)
    tet_set = set()
    for t in triangles:
        a, b, c = t["i"], t["j"], t["k"]
        for d in edge_nodes.get((a, b), set()):
            if d in (a, b, c):
                continue
            if d not in edge_nodes.get((a, c), set()) or d not in edge_nodes.get((b, c), set()):
                continue
            q = tuple(sorted([a, b, c, d]))
            if all(tuple(sorted(f)) in tri_set for f in combinations(q, 3)):
                tet_set.add(q)
    return [{"nodes": list(q)} for q in sorted(tet_set)]


def _compute_b3_curl(triangles, tetras):
    from collections import defaultdict
    nT = len(triangles)
    nTet = len(tetras)
    tri_idx = {(t["i"], t["j"], t["k"]): c for c, t in enumerate(triangles)}
    s2 = np.array([t["agreement"] for t in triangles], dtype=np.float64)

    b3_rows, b3_cols, b3_vals = [], [], []
    s3 = np.zeros(nTet)
    for col, tet in enumerate(tetras):
        a, b, c, d = sorted(tet["nodes"])
        for face, sign in [((b, c, d), +1), ((a, c, d), -1), ((a, b, d), +1), ((a, b, c), -1)]:
            row = tri_idx.get(face)
            if row is None:
                continue
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
    norm_curl_sq = sum(v * v for v in row_contribs.values())
    norm_s2_sq = float(np.dot(s2, s2))
    for col, tet in enumerate(tetras):
        tet["s3"] = round(float(s3[col]), 8)
    return s3, (norm_curl_sq / norm_s2_sq if norm_s2_sq > 0 else 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to canonical-triplet CSV")
    ap.add_argument("--outdir", default="output", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_out = outdir / "data.json"
    d2_npz = outdir / "D2.npz"
    d2_mtx = outdir / "D2.mtx"
    edge_index_csv = outdir / "edge_index.csv"
    s1_csv = outdir / "s1_star.csv"

    print("Loading triplets ...")
    df = load_triplets(args.csv)
    print(f"  {len(df):,} canonical triplets")

    print("Building indices ...")
    node_ids, node_idx_of_id, edge_list, edge_idx_of = build_indices(df)
    print(f"  {len(node_ids):,} nodes, {len(edge_list):,} edges")

    print("Building D2 (edges x triangles) ...")
    D2 = build_D2(df, edge_idx_of)
    print(f"  D2 shape = {D2.shape}, nnz = {D2.nnz}")

    print("Building D1 (nodes x edges) for validation ...")
    D1 = build_D1(edge_list, node_idx_of_id)

    print("Validating D2 ...")
    checks = validate_D2(D2, D1)
    for key, val in checks.items():
        print(f"  {key}: {val}")

    print("Computing edge similarity weights ...")
    similarity = compute_edge_weights(df, edge_list, edge_idx_of)
    print(f"  similarity range [{similarity.min():.3f}, {similarity.max():.3f}]")

    print("Computing per-node oddness statistics ...")
    node_stats = compute_node_stats(df)
    from collections import Counter
    cat_counts = Counter(s["category"] for s in node_stats.values())
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat}: {cnt}")

    print("Hodge projection: s^1_* = pinv(D2 D2^T) D2 s^2 (s^2 = agreement) ...")
    s1_star, s2_hat, residual, frac = hodge_project(D2, df)
    print(f"  s1_star shape = ({len(s1_star)},)")
    print(f"  s1_star range = [{s1_star.min():.6f}, {s1_star.max():.6f}]")
    print(f"  ||residual|| / ||s2|| = {np.linalg.norm(residual) / np.linalg.norm(df['agreement_score'].values):.6f}")
    print(f"  fraction explained by edges (R^2) = {frac:.6f} ({frac*100:.2f}%)")

    print("Exporting s1_star.csv ...")
    export_s1(s1_star, edge_list, s1_csv)

    print("Exporting D2 sparse matrix ...")
    export_D2(D2, edge_list, similarity, df, d2_npz, d2_mtx, edge_index_csv)

    print("Exporting data.json ...")
    payload = export_json(df, node_ids, edge_list, similarity, node_stats, json_out)
    print(f"  wrote {json_out} ({len(payload['nodes'])} nodes, {len(payload['edges'])} edges, {len(payload['triangles'])} triangles)")

    print("Computing B3 curl component ...")
    tris_for_b3 = [{"i": int(r.i), "j": int(r.j), "k": int(r.k), "agreement": float(r.agreement_score)}
                   for _, r in df.iterrows()]
    tetras_for_b3 = _find_tetrahedra(tris_for_b3)
    print(f"  {len(tetras_for_b3)} tetrahedra found")
    MAX_TETRAHEDRA_FOR_CURL = 6000  # B3^T B3 is dense O(n_tet^2); guard against blowup
    if len(tetras_for_b3) > MAX_TETRAHEDRA_FOR_CURL:
        print(f"  skipping curl computation: {len(tetras_for_b3)} tetrahedra exceeds "
              f"dense-matrix guard of {MAX_TETRAHEDRA_FOR_CURL} "
              f"(B3^T B3 would need ~{(len(tetras_for_b3)**2 * 8) / 1e9:.1f} GB)")
    elif tetras_for_b3:
        _, curl_frac = _compute_b3_curl(tris_for_b3, tetras_for_b3)
        harmonic_frac = max(0.0, 1.0 - frac - curl_frac)
        print(f"  gradient R^2 = {frac*100:.4f}%")
        print(f"  curl R^2 = {curl_frac*100:.6f}%")
        print(f"  harmonic R^2 = {harmonic_frac*100:.6f}%")
    else:
        print(f"  no tetrahedra -- curl R^2 = 0%, harmonic R^2 = {(1.0-frac)*100:.4f}%")

    print("Done.")


if __name__ == "__main__":
    main()
