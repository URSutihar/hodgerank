"""
Hofstadter Butterfly on an Erdős–Rényi Random Graph
====================================================
Following Daniel's 5-step recipe:
  1. G(n,p) random graph → largest connected component
  2. Eigenvalue spectrum of graph Laplacian L_0
  3. Spectral embedding into 2D via Fiedler eigenvectors (v2, v3)
  4. Magnetic Laplacian with Peierls phase from uniform B-field
  5. Eigenvalues vs B → random-graph Hofstadter butterfly
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
import networkx as nx
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Erdős–Rényi graph, largest connected component ──────────────

def generate_graph(n=100, p=0.02, seed=42):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
    print(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(G)}")
    return G


# ── Step 2: Graph Laplacian spectrum ────────────────────────────────────

def laplacian_spectrum(G):
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(eigvalsh(L))
    return L, eigenvalues


def plot_spectrum(eigenvalues, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.stem(range(len(eigenvalues)), eigenvalues, linefmt='#4a9eff',
             markerfmt='o', basefmt='k-')
    ax1.set_xlabel('Index k')
    ax1.set_ylabel(r'$\lambda_k$')
    ax1.set_title(f'Laplacian Eigenvalue Spectrum{title_suffix}')

    gaps = np.diff(eigenvalues)
    avg_gap = np.mean(gaps)
    ax2.bar(range(len(gaps)), gaps, color='#f59e0b', alpha=0.8)
    ax2.axhline(avg_gap, color='red', ls='--', label=f'mean gap = {avg_gap:.3f}')
    ax2.set_xlabel('Gap index')
    ax2.set_ylabel(r'$\lambda_{k+1} - \lambda_k$')
    ax2.set_title('Eigenvalue Gaps')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'laplacian_spectrum.png'), dpi=150)
    plt.show()

    large_gaps = np.where(gaps > 2 * avg_gap)[0]
    if len(large_gaps) > 0:
        print(f"Large gaps (> 2× mean) at indices: {large_gaps}")
        for i in large_gaps:
            print(f"  gap {i}: λ_{i}={eigenvalues[i]:.4f} → λ_{i+1}={eigenvalues[i+1]:.4f}, "
                  f"Δ={gaps[i]:.4f}")
    else:
        print("No gaps larger than 2× the mean spacing.")


# ── Step 3: Spectral embedding into 2D ─────────────────────────────────

def spectral_embedding(G, L):
    n = G.number_of_nodes()
    L_sparse = sparse.csr_matrix(L)
    # k=3 gives eigenvalues λ_0, λ_1, λ_2 and eigenvectors v_1, v_2, v_3
    eigenvalues, eigenvectors = eigsh(L_sparse, k=3, which='SM')
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # v_2 and v_3 (0-indexed: columns 1 and 2)
    v2 = eigenvectors[:, 1]
    v3 = eigenvectors[:, 2]

    print(f"Spectral embedding eigenvalues: λ_0={eigenvalues[0]:.6f}, "
          f"λ_1={eigenvalues[1]:.6f}, λ_2={eigenvalues[2]:.6f}")

    # Rescale embedding so coordinates span [-1, 1]
    # This normalizes the "lattice constant" so B has physical meaning:
    # a flux quantum threads a unit-area plaquette at B = 2π
    scale2 = np.max(np.abs(v2))
    scale3 = np.max(np.abs(v3))
    if scale2 > 0: v2 = v2 / scale2
    if scale3 > 0: v3 = v3 / scale3
    print(f"Embedding rescaled to [-1, 1] range")

    return v2, v3


def plot_embedding(G, v2, v3):
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, w in G.edges():
        ax.plot([v2[u], v2[w]], [v3[u], v3[w]], '-', color='#4a9eff',
                alpha=0.3, linewidth=0.6)
    ax.scatter(v2, v3, c='#f59e0b', s=30, zorder=5, edgecolors='#1a1205',
               linewidth=0.5)
    ax.set_xlabel(r'$v_2(i)$ — Fiedler vector')
    ax.set_ylabel(r'$v_3(i)$')
    ax.set_title('Spectral Embedding of Largest Connected Component')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'spectral_embedding.png'), dpi=150)
    plt.show()


# ── Step 4 & 5: Magnetic Laplacian and Hofstadter butterfly ────────────

def magnetic_laplacian(G, v2, v3, B):
    """
    Construct the magnetic Laplacian with Peierls substitution.

    For the Landau gauge A = (0, Bx, 0), the Peierls phase on edge (n,m) is:
        φ_{nm} = B · (v3[m] - v3[n]) · (v2[m] + v2[n]) / 2

    This comes from the line integral of A along the edge:
        φ_{nm} = ∫_n^m A · dl

    The magnetic Laplacian has:
        L_{nm} = -e^{iφ_{nm}}  for connected n,m
        L_{nn} = degree(n)     (diagonal unchanged)
        L_{mn} = L_{nm}*       (Hermitian)
    """
    n = G.number_of_nodes()
    L_B = np.zeros((n, n), dtype=complex)

    for u, w in G.edges():
        # Peierls phase: φ = B * (x_m - x_n) * (y_m + y_n) / 2
        # where x = v2, y = v3 (from spectral embedding)
        # Using Landau gauge A = (0, B*x, 0):
        # φ_{nm} = B * (v3[m] - v3[n]) * (v2[m] + v2[n]) / 2
        phase = B * (v3[w] - v3[u]) * (v2[w] + v2[u]) / 2.0
        L_B[u, w] = -np.exp(1j * phase)
        L_B[w, u] = -np.exp(-1j * phase)

    for i in range(n):
        L_B[i, i] = G.degree(i)

    return L_B


def compute_butterfly(G, v2, v3, B_values):
    """Compute eigenvalues of magnetic Laplacian for each B value."""
    all_eigenvalues = []
    for i, B in enumerate(B_values):
        L_B = magnetic_laplacian(G, v2, v3, B)
        eigs = np.sort(np.real(eigvalsh(L_B)))
        all_eigenvalues.append(eigs)
        if (i + 1) % 50 == 0:
            print(f"  computed {i+1}/{len(B_values)} B values...")
    return np.array(all_eigenvalues)


def plot_butterfly(B_values, all_eigenvalues, p_val, title_extra=""):
    fig, ax = plt.subplots(figsize=(14, 9))
    n_eigs = all_eigenvalues.shape[1]
    # Plot each eigenvalue branch as a line
    for k in range(n_eigs):
        ax.plot(B_values, all_eigenvalues[:, k], color='#4a9eff',
                linewidth=0.15, alpha=0.7, rasterized=True)
    ax.set_xlabel('Magnetic field strength B', fontsize=13)
    ax.set_ylabel('Eigenvalue λ', fontsize=13)
    ax.set_title(f'Hofstadter Butterfly on Erdős–Rényi G(n, p={p_val}){title_extra}',
                 fontsize=15)
    ax.set_facecolor('#0a0c10')
    fig.patch.set_facecolor('#13161d')
    ax.tick_params(colors='#e2e6ee')
    ax.xaxis.label.set_color('#e2e6ee')
    ax.yaxis.label.set_color('#e2e6ee')
    ax.title.set_color('#f59e0b')
    for spine in ax.spines.values():
        spine.set_color('#2a2f3a')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'hofstadter_butterfly_p{p_val}.png'),
                dpi=300, facecolor=fig.get_facecolor())
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────

def run(n=100, p=0.02, n_B=3000, B_max=50.0, seed=42):
    print("=" * 60)
    print(f"Hofstadter Butterfly on G({n}, {p})")
    print("=" * 60)

    # Step 1
    print("\n── Step 1: Generate Erdős–Rényi graph ──")
    G = generate_graph(n, p, seed)

    # Step 2
    print("\n── Step 2: Laplacian eigenvalue spectrum ──")
    L, eigenvalues = laplacian_spectrum(G)
    plot_spectrum(eigenvalues)

    # Step 3
    print("\n── Step 3: Spectral embedding ──")
    v2, v3 = spectral_embedding(G, L)
    plot_embedding(G, v2, v3)

    # Steps 4–5
    print(f"\n── Steps 4–5: Magnetic Laplacian, B ∈ [0, {B_max}] ──")
    B_values = np.linspace(0, B_max, n_B)
    all_eigs = compute_butterfly(G, v2, v3, B_values)
    plot_butterfly(B_values, all_eigs, p)

    # Bonus: try higher p values
    for p2 in [0.05, 0.10, 0.30]:
        print(f"\n── Comparison: p = {p2} ──")
        G2 = generate_graph(n, p2, seed)
        L2, eigs2 = laplacian_spectrum(G2)
        v2_2, v3_2 = spectral_embedding(G2, L2)
        all_eigs2 = compute_butterfly(G2, v2_2, v3_2, B_values)
        plot_butterfly(B_values, all_eigs2, p2)

    print("\n✓ All plots saved to output/")


if __name__ == '__main__':
    run()
