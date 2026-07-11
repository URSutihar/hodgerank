# HodgeRank — Simplicial Complex Explorer

Higher-order Hodge decomposition applied to perceptual triplet data from the [THINGS dataset](https://things-initiative.org/). Implements the HodgeRank framework to decompose agreement scores on a simplicial complex into gradient, curl, and harmonic components.

## What it does

- Parses odd-one-out triplet judgments into a simplicial complex (nodes, edges, triangles, tetrahedra)
- Computes the D2 boundary operator and projects agreement scores via LSQR (Hodge projection)
- Calculates Global Rankability (R²), Local Inconsistency (curl), and harmonic residual
- Interactive 3D visualization with Three.js — filter by agreement, oddness, similarity, s1* projection
- Tetrahedra (4-cliques) detection and isolated visualization
- Built-in AI assistant ("Ask the data") for querying the loaded dataset

## Quick start

### 1. Serve the visualization

Any static HTTP server works:

```bash
python3 -m http.server 8755
```

Open `http://localhost:8755/visualization.html` in your browser. The app loads a small bundled test dataset by default.

### 2. Load the full THINGS dataset

Download the raw triplet data from the THINGS project:

**Source:** [https://osf.io/f5rn6/](https://osf.io/f5rn6/overview)

Place the file at:
```
data/triplets_large_final_correctednc_correctedorder.csv
```

Then pre-compute the full graph (takes a few minutes):

```bash
pip install numpy scipy networkx
python3 precompute.py
```

This generates `output/full_graph.json` and `output/full_graph.json.gz`. Once generated, click **"Load full THINGS dataset"** in the visualization to load all 4.5M triplets, 1,854 nodes, and 98 tetrahedra.

### 3. Run the Python pipeline on test data

```bash
pip install numpy scipy pandas
python3 hodgerank.py
```

This processes the bundled test datasets and outputs the D2 boundary operator, edge indices, and Hodge projection results to `output/`.

## AI chat feature (optional)

The visualization includes an **"Ask the data"** button that lets you query the loaded dataset in natural language. It works in two modes:

- **Without an API key:** uses a built-in local pattern-matching engine (limited)
- **With an API key:** uses an LLM via OpenRouter for open-ended analysis

To enable the full AI chat:

1. Get an API key from [OpenRouter](https://openrouter.ai/) (create an account and generate a key)
2. In the visualization, click **"Ask the data"** then the **gear icon**
3. Paste your API key (starts with `sk-or-...`)
4. The key is stored only in your browser's localStorage — never sent anywhere except OpenRouter

## Key results (full THINGS dataset)

| Metric | Value |
|--------|-------|
| Triplets | 4,567,526 |
| Nodes | 1,854 |
| Edges | 1,717,731 (99.99% of possible) |
| Triangles | 4,567,526 |
| Tetrahedra | 98 |
| R² (fraction explained) | 99.97% |
| Global Rankability | 0.9998 |

## Visualization controls

- **Drag** = rotate, **scroll** = zoom, **right-drag** = pan
- **R** = recenter, **F** = fit to view
- **Double-click** a node = focus, a triangle/tetrahedron = isolate
- **Double-click** empty space = reset
- **Tetrahedra checkbox** = show only the 98 tetrahedra with their edges and faces

## Project structure

```
visualization.html    Main 3D visualization (Three.js, runs in browser)
hodgerank.py          Python pipeline for test datasets
precompute.py         Pre-compute full THINGS dataset for browser loading
data/                 Triplet data files
  testset1.txt        Bundled test dataset (small)
  testset2.txt        Additional test dataset
  testset3.txt        Additional test dataset
output/               Generated outputs (D2 matrices, projections, etc.)
```

## Theory

The Hodge decomposition on the simplicial complex gives:

**s² = s²_gradient + s²_curl + s²_harmonic**

where:
- **s²_gradient** = D₂ · s¹* (the projectable part — captures global ranking)
- **s²_curl** = B₃ · pinv(B₃ᵀB₃) · B₃ᵀ · s² (local inconsistency from tetrahedra)
- **s²_harmonic** = residual (global inconsistency not captured by either)

**Global Rankability** = √(R²) measures how well edge-level scores explain the triangle-level agreement.

## Requirements

- Python 3.12+ with `numpy`, `scipy`, `pandas`, `networkx`
- A modern browser (Chrome/Firefox/Safari) for the visualization
