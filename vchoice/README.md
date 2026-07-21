# vchoice: HodgeRank applied to e-commerce subset-choice data

This extends the original THINGS-triplet HodgeRank pipeline to a second
dataset: e-commerce variable-choice ("vchoice") data, where each session
records a browsing choice set (clicked items) and the subset of it that was
purchased.

Source data: derived from the 2015 RecSys Challenge
(http://2015.recsyschallenge.com/). See `data/DATA_README.txt` for the exact
line format.

## Why this needed a bridge, not a drop-in

`hodgerank.py`'s pipeline expects canonical **odd-one-out triplets**: for a
triangle of 3 items, what fraction of judges picked each one as the outlier
(`odd_i_pct`, `odd_j_pct`, `odd_k_pct`, `agreement_score`, `total_count`).
vchoice data has no such judgment — instead it has purchase behavior over a
shown choice set.

`convert_vchoice.py` bridges the two: for every triangle `{a,b,c}` of items
that co-occurred in a choice set, purchase status supplies an odd-one-out
label —

- exactly 1 of 3 purchased → that item is the odd one out (it stood out by
  being chosen while its co-shown neighbors weren't)
- exactly 2 of 3 purchased → the unpurchased item is the odd one out
- 0 or 3 of 3 purchased → no relative-preference signal; triangle is dropped

Aggregated across sessions, this produces the exact CSV schema
`hodgerank.py` expects, so the existing D2 boundary-operator / Hodge
projection code runs on it unmodified in spirit (see `vchoice_hodgerank.py`,
a copy of `hodgerank.py` with I/O made CLI-configurable and one added guard,
described below).

## Files

- `convert_vchoice.py` — converts a vchoice txt file into a canonical
  triplet CSV.
- `vchoice_hodgerank.py` — the repo's `hodgerank.py` pipeline, unchanged in
  method, with two differences from the original:
  1. `CSV_PATH`/output paths are CLI args (`--csv`, `--outdir`) instead of
     hardcoded, so it can point at either dataset.
  2. The tetrahedra/curl step (`_compute_b3_curl`) is guarded with a
     `MAX_TETRAHEDRA_FOR_CURL` cap. The original THINGS complex has 98
     tetrahedra; item co-purchase cliques in vchoice data produced ~895,000,
     which would require a multi-terabyte dense `B3^T B3` matrix. Above the
     cap, curl is skipped and the reason is printed; the core gradient
     decomposition (the main result) is unaffected.
- `data/vchoice-Yc-Items.txt` — full dataset.
- `data/vchoice-Yc-Items-5-10-4-8.txt` — restricted subset (choice set ≤10,
  selection ≤5, every item selected/shown often enough); matches the
  original paper's tractable experimental setup and is what the commands
  below use.

## Quick start

```bash
pip install numpy scipy pandas

# 1. Convert vchoice sessions into canonical odd-one-out triplets
python3 convert_vchoice.py data/vchoice-Yc-Items-5-10-4-8.txt \
    --out output/vchoice_restricted_triplet_entropy.csv

# 2. Run the HodgeRank pipeline on the converted data
python3 vchoice_hodgerank.py \
    --csv output/vchoice_restricted_triplet_entropy.csv \
    --outdir output/restricted
```

For the full (unrestricted) dataset, cap slate size to avoid combinatorial
blowup (some sessions have choice sets up to 196 items):

```bash
python3 convert_vchoice.py data/vchoice-Yc-Items.txt \
    --out output/vchoice_full_triplet_entropy.csv \
    --max-slate-size 15
```

## Results on the restricted dataset

156,039 sessions → 2,975 items, 763,892 canonical triangles, 296,226 edges.

- **Gradient R² = 95.8%** — most of the triangle-level "odd one out" signal
  is explainable by a consistent pairwise (edge-level) preference structure;
  purchase preference across the catalog is highly rankable rather than
  dominated by cyclic inconsistency.
- **Node categorization**: 2,969 / 2,975 items are `context_dependent`
  (whether an item is the odd one out varies a lot by which other items were
  in the slate) vs. only 6 items with a consistently one-directional
  tendency. Unlike THINGS's perceptual oddness (a fairly stable property of
  a concept), purchase "oddness" here depends heavily on choice-set context.

## Known limitation

An attempt to recover a single global per-item ranking score by projecting
`s1_star` back onto node potentials (`D1^T r ≈ s1_star`) does not work: as
defined here, `s1_star = pinv(D2 D2^T) D2 s²` lives in the triangle-boundary
(curl) subspace of edge space, which is orthogonal to node-potential
gradients by construction (`D1 @ D2 = 0`). That recovery step returns
numerical noise (~1e-10) rather than a meaningful ranking, so it isn't
included here.
