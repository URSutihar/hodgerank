# Qwen3.8-27B face-age judgement pipeline

Runs a local vision model over the same face pairs humans judged in
IMDB-WIKI-SbS, so model and human judgements can be compared directly — on
accuracy, on position bias, and on the degree-1 Hodge decomposition this repo
settled on earlier.

Works on Linux, macOS and Windows.

## DeepSeek API smoke test (macOS)

`config.deepseek-smoke.yaml` uses the image-capable `deepseek-flash` API and
reads its credential from macOS Keychain service `deepseek-anthropic`, account
`urs`. Set `model.keychain_service` and `model.keychain_account` to match your
own Keychain entry. The key stays in memory and is never saved in config or
results. Face images are sent to DeepSeek; this backend is not local inference.

Run from the repository root:

```bash
python3 llm_judge/cli.py setup -c llm_judge/config.deepseek-smoke.yaml
python3 llm_judge/cli.py run -c llm_judge/config.deepseek-smoke.yaml --limit 10
python3 llm_judge/cli.py analyze -c llm_judge/config.deepseek-smoke.yaml
```

This config selects five human-judged pairs across age-gap bins and makes ten
judgments with swapped presentations. Results go to a separate dated folder.
It checks API inference and exports; five pairs provide no meaningful Hodge
estimate. Use a separate output directory for mock runs or another model.
Image support documentation: https://api-docs.deepseek.com/guides/vision/

---

## Install

```bash
pip install -r llm_judge/requirements.txt
```

Then Ollama (https://ollama.com/download), and the model:

```bash
ollama serve
ollama pull qwen3.8:27b-q4
```

### Picking a quantization for 24 GB VRAM

Weights, plus ~0.9 GB for the vision projector, plus KV cache.

| tag | size | 24 GB? |
|---|---|---|
| `qwen3.8:27b-q4` | ~16.1 GB | **fits — recommended** |
| `qwen3.8:27b-mtp-q4` | ~16.1 GB | fits; multi-token prediction, faster decode |
| `qwen3.8:27b-nvfp4` | ~16 GB | fits; Blackwell-class GPUs only |
| `qwen3.8:27b-q8` | ~29 GB | **does not fit** — needs ~32 GB |

You asked for 8-bit or 4-bit: at 27B, **8-bit does not fit in 24 GB**. Use q4.
If you want 8-bit fidelity, drop to `qwen3-vl:8b-instruct-q8` (~10 GB).

---

## Run

```bash
cd llm_judge

python cli.py setup      # check deps, data, images, Ollama, model
python cli.py images     # download 9,150 face images (~136 MB, resumable)
python cli.py pairs      # build the pair list
python cli.py run        # judge (resumable, Ctrl-C safe)
python cli.py analyze    # accuracy, position bias, Hodge decomposition
```

No GPU yet? Validate the entire pipeline offline:

```bash
python cli.py run --mock
```

Other commands: `status` (progress), `sync` (rebuild xlsx/parquet from the log).

To run both pair sources without them colliding, point each at its own
directory:

```bash
python cli.py run --out results/human_judged
python cli.py run --out results/clique      # after setting pair_source: clique
```

---

## Output

Three files in `llm_judge/results/`, each with a different job.

| file | role |
|---|---|
| `judgments.jsonl` | **authoritative.** Append-only, `fsync`'d per row. |
| `judgments.parquet` | fast columnar snapshot for analysis |
| `judgments.xlsx` | human-readable snapshot, formatted, two sheets |

### Opening the Excel file mid-run

Safe, and that is by design. The workbook is a *snapshot*, never the source of
truth. On Windows, Excel takes an exclusive lock while a file is open, so a
naive pipeline either crashes or loses rows. This one:

1. keeps appending to `judgments.jsonl` regardless — nothing is ever lost;
2. notices the workbook is locked and prints
   `[info] judgments.xlsx is open — run continues, will sync when you close it`;
3. rewrites the workbook on the next flush after you close it, bringing it to
   the exact judgement count reached — `[ok] Excel unlocked — synced N rows`.

Writes are atomic (temp file + rename), so the workbook is never half-written
even if the process dies mid-save. If a run ends while the file is still open,
close it and run `python cli.py sync`.

### Columns

Model side, human side, and ground truth, side by side:

`row`, `pair_id`, `presentation`, `left_image`, `right_image`, `left_age`,
`right_age`, `age_gap`, `true_older_side`, `model_side`, `model_choice`,
`model_correct`, `model_confidence`, `human_side`, `human_choice`,
`human_correct`, `annotator`, `model_agrees_with_human`, `model_name`,
`quantization`, `prompt_version`, `latency_s`, `raw_response`, `error`,
`timestamp_iso`

`model_correct` and `human_correct` are colour-coded green/red. The **summary**
sheet carries live accuracy, agreement, position-bias rate and per-difficulty
breakdown — it updates with every sync, so opening the file mid-run shows real
progress.

---

## Design notes

**Two pair sources, answering different questions** (`run.pair_source`).

| | `human_judged` (default) | `clique` |
|---|---|---|
| pairs | only ones humans judged | every pair among K faces |
| human coverage | 100% | ~7% |
| triangles @ ~2,400 pairs | ~64 | 54,740 |
| curl | weakly estimated | well estimated |
| harmonic | meaningful | **n/a — structurally 0** |
| answers | does the model match humans? | is the model self-consistent? |

`human_judged` reuses the real comparisons in `crowd_labels.csv`, so every row
has a human choice *and* a ground-truth age gap — that is what makes
model-vs-human comparison possible. Only 250,249 of the 41,856,675 possible
pairs were ever judged (0.6%), and that graph is triangle-poor, so curl stays
weak.

`clique` judges all C(K,2) pairs among K faces, giving C(K,3) triangles —
K=70 buys 54,740 triangles for the same 2,415 judgements a sparse sample
spends on ~64. But a complete graph is contractible: H1 = 0, so **harmonic is
structurally zero for every judge, random noise included**. `analyze` detects
this and prints `n/a` rather than `0.00%`, because a structural constant read
as a finding is exactly the mistake this project already made once.

You cannot have both. Curl needs cycles filled by triangles; harmonic needs
cycles that are *not* filled. Measured, at K=70:

| `clique_keep_frac` | pairs | triangles | oracle harmonic | null harmonic |
|---|---|---|---|---|
| 1.00 | 2,415 | 54,740 | n/a | n/a |
| 0.50 | 1,207 | 6,826 | n/a | n/a |
| 0.25 | 603 | 845 | 0.27% | 1.17% |
| 0.15 | 362 | 188 | 7.80% | 31.60% |

Harmonic only appears once triangles have collapsed, and by then the null
swamps it. Run both modes for the full picture — use `--out` to keep the
results apart.

**Each pair is shown twice, sides swapped** (`swap_repeat: true`). Doubles the
cost and buys the single most important diagnostic for a VLM: if the model's
answer flips when you flip the images, it is reading position, not faces.
`analyze` reports this as `order consistency`; below ~0.75 the judgements are
not about the faces.

**Pair selection defaults to `connected`, and this matters.** Drawing pairs at
random from 250,249 over 9,150 faces yields essentially zero triangles — about
`(n/N)³ × 27,268`, under 1 for any n you can run locally. With no shared faces
there are no triangles, curl and harmonic are undefined, and *every* judge
including random noise scores 100% gradient. That is the exact degeneracy this
project already hit once. So the sampler snowballs a dense subgraph instead.

Even so, curl is weakly estimated at small `n` — the full human graph only has
27,268 triangles over 250,249 pairs (0.109 per pair), so it is triangle-poor by
nature. Gradient and harmonic are sound; treat curl as indicative.

**Read the null, not the raw number.** `analyze` always reports a random-flow
null on the *same* complex. Gradient percentages are not comparable across
different pair sets — the null is the scale. On the full 250K-pair graph the
null sits near 3.6%; on a 2,000-pair subgraph it is far higher simply because
the complex is smaller.

**Determinism.** `temperature: 0.0` and a fixed seed, so a re-run reproduces.

---

## Cost

At 2,000 pairs with `swap_repeat` that is 4,000 model calls. On a 24 GB card
with `27b-q4`, expect roughly 1.5–4 s per call, so about 2–4 hours. Reduce
`n_pairs`, or set `swap_repeat: false` to halve it (at the cost of the
position-bias check). Interrupt any time — `run` resumes exactly where it
stopped.
