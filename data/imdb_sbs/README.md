# IMDB-WIKI-SbS Data

From the repository root, after installing llm_judge/requirements.txt:

```bash
python3 llm_judge/cli.py data
python3 llm_judge/cli.py images
```

The first command downloads gt.csv and crowd_labels.csv from the
[official dataset repository](https://github.com/Toloka/IMDB-WIKI-SbS), at
revision 6087435b6eb61993c1169e232a827fd18b51a7c1, and checks SHA-256 hashes.
The second downloads images/ from the URLs in gt.csv. Existing nonempty image
files are skipped; rerun images after a failed or interrupted download.

CSV files, photos, and partial downloads are intentionally Git-ignored.
Only this bootstrap guide is committed. Expect roughly 70 MB of metadata
plus 136 MB of photos. Downloads require network access; upstream image URLs
may become unavailable. File paths can be changed under data in the YAML config.

Do not replace these files with the generic IMDB-WIKI face archive: this
pipeline needs the specific SbS human comparisons and image URL table.

Dataset: Nikita Pavlichenko and Dmitry Ustalov, *IMDB-WIKI-SbS: An Evaluation
Dataset for Crowdsourced Pairwise Comparisons* (2021),
[paper](https://arxiv.org/abs/2110.14990). Consult upstream terms and source
image rights before redistribution or other uses.
