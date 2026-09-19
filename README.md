# How Far Do Threshold Rules Go?

This repository compares hand-written threshold rules with lightweight learned classifiers (decision tree, random forest, gradient boosting) for rodent behaviour classification. Every method gets the **same** low-dimensional per-frame features from a CPU-only silhouette tracker. The behaviours are supported rearing, unsupported rearing and grooming. Evaluation covers animal shift (leave-one-video-out) and lab shift (an external dataset).

The analysis was pre-registered. `config/prereg.yaml` and `protocol.md` were committed and tagged (`prereg-v1`) before any tracker feature was joined with a human label. `protocol.md` logs every later change, including one post-hoc analysis that is labelled as post-hoc.

## Data (downloaded by `scripts/00_download.py`, not redistributed)
| Source | What | License | Pinned |
|---|---|---|---|
| Zenodo [10.5281/zenodo.3608658](https://doi.org/10.5281/zenodo.3608658) | 20 open-field videos (Sturman et al., *Neuropsychopharmacology* 2020) | CC-BY-4.0 | md5 from the Zenodo API |
| [ETHZ-INS/DLCAnalyzer](https://github.com/ETHZ-INS/DLCAnalyzer) | 3-rater labels `data/OFT/Labels/AllLabDataOFT_final.csv` | GPL-3.0 | commit `d6f9532d191e388a32bef2063f7fd4067946dcec` |
| same repo | DeepLabCut outputs (arena corners; pose used only in one reference ablation) | GPL-3.0 | commit `78cf2233706dda43d6fd7aa12a0114a8743625aa` |
| Zenodo [10.5281/zenodo.14382973](https://doi.org/10.5281/zenodo.14382973) | Stockholm external set (Mlost et al., *Patterns* 2025): batch-2 videos, tracking, manual labels | CC-BY-4.0 | md5 from the Zenodo API |

## Released artifacts
- `release/features/*.parquet`: per-frame tracker features for all 23 videos (float32).
- `release/predictions/*.npz`: leave-one-video-out predictions per method (0 Other, 1 Supported, 2 Unsupported, 3 Grooming).
- `results/*.json`: every number reported in the paper. `paper/generated/numbers.tex` is generated from these files by `scripts/12_tables.py`.

Consensus labels are not redistributed. `scripts/01_labels.py` rebuilds them from the pinned source.

## Reproduce
```bash
~/miniforge3/bin/python3.12 -m venv --system-site-packages .venv   # needs numpy, pandas, scikit-learn, opencv, scipy, pyyaml, pyarrow
.venv/bin/pip install pytest
.venv/bin/python scripts/00_download.py          # verified downloads
# Stockholm: extract Trials 1-3 from batch2_videos.rar and batch2_tracking.rar into data/raw/stockholm/{videos,tracking}
.venv/bin/python -m pytest -q                    # label truth-table and post-processing tests
.venv/bin/python scripts/03_track.py && .venv/bin/python scripts/03_track.py --stockholm
.venv/bin/python scripts/01_labels.py && .venv/bin/python scripts/04_align_check.py && .venv/bin/python scripts/05_features.py
.venv/bin/python scripts/06_lovo.py fixed_rules tuned_rules tuned_rules_wide dt rf hgb rf_rule
.venv/bin/python scripts/07_ablate.py windows && .venv/bin/python scripts/07_ablate.py source && .venv/bin/python scripts/07_ablate.py sensitivity
.venv/bin/python scripts/08_human.py && .venv/bin/python scripts/13_error.py && .venv/bin/python scripts/09_cost.py
.venv/bin/python scripts/10_stockholm.py         # one-shot external test (locked after the first run)
.venv/bin/python scripts/11_figures.py && .venv/bin/python scripts/12_tables.py
```
Before the Stockholm run, `results/stockholm_mapping.json` must list the video-to-label-file mapping. It is derived from frame counts and the mutual information between the deposited unsupervised cluster sequences and the pose tracks, without using any labels.

The tracker's segmentation constants and head/tail logic come from the author's own MIT-licensed tracker.

## License
Code: MIT (see `LICENSE`). Released features and predictions: CC-BY-4.0, with attribution to the source datasets above.
