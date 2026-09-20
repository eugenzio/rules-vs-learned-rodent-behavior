# Protocol

This project compares threshold rules and learned classifiers on the same low-dimensional CPU tracking features.

## Sequence and integrity rules
1. `config/prereg.yaml` is committed and tagged `prereg-v1` before any tracker feature is joined with a human label.
2. Label-blind quality control may change tracker constants. It uses background images, overlays, coverage and geometry, never labels. Each change is committed as `prereg-v1.x` with its reason in the log below, before `05_features` joins labels.
3. Fixed-rule thresholds are never changed after looking at labels. Tuned rules are tuned on the training folds only.
4. The Stockholm set is external and used once. `10_stockholm.py` refuses to run on a dirty tree. It records the git SHA and model hashes and writes a lock file. Any rerun is disclosed in the paper.
5. Every number in the paper is generated from `results/*.json` into `paper/generated/*.tex`.

## Amendment log
| tag | date | change | reason | label-blind? |
|---|---|---|---|---|
| prereg-v1 | 2026-09-19 | initial | — | yes |
| (not adopted) | 2026-09-19 | considered adding a 7x7 morphological opening to remove tail pixels | label-blind QC: the opening changed median ellipse major axis by only 0.09 cm (STK_1) and 0.06 cm (OFT_5), with major-axis SD 1.34→1.34 and 1.16→1.04 cm, so the tracker was left unchanged | yes |
| post-hoc (exploratory) | 2026-09-19 | added `tuned_rules_wide`: same rule form, rear length/area thresholds searched over 0.60–1.10 (pre-registered max 0.90/0.95) and θ_m percentile 10 added | smoke-fold diagnostics on ETH labels showed rearing frames have median body_length_ratio 0.97–1.00 and area_ratio 0.91–0.95, outside the pre-registered tuning range. Reported **separately** as post-hoc. The pre-registered `tuned_rules` stays the primary comparison. Tuning is still restricted to the training folds. | **no** (designed after seeing label-conditioned feature medians) |

## Notes from the independent audit (2026-09-19)
- **Trial window.** The tracker's frame range comes from the `Start/End` trial markers in the label file, not from behaviour labels. Per-video feature references (median area, median major axis, median motion energy) and the grooming threshold are computed inside that window. This is disclosed in the pre-registration; "label-blind" above means blind to behaviour labels.
- **References after gap filling.** The per-video medians are taken over the forward-filled series (gaps up to 0.5 s), not only over frames with a detection. The pre-registration says "tracked in-trial frames". The affected fraction is at most 1.2% of frames in one video and below 0.1% elsewhere, so the series were not recomputed.
- **Exact tests.** `paired_wilcoxon` originally fell back to the normal approximation whenever any fold had a zero difference, which affected the two grooming comparisons. Fixed to use the exact test on the non-zero pairs (scipy drops zeros with `zero_method="wilcox"`); the corrected p-values are smaller, so no conclusion changed.
- **Error-analysis zones.** Frames without a tracked centroid were being binned as "centre"; they now form their own `untracked` bucket.

## External test: used once (2026-09-19)
`scripts/10_stockholm.py` ran at 23:15 on commit `6775a1f`, with the working tree
clean (the script refuses to run otherwise). It recorded the SHA-256 of every model
it loaded and wrote `results/stockholm/LOCK`. No feature, threshold, model or label
mapping was changed after that run; the Stockholm numbers in the paper are from this
single execution. A rerun would have to be disclosed here.
