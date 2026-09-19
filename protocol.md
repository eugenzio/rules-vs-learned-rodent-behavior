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
