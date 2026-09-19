"""Frame- and bout-level metrics. Codes O=0,S=1,U=2,G=3; only frames with gt in 0..3 are scored."""
from __future__ import annotations

import numpy as np

CLASSES = {1: "Supported", 2: "Unsupported", 3: "Grooming"}


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    m = (gt >= 0) & (gt <= 3)
    return np.bincount(gt[m].astype(int) * 4 + pred[m].astype(int), minlength=16).reshape(4, 4)


def prf_from_cm(cm: np.ndarray, c: int):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan   # NaN only if no gt and no pred
    return p, r, f


def macro_f1(cm: np.ndarray, classes=(1, 2, 3)) -> float:
    fs = [prf_from_cm(cm, c)[2] for c in classes]
    return float(np.nanmean(fs)) if not all(np.isnan(fs)) else np.nan


def scores(cm: np.ndarray) -> dict:
    out = {}
    for c, name in CLASSES.items():
        p, r, f = prf_from_cm(cm, c)
        out[f"{name}_P"], out[f"{name}_R"], out[f"{name}_F1"] = p, r, f
    out["macro3"] = macro_f1(cm, (1, 2, 3))
    out["macro4"] = macro_f1(cm, (0, 1, 2, 3))
    out["macro_SU"] = macro_f1(cm, (1, 2))
    out["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() else np.nan
    return out


def bouts(seq: np.ndarray, c: int, fps: float = 25.0):
    m = np.concatenate([[False], seq == c, [False]])
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return len(s), float((e - s).sum() / fps)
