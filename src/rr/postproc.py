"""Shared post-processing for every method: 9-frame mode filter, then short non-Other bouts -> Other."""
from __future__ import annotations

import numpy as np

N_CLASSES = 4  # O,S,U,G


def mode_filter(pred: np.ndarray, width: int = 9) -> np.ndarray:
    """Centred majority filter over class codes 0..3 (window truncated at the edges).
    Ties: keep the centre label if it is among the tied maxima, else the smallest code."""
    n = len(pred)
    h = width // 2
    onehot = np.zeros((N_CLASSES, n + 1), dtype=np.int32)
    onehot[pred, np.arange(1, n + 1)] = 1
    cs = np.cumsum(onehot, axis=1)
    idx = np.arange(n)
    lo = np.clip(idx - h, 0, n)
    hi = np.clip(idx + h + 1, 0, n)
    counts = cs[:, hi] - cs[:, lo]                   # (4, n)
    mx = counts.max(0)
    centre_is_max = counts[pred, idx] == mx
    first_max = np.argmax(counts == mx, axis=0)
    return np.where(centre_is_max, pred, first_max).astype(pred.dtype)


def remove_short_bouts(pred: np.ndarray, min_len: int = 5, other: int = 0) -> np.ndarray:
    """Runs of a non-Other class shorter than min_len become Other."""
    out = pred.copy()
    n = len(pred)
    if n == 0:
        return out
    change = np.flatnonzero(np.diff(pred)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    for a, b in zip(starts, ends):
        if pred[a] != other and b - a < min_len:
            out[a:b] = other
    return out


def postprocess(pred: np.ndarray, width: int = 9, min_len: int = 5) -> np.ndarray:
    return remove_short_bouts(mode_filter(np.asarray(pred, dtype=np.int8), width), min_len)


def keep_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs of length >= min_len."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros_like(mask)
    for a, b in zip(s, e):
        if b - a >= min_len:
            out[a:b] = True
    return out
