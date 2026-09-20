"""Paired tests and bootstrap CIs (Holm implemented by hand; statsmodels not installed)."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a, b):
    """Exact two-sided Wilcoxon signed-rank on paired values (NaN pairs dropped).
    Returns n, median diff (a-b), p, matched-pairs rank-biserial r."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    n = int(m.sum())
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n": n, "n_nonzero": int(len(nz)), "median_diff": 0.0, "p": 1.0, "rank_biserial": 0.0}
    method = "exact" if len(nz) <= 50 else "approx"   # zeros are dropped by zero_method="wilcox"
    res = wilcoxon(a[m], b[m], alternative="two-sided", method=method, zero_method="wilcox")
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(nz))   # average ranks for ties in |d|
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {"n": n, "n_nonzero": int(len(nz)), "median_diff": float(np.median(d)), "mean_diff": float(np.mean(d)), "p": float(res.pvalue),
            "rank_biserial": float((wp - wm) / (wp + wm)), "method": method}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def bootstrap_ci(values, stat=np.mean, n_boot=10000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def block_bootstrap_f1(gt, pred, cls, block=25, n_boot=10000, seed=0):
    """CI of frame-level F1 for class `cls` resampling contiguous blocks of frames."""
    gt, pred = np.asarray(gt), np.asarray(pred)
    n = len(gt)
    nb = int(np.ceil(n / block))
    starts = np.arange(nb) * block
    rng = np.random.default_rng(seed)
    tp = np.array([np.sum((gt[s:s + block] == cls) & (pred[s:s + block] == cls)) for s in starts])
    fp = np.array([np.sum((gt[s:s + block] != cls) & (pred[s:s + block] == cls)) for s in starts])
    fn = np.array([np.sum((gt[s:s + block] == cls) & (pred[s:s + block] != cls)) for s in starts])
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, nb, nb)
        T, P, N = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
        out.append(2 * T / (2 * T + P + N) if (2 * T + P + N) else np.nan)
    out = np.asarray(out)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))
