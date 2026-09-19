"""DLC dense-pose reference features (ETH only; ablation). Alignment lag verified = 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calib import canvas_to_cm, signed_wall_dist

BODY = ["nose", "headcentre", "neck", "earl", "earr", "bodycentre", "bcl", "bcr", "hipl", "hipr",
        "tailbase", "tailcentre", "tailtip"]


def _cm(H, x, y):
    q = np.asarray(H) @ np.stack([x, y, np.ones_like(x)], 0)
    return canvas_to_cm(np.stack([q[0] / q[2], q[1] / q[2]], 1))


def dlc_features(csv, H, frame_range, side_cm, jitter_h=5, lik_min=0.6):
    d = pd.read_csv(csv, header=[1, 2], index_col=0)
    f0, f1 = frame_range
    d = d.iloc[f0:f1]
    pts = {}
    for p in ("nose", "bodycentre", "tailbase"):
        xy = _cm(H, d[p]["x"].to_numpy(), d[p]["y"].to_numpy())
        xy[d[p]["likelihood"].to_numpy() < lik_min] = np.nan
        pts[p] = pd.DataFrame(xy, columns=["x", "y"]).ffill(limit=12).to_numpy()
    L = np.hypot(*(pts["nose"] - pts["tailbase"]).T)
    Lref = np.nanmedian(L)
    w = 2 * jitter_h + 1
    nose = pd.DataFrame(pts["nose"], columns=["x", "y"])
    out = pd.DataFrame({
        "dlc_nose_ratio": np.hypot(*(pts["nose"] - pts["bodycentre"]).T) / Lref,
        "dlc_len_ratio": L / Lref,
        "dlc_nose_jitter": np.sqrt(nose["x"].rolling(w, center=True, min_periods=3).var()
                                   + nose["y"].rolling(w, center=True, min_periods=3).var()).to_numpy(),
        "dlc_mean_likelihood": np.mean([d[p]["likelihood"].to_numpy() for p in BODY], axis=0),
        "dlc_nose_wall_dist": signed_wall_dist(pts["nose"], side_cm),
    }, index=np.arange(f0, f0 + len(d)))
    out.index.name = "frame"
    return out
