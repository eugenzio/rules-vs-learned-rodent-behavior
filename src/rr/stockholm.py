"""Stockholm external set: label-blind arena detection and label loading."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .labels import G, O, S, U

CLASS_MAP = {"supported rear": S, "unsupported rear": U, "groom": G}


def _fit_line(t, v):
    """Robust least squares v = a*t + b (two passes, drop |resid| > 2*MAD)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    a, b = np.polyfit(t, v, 1)
    r = v - (a * t + b)
    keep = np.abs(r) <= 2 * max(np.median(np.abs(r)), 0.5)
    return np.polyfit(t[keep], v[keep], 1), float(keep.mean())


def detect_floor_corners(bg_bgr: np.ndarray):
    """Floor/wall boundary lines from gradient peaks on the median background.
    Returns corners (tl,tr,br,bl) in image px and fit diagnostics."""
    g = cv2.GaussianBlur(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    H, W = g.shape
    rows = np.arange(int(0.25 * H), int(0.75 * H))
    cols = np.arange(int(0.25 * W), int(0.75 * W))
    lx = int(0.05 * W) + np.argmax(gx[rows, int(0.05 * W):int(0.40 * W)], axis=1)
    rx = int(0.60 * W) + np.argmax(gx[rows, int(0.60 * W):int(0.95 * W)], axis=1)
    ty = int(0.03 * H) + np.argmax(gy[int(0.03 * H):int(0.35 * H), cols], axis=0)
    by = int(0.65 * H) + np.argmax(gy[int(0.65 * H):int(0.97 * H), cols], axis=0)
    (la, lb), lk = _fit_line(rows, lx)   # x = la*y + lb
    (ra, rb), rk = _fit_line(rows, rx)
    (ta, tb), tk = _fit_line(cols, ty)   # y = ta*x + tb
    (ba, bb), bk = _fit_line(cols, by)

    def cross(xa, xb, ya, yb):  # x = xa*y + xb ; y = ya*x + yb
        y = (ya * xb + yb) / (1 - ya * xa)
        return xa * y + xb, y

    tl, tr = cross(la, lb, ta, tb), cross(ra, rb, ta, tb)
    br, bl = cross(ra, rb, ba, bb), cross(la, lb, ba, bb)
    diag = {"inlier_frac": {"left": lk, "right": rk, "top": tk, "bottom": bk}}
    return np.array([tl, tr, br, bl], float), diag


def load_manual_labels(path, n_frames: int) -> pd.DataFrame:
    """Per-frame labels. `Frames` is 1-based -> 0-based index."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[d["Frames"].notna()].copy()
    d["frame"] = d["Frames"].astype(int) - 1
    assert d["frame"].min() >= 0 and d["frame"].max() < n_frames, "label index out of range"
    assert d["frame"].is_unique
    d["y"] = d["Detailed label"].map(CLASS_MAP).fillna(O).astype(int)
    d["stand_and_sniff"] = d["Detailed label"].eq("stand and sniff")
    return d[["frame", "Detailed label", "y", "stand_and_sniff"]].reset_index(drop=True)
