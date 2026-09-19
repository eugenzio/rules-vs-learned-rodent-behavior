"""Arena calibration: image px -> metric top-down canvas (10 px/cm, fixed margin)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

CORNER_ORDER = ["tl", "tr", "br", "bl"]


def canvas_geometry(side_cm: float, ppcm: float = 10.0, margin_cm: float = 8.0):
    size = int(round((side_cm + 2 * margin_cm) * ppcm))
    m = margin_cm * ppcm
    s = side_cm * ppcm
    dst = np.float32([[m, m], [m + s, m], [m + s, m + s], [m, m + s]])
    return size, dst


def homography(corners_px: np.ndarray, side_cm: float, ppcm: float = 10.0, margin_cm: float = 8.0):
    """corners_px: (4,2) in order tl, tr, br, bl (image coordinates)."""
    size, dst = canvas_geometry(side_cm, ppcm, margin_cm)
    H = cv2.getPerspectiveTransform(np.float32(corners_px), dst)
    return H, size


def canvas_to_cm(xy_px: np.ndarray, ppcm: float = 10.0, margin_cm: float = 8.0) -> np.ndarray:
    """Canvas px -> arena cm, origin at the tl corner of the arena square."""
    return np.asarray(xy_px, dtype=float) / ppcm - margin_cm


def signed_wall_dist(xy_cm: np.ndarray, side_cm: float) -> np.ndarray:
    """Minimum distance to the arena boundary; negative outside the square."""
    x, y = np.asarray(xy_cm, dtype=float).T
    inside = np.minimum.reduce([x, side_cm - x, y, side_cm - y])
    return inside


def dlc_corners(dlc_csv, likelihood_min: float = 0.95) -> tuple[np.ndarray, dict]:
    d = pd.read_csv(dlc_csv, header=[1, 2], index_col=0)
    pts, qc = [], {}
    for c in CORNER_ORDER:
        ok = d[c]["likelihood"] > likelihood_min
        x, y = d.loc[ok, (c, "x")], d.loc[ok, (c, "y")]
        pts.append([x.median(), y.median()])
        qc[c] = {"frac_ok": float(ok.mean()), "sd_x": float(x.std()), "sd_y": float(y.std())}
    return np.array(pts, dtype=float), qc
