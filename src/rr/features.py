"""Per-frame raw features (14) from tracker geometry + centred window statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calib import signed_wall_dist

RAW = ["speed", "accel", "area_ratio", "body_length_ratio", "aspect", "ang_speed", "solidity",
       "wall_dist", "motion_energy", "head_ratio", "d_wall_min", "head_jitter", "seg_quality",
       "head_wall_dist"]
SIL = RAW[:9]
RULE_FEATS = ["body_length_ratio", "area_ratio", "head_wall_dist", "speed", "motion_energy"]
DLC_RAW = ["dlc_nose_ratio", "dlc_len_ratio", "dlc_nose_jitter", "dlc_mean_likelihood", "dlc_nose_wall_dist"]
STATS = ["mean", "std", "min", "max"]
GEOM = ["cx", "cy", "area", "major", "minor", "angle", "solidity", "head_x", "head_y", "tip_dist",
        "d_wall_min", "seg_quality", "me_raw"]


def _cdiff(x: np.ndarray, h: int, fps: float) -> np.ndarray:
    """Centred difference x[t+h] - x[t-h] divided by the time span."""
    out = np.full(len(x), np.nan)
    out[h:-h] = (x[2 * h:] - x[:-2 * h]) / (2 * h / fps)
    return out


def raw_features(tr: pd.DataFrame, n_frames: int, frame_range: tuple[int, int], side_cm: float,
                 fps: float = 25.0, diff_h: int = 2, jitter_h: int = 5, ffill_max: int = 12) -> pd.DataFrame:
    """tr: tracker output (one row per tracked frame). Returns one row per frame in frame_range."""
    f0, f1 = frame_range
    g = pd.DataFrame(index=np.arange(f0, f1), columns=GEOM, dtype=float)
    t = tr.set_index("frame")
    t = t[t["found"].astype(bool)]
    g.loc[t.index.intersection(g.index), GEOM] = t.loc[t.index.intersection(g.index), GEOM].to_numpy(float)
    missing = g["cx"].isna().to_numpy()
    g = g.ffill(limit=ffill_max)
    ref_area = np.nanmedian(g["area"]); ref_major = np.nanmedian(g["major"]); ref_me = np.nanmedian(g["me_raw"])
    c = g[["cx", "cy"]].to_numpy()
    sp = np.full(len(g), np.nan)
    sp[diff_h:-diff_h] = np.hypot(*(c[2 * diff_h:] - c[:-2 * diff_h]).T) / (2 * diff_h / fps)
    th = g["angle"].to_numpy()
    dth = np.full(len(g), np.nan)
    d = th[2 * diff_h:] - th[:-2 * diff_h]
    d = (d + 90.0) % 180.0 - 90.0      # axis angle is mod 180
    dth[diff_h:-diff_h] = np.abs(d) / (2 * diff_h / fps)
    head = g[["head_x", "head_y"]]
    w = 2 * jitter_h + 1
    jit = np.sqrt(head["head_x"].rolling(w, center=True, min_periods=3).var()
                  + head["head_y"].rolling(w, center=True, min_periods=3).var())
    out = pd.DataFrame({
        "speed": sp,
        "accel": _cdiff(sp, diff_h, fps),
        "area_ratio": g["area"] / ref_area,
        "body_length_ratio": g["major"] / ref_major,
        "aspect": g["minor"] / g["major"],
        "ang_speed": dth,
        "solidity": g["solidity"],
        "wall_dist": signed_wall_dist(c, side_cm),
        "motion_energy": g["me_raw"] / ref_me,
        "head_ratio": g["tip_dist"] / ref_major,
        "d_wall_min": g["d_wall_min"],
        "head_jitter": jit.to_numpy(),
        "seg_quality": g["seg_quality"],
        "head_wall_dist": signed_wall_dist(head.to_numpy(), side_cm),
    }, index=g.index)
    out["missing"] = missing.astype(np.int8)
    out.index.name = "frame"
    return out


def windowed(raw: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Centred rolling mean/std/min/max over +-k frames for each k (min_periods=1)."""
    parts = []
    for k in windows:
        r = raw[cols].rolling(2 * k + 1, center=True, min_periods=1)
        for stat in STATS:
            p = getattr(r, stat)()
            p.columns = [f"{c}_w{k}_{stat}" for c in cols]
            parts.append(p)
    return pd.concat(parts, axis=1)


def design(raw: pd.DataFrame, cols: list[str], windows: list[int], include_missing: bool = True) -> pd.DataFrame:
    X = pd.concat([raw[cols]] + ([windowed(raw, cols, windows)] if windows else []), axis=1)
    if include_missing:
        X["missing"] = raw["missing"]
    return X
