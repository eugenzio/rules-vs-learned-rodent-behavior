"""Threshold rules (fixed a priori, and tuned on training videos by a staged cached grid)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import confusion, macro_f1
from .postproc import keep_runs, postprocess

O, S, U, G = 0, 1, 2, 3


def theta_m(raw: pd.DataFrame, pct: float = 50.0, slow: float = 2.0) -> float:
    """Label-free per-video grooming motion threshold: percentile of motion_energy on slow frames."""
    me = raw.loc[raw["speed"] < slow, "motion_energy"].to_numpy(float)
    me = me[np.isfinite(me)]
    return float(np.percentile(me, pct)) if len(me) else np.inf


def fixed_params(cfg: dict) -> dict:
    f = cfg["rules"]["fixed"]
    return {"rear_length_ratio_lt": f["rear_length_ratio_lt"], "rear_area_ratio_lt": f["rear_area_ratio_lt"],
            "rear_min_frames": f["rear_min_frames"], "supported_head_wall_dist_lt_cm": f["supported_head_wall_dist_lt_cm"],
            "groom_speed_lt": f["groom_speed_lt"], "groom_area_ratio_gt": f["groom_area_ratio_gt"],
            "groom_area_ratio_lt": f["groom_area_ratio_lt"], "groom_theta_m_percentile": 50,
            "groom_min_frames": f["groom_min_frames"]}


def apply_rules(raw: pd.DataFrame, p: dict) -> np.ndarray:
    """Raw per-frame prediction (before shared post-processing). NaN comparisons are False."""
    with np.errstate(invalid="ignore"):
        blr = raw["body_length_ratio"].to_numpy(float)
        ar = raw["area_ratio"].to_numpy(float)
        rear = keep_runs((blr < p["rear_length_ratio_lt"]) & (ar < p["rear_area_ratio_lt"]), p["rear_min_frames"])
        sup = rear & (raw["head_wall_dist"].to_numpy(float) < p["supported_head_wall_dist_lt_cm"])
        th = theta_m(raw, p["groom_theta_m_percentile"])
        groom = keep_runs((raw["speed"].to_numpy(float) < p["groom_speed_lt"]) & (ar > p["groom_area_ratio_gt"])
                          & (ar < p["groom_area_ratio_lt"]) & (raw["motion_energy"].to_numpy(float) > th),
                          p["groom_min_frames"])
    pred = np.full(len(raw), O, dtype=np.int8)
    pred[groom] = G
    pred[rear & ~sup] = U          # rear has priority over groom
    pred[sup] = S
    return pred


def predict_rules(raw, p, pp):
    return postprocess(apply_rules(raw, p), pp["mode_filter_frames"], pp["min_bout_frames"])


class TunedRuleSearch:
    """Staged grid search with per-video confusion caches (exact: fold objective = sum over its videos)."""

    def __init__(self, raws: dict, gts: dict, cfg: dict):
        self.raws, self.gts, self.cfg = raws, gts, cfg
        self.pp = cfg["postprocess"]
        self.base = fixed_params(cfg)
        self.cache: dict = {}

    def _cm(self, vid, p):
        key = (vid, tuple(sorted(p.items())))
        if key not in self.cache:
            self.cache[key] = confusion(self.gts[vid], predict_rules(self.raws[vid], p, self.pp))
        return self.cache[key]

    def _best(self, train, grid, start):
        names = list(grid)
        best, best_score = None, -np.inf
        for vals in itertools.product(*(grid[n] for n in names)):
            p = dict(start, **dict(zip(names, vals)))
            cm = sum(self._cm(v, p) for v in train)
            s = macro_f1(cm, (1, 2, 3))
            if s > best_score + 1e-12:
                best, best_score = p, s
        return best, best_score

    def fit(self, train):
        t = self.cfg["rules"]["tuned"]
        p, _ = self._best(train, t["stage1"], self.base)
        p, _ = self._best(train, t["stage2"], p)
        p, s = self._best(train, t["stage3"], p)
        return p, s
