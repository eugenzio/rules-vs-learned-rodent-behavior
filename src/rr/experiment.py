"""Shared experiment machinery: data loading, design matrices, models, LOVO loop."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .features import RAW, RULE_FEATS, design
from .metrics import confusion, scores
from .postproc import postprocess
from .rules import TunedRuleSearch, fixed_params, predict_rules

ROOT = Path(__file__).resolve().parents[2]
LEARNED = {"dt", "rf", "hgb", "rf_rule"}


def load_video(vid: str, with_labels: bool = True):
    raw = pd.read_parquet(ROOT / f"data/interim/features/{vid}.parquet")
    if not with_labels:
        return raw, None
    lab = pd.read_parquet(ROOT / f"data/interim/labels/{vid}.parquet").set_index("frame")
    assert (lab.index == raw.index).all(), vid
    return raw, lab


def make_model(name: str, cfg: dict):
    m, seed = cfg["models"], cfg["seed"]
    cw = m["class_weight"]
    if name == "dt":
        return DecisionTreeClassifier(max_depth=m["dt"]["max_depth"], class_weight=cw, random_state=seed)
    if name in ("rf", "rf_rule"):
        r = m["rf"]
        return RandomForestClassifier(n_estimators=r["n_estimators"], min_samples_leaf=r["min_samples_leaf"],
                                      max_samples=r["max_samples"], class_weight=cw, n_jobs=r["n_jobs"],
                                      random_state=seed)
    if name == "hgb":
        h = m["hgb"]
        return HistGradientBoostingClassifier(max_iter=h["max_iter"], early_stopping=h["early_stopping"],
                                              class_weight=cw, random_state=seed)
    raise ValueError(name)


def feature_cols(method: str, source: str = "main") -> list[str]:
    if method == "rf_rule":
        return list(RULE_FEATS)
    if source == "silhouette_only":
        return RAW[:9]
    if source == "dlc_reference":
        return RAW[:9] + ["dlc_nose_ratio", "dlc_len_ratio", "dlc_nose_jitter", "dlc_mean_likelihood",
                          "dlc_nose_wall_dist"]
    return list(RAW)


class Dataset:
    def __init__(self, vids, gt_key="gt"):
        self.vids = list(vids)
        self.raw, self.lab, self.gt = {}, {}, {}
        for v in self.vids:
            r, l = load_video(v)
            self.raw[v], self.lab[v], self.gt[v] = r, l, l[gt_key].to_numpy().astype(np.int8)
        self._X: dict = {}

    def X(self, vid, cols, windows):
        key = (vid, tuple(cols), tuple(windows))
        if key not in self._X:
            self._X[key] = design(self.raw[vid], cols, windows).to_numpy(np.float32)
        return self._X[key]


def train_predict(method, data: Dataset, train, test_vids, cfg, cols, windows, test_raw=None, test_X=None):
    """Fit on `train` videos, return {vid: postprocessed prediction} for test videos (+ fitted object)."""
    pp = cfg["postprocess"]
    if method == "fixed_rules":
        p = fixed_params(cfg)
        return {v: predict_rules(test_raw[v], p, pp) for v in test_vids}, p
    if method == "tuned_rules":
        raise RuntimeError("use TunedRuleSearch via lovo()")
    Xtr = np.concatenate([data.X(v, cols, windows)[data.gt[v] >= 0] for v in train])
    ytr = np.concatenate([data.gt[v][data.gt[v] >= 0] for v in train])
    model = make_model(method, cfg).fit(Xtr, ytr)
    out = {}
    for v in test_vids:
        Xte = test_X[v] if test_X is not None else data.X(v, cols, windows)
        out[v] = postprocess(model.predict(Xte).astype(np.int8), pp["mode_filter_frames"], pp["min_bout_frames"])
    return out, model


def lovo(method, data: Dataset, cfg, source="main", windows=None, folds=None, pred_dir=None, log=print):
    windows = cfg["features"]["windows_frames"] if windows is None else windows
    cols = feature_cols(method, source)
    res, extra = {}, {}
    search = TunedRuleSearch(data.raw, data.gt, cfg) if method == "tuned_rules" else None
    for i, test in enumerate(data.vids):
        if folds is not None and i not in folds:
            continue
        train = [v for v in data.vids if v != test]
        t0 = time.perf_counter()
        if method == "tuned_rules":
            p, s = search.fit(train)
            pred = predict_rules(data.raw[test], p, cfg["postprocess"])
            extra[test] = {"params": p, "train_macro3": s}
        else:
            preds, _ = train_predict(method, data, train, [test], cfg, cols, windows, test_raw=data.raw)
            pred = preds[test]
        cm = confusion(data.gt[test], pred)
        res[test] = {"cm": cm.tolist(), **scores(cm), "seconds": time.perf_counter() - t0}
        if pred_dir is not None:
            Path(pred_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(pred_dir) / f"{test}.npy", pred)
        log(f"  {method:12s} {test:7s} macro3={res[test]['macro3']:.3f}  S={res[test]['Supported_F1']:.3f} "
            f"U={res[test]['Unsupported_F1']:.3f} G={res[test]['Grooming_F1']:.3f}  ({res[test]['seconds']:.1f}s)")
    return res, extra
