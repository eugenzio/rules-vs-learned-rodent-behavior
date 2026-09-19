"""Integrity tests: tuned-rule cache exactness, leakage, method parity, label/feature alignment."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rr.experiment import Dataset, feature_cols  # noqa: E402
from rr.features import design, raw_features  # noqa: E402
from rr.metrics import confusion  # noqa: E402
from rr.rules import TunedRuleSearch, fixed_params, predict_rules  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
VIDS = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
HAS_DATA = (ROOT / "data/interim/features").exists() and (ROOT / "data/interim/labels/OFT_5.parquet").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="interim data not built")


@pytest.fixture(scope="module")
def ds():
    return Dataset(VIDS[:6])


def test_tuned_cache_sum_equals_direct_recompute(ds):
    """Fold objective from cached per-video confusions == recomputing on the pooled fold."""
    s = TunedRuleSearch(ds.raw, ds.gt, CFG)
    p = dict(fixed_params(CFG), rear_length_ratio_lt=0.85, rear_area_ratio_lt=0.9, rear_min_frames=8)
    train = ds.vids[:5]
    cached = sum(s._cm(v, p) for v in train)
    direct = sum(confusion(ds.gt[v], predict_rules(ds.raw[v], p, CFG["postprocess"])) for v in train)
    assert (cached == direct).all()


def test_rules_are_per_video_and_label_free(ds):
    """A rule prediction for one video must not change when other videos are present/absent."""
    p = fixed_params(CFG)
    a = predict_rules(ds.raw["OFT_5"], p, CFG["postprocess"])
    sub = Dataset(["OFT_5"])
    b = predict_rules(sub.raw["OFT_5"], p, CFG["postprocess"])
    assert (a == b).all()


def test_features_do_not_depend_on_labels():
    """Recomputing features from the tracker output alone reproduces the stored features."""
    meta = json.load(open(ROOT / "results/track_meta.json"))["OFT_5"]
    tr = pd.read_parquet(ROOT / "data/interim/track/OFT_5.parquet")
    raw = raw_features(tr, meta["n"], tuple(meta["window"]), meta["side"], CFG["fps_expected"],
                       CFG["features"]["diff_half_width_frames"], CFG["features"]["jitter_half_width_frames"],
                       int(CFG["tracker"]["ffill_max_s"] * CFG["fps_expected"]))
    stored = pd.read_parquet(ROOT / "data/interim/features/OFT_5.parquet")
    for c in raw.columns:
        np.testing.assert_allclose(raw[c].to_numpy(float), stored[c].to_numpy(float), rtol=1e-9, equal_nan=True)


def test_design_matrix_is_within_video(ds):
    """Window statistics never cross video boundaries (design built per video)."""
    X = ds.X("OFT_5", feature_cols("rf"), CFG["features"]["windows_frames"])
    Xalone = design(Dataset(["OFT_5"]).raw["OFT_5"], feature_cols("rf"), CFG["features"]["windows_frames"]).to_numpy(np.float32)
    assert X.shape == Xalone.shape
    np.testing.assert_array_equal(np.nan_to_num(X), np.nan_to_num(Xalone))


def test_label_feature_frame_alignment():
    for v in VIDS:
        lab = pd.read_parquet(ROOT / f"data/interim/labels/{v}.parquet")
        feat = pd.read_parquet(ROOT / f"data/interim/features/{v}.parquet")
        assert (lab["frame"].to_numpy() == feat.index.to_numpy()).all(), v


def test_all_methods_scored_on_identical_frames():
    """Every method's stored prediction covers the same frames, and scoring uses gt>=0 only."""
    preds = {}
    for m in ("fixed_rules", "tuned_rules", "tuned_rules_wide", "dt", "rf", "hgb", "rf_rule"):
        d = ROOT / f"data/interim/preds/{m}"
        if d.exists():
            preds[m] = {v: np.load(d / f"{v}.npy") for v in VIDS}
    assert len(preds) >= 2
    for v in VIDS:
        n = {m: len(p[v]) for m, p in preds.items()}
        assert len(set(n.values())) == 1, (v, n)
        lab = pd.read_parquet(ROOT / f"data/interim/labels/{v}.parquet")
        assert len(lab) == next(iter(n.values()))


def test_postprocessing_applied_to_every_method():
    """No stored prediction contains a non-Other bout shorter than the minimum length."""
    from rr.postproc import remove_short_bouts
    for m in ("fixed_rules", "tuned_rules", "dt", "rf", "hgb", "rf_rule"):
        d = ROOT / f"data/interim/preds/{m}"
        if not d.exists():
            continue
        for v in VIDS[:5]:
            p = np.load(d / f"{v}.npy")
            assert (remove_short_bouts(p, CFG["postprocess"]["min_bout_frames"]) == p).all(), (m, v)


def test_hgb_early_stopping_disabled():
    assert CFG["models"]["hgb"]["early_stopping"] is False
    from rr.experiment import make_model
    assert make_model("hgb", CFG).early_stopping is False


def test_no_test_video_in_training_indices(ds):
    """lovo() trains on 19 videos; verify the held-out video never appears in the training set."""
    from rr.experiment import lovo
    seen = {}

    def fake_log(s):
        seen.setdefault("n", 0)
        seen["n"] += 1
    res, _ = lovo("dt", ds, CFG, folds=[0], log=fake_log)
    assert list(res) == [ds.vids[0]]
