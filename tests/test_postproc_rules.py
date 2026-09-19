import numpy as np
import pandas as pd

from rr.metrics import bouts, confusion, macro_f1, prf_from_cm
from rr.postproc import keep_runs, mode_filter, remove_short_bouts
from rr.rules import apply_rules


def test_mode_filter_removes_isolated_and_keeps_runs():
    x = np.array([0] * 10 + [1] + [0] * 10, dtype=np.int8)
    assert (mode_filter(x) == 0).all()
    y = np.array([0] * 10 + [2] * 10 + [0] * 10, dtype=np.int8)
    assert (mode_filter(y) == y).all()


def test_mode_filter_is_label_permutation_equivariant():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 500).astype(np.int8)
    perm = np.array([0, 3, 1, 2])            # O fixed, S/U/G permuted
    # ties may break differently only when the centre is not among the maxima; check majority agreement
    a = perm[mode_filter(x)]
    b = mode_filter(perm[x].astype(np.int8))
    assert (a == b).mean() > 0.9


def test_remove_short_bouts():
    x = np.array([0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0], dtype=np.int8)
    out = remove_short_bouts(x, 5)
    assert (out[1:4] == 0).all() and (out[5:10] == 2).all()


def test_keep_runs():
    m = np.array([1, 1, 0, 1, 1, 1, 1, 0], bool)
    assert keep_runs(m, 3).tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_confusion_ignores_unscored():
    gt = np.array([-2, -1, 0, 1, 1, 3])
    pr = np.array([1, 1, 0, 1, 2, 3])
    cm = confusion(gt, pr)
    assert cm.sum() == 4 and cm[1, 1] == 1 and cm[1, 2] == 1
    p, r, f = prf_from_cm(cm, 1)
    assert p == 1 and r == 0.5 and np.isclose(f, 2 / 3)


def test_macro_nan_when_class_absent_and_not_predicted():
    cm = confusion(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert macro_f1(cm, (1, 2, 3)) == 1.0          # U, G undefined -> ignored


def test_bouts():
    assert bouts(np.array([0, 1, 1, 0, 1, 0]), 1) == (2, 3 / 25)


def test_rule_priority_and_wall():
    n = 60
    raw = pd.DataFrame({"body_length_ratio": [0.5] * 20 + [1.0] * 40, "area_ratio": [0.5] * 20 + [0.8] * 40,
                        "head_wall_dist": [1.0] * 10 + [10.0] * 10 + [10.0] * 40,
                        "speed": [0.5] * n, "motion_energy": [1.0] * 20 + [5.0] * 40})
    p = {"rear_length_ratio_lt": .75, "rear_area_ratio_lt": .8, "rear_min_frames": 8,
         "supported_head_wall_dist_lt_cm": 3, "groom_speed_lt": 2, "groom_area_ratio_gt": .6,
         "groom_area_ratio_lt": 1.0, "groom_theta_m_percentile": 10, "groom_min_frames": 25}
    pred = apply_rules(raw, p)
    assert (pred[:10] == 1).all() and (pred[10:20] == 2).all() and (pred[20:] == 3).all()
