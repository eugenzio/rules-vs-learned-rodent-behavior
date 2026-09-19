from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rr.labels import (ABSTAIN, AMBIG, EXCL, G, O, RATERS, S, U, bout_frames, consensus,
                       drop_invalid, load_labels, rater_masks, rater_trial_window, rater_votes,
                       video_labels)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/raw/labels/AllLabDataOFT_final.csv"


def rows(*triples):
    return pd.DataFrame(triples, columns=["type", "from", "to"])


# ---------- frame conversion ----------
def test_bout_frames_half_open_round():
    assert bout_frames(1.0, 2.0, 1000) == (25, 50)
    assert bout_frames(4.836, 6.586, 1000) == (round(4.836 * 25), round(6.586 * 25))
    assert bout_frames(-0.1, 0.5, 1000) == (0, 12)          # clipped at 0
    assert bout_frames(39.0, 41.0, 1000) == (975, 1000)     # clipped at n


# ---------- rater vote truth table ----------
@pytest.mark.parametrize("labels,expected", [
    ([], O),
    (["Supported"], S),
    (["Unsupported"], U),
    (["Grooming"], G),
    (["Supported", "Supported"], S),          # duplicate bouts of same class
    (["Supported", "Unsupported"], ABSTAIN),
    (["Unsupported", "Grooming"], ABSTAIN),
    (["Supported", "Grooming"], ABSTAIN),
])
def test_rater_vote_table(labels, expected):
    r = rows(*[(t, 1.0, 2.0) for t in labels]) if labels else rows()
    cov, _ = rater_masks(r, 100)
    v = rater_votes(cov)
    assert v[30] == expected and v[10] == O


# ---------- consensus truth table ----------
@pytest.mark.parametrize("votes,expected", [
    ((S, S, O), S), ((S, S, S), S), ((S, S, ABSTAIN), S), ((U, U, G), U), ((G, G, O), G),
    ((O, O, ABSTAIN), O), ((O, O, O), O),
    ((S, U, O), AMBIG), ((S, O, ABSTAIN), AMBIG), ((S, ABSTAIN, ABSTAIN), AMBIG),
    ((ABSTAIN, ABSTAIN, ABSTAIN), AMBIG), ((S, U, G), AMBIG),
])
def test_consensus_table(votes, expected):
    v = np.array(votes, dtype=np.int8)[:, None]
    assert consensus(v)[0] == expected


# ---------- trial window ----------
def test_trial_window_ignores_stray_and_uses_edges():
    r = rows(("StartEnd", 2.0, 3.0), ("StartEnd", 172.0, 173.0), ("StartEnd", 600.0, 601.0))
    assert rater_trial_window(r) == (75, 15000)


def test_trial_window_double_start_uses_earliest_marker():
    r = rows(("StartEnd", 4.261, 5.261), ("StartEnd", 3.856, 4.648), ("StartEnd", 600.1, 601.1))
    assert rater_trial_window(r)[0] == round(4.648 * 25)


def test_trial_window_missing_end_is_none():
    assert rater_trial_window(rows(("StartEnd", 1.9, 2.1)))[1] is None


# ---------- real data ----------
@pytest.fixture(scope="module")
def df():
    if not LABELS.exists():
        pytest.skip("labels not downloaded")
    d, bad = drop_invalid(load_labels(LABELS))
    assert len(bad) == 1 and bad.iloc[0]["ID"] == "OFT_50" and bad.iloc[0]["Experimenter"] == "Oliver"
    return d


def test_file_shape(df):
    assert len(df) == 5517 and set(df["Experimenter"]) == set(RATERS) and df["ID"].nunique() == 20


def test_one_dlcfile_per_id(df):
    assert (df.groupby("ID")["DLCFile"].nunique() == 1).all()


def _union_len(intervals):
    """Independent implementation: merged length of half-open intervals."""
    tot, cur = 0, None
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0)


def test_conservation_all_raters_videos(df):
    n = 15300
    for (vid, rater), d in df.groupby(["ID", "Experimenter"]):
        cov, _ = rater_masks(d, n)
        for code, name in ((S, "Supported"), (U, "Unsupported"), (G, "Grooming")):
            iv = [bout_frames(a, b, n) for a, b in d.loc[d["type"] == name, ["from", "to"]].to_numpy()]
            assert cov[code - 1].sum() == _union_len(iv), (vid, rater, name)


def test_windows_sane(df):
    for vid in df["ID"].unique():
        w0, w1 = video_labels(df, vid, 15300)["window"]
        assert 25 * 1 < w0 < 25 * 6 and 25 * 597 < w1 < 25 * 602, (vid, w0, w1)


def test_furkan_oft41_double_labelled_rears_abstain(df):
    out = video_labels(df, "OFT_41", 15300)
    fur = out["votes"][RATERS.index("Furkan")]
    seg = fur[round(13.8 * 25):round(15.4 * 25)]      # first S+U double-labelled rear
    assert (seg == ABSTAIN).all()


def test_jumping_excluded_for_everyone(df):
    j = df[df["type"] == "Jumping"].iloc[0]
    out = video_labels(df, j["ID"], 15300)
    a, b = bout_frames(j["from"], j["to"], 15300)
    assert (out["gt"][a:b] == EXCL).all()


def test_consensus_codes_only(df):
    out = video_labels(df, "OFT_5", 15300)
    assert set(np.unique(out["gt"])) <= {EXCL, AMBIG, O, S, U, G}
