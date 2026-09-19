"""Bout labels (seconds) -> per-rater frame votes -> 3-rater consensus.

Codes: O=0 (Other), S=1 (Supported rear), U=2 (Unsupported rear), G=3 (Grooming).
Rater vote extra code: ABSTAIN=4 (rater gave >=2 distinct behaviour labels to the frame).
Consensus extra codes: AMBIG=-2 (no class with >=2 votes), EXCL=-1 (outside trial window
or covered by any rater's Jumping/_DEFAULT bout).
Truth table: config/prereg.yaml -> labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

O, S, U, G, ABSTAIN = 0, 1, 2, 3, 4
AMBIG, EXCL = -2, -1
BEHAV = {"Supported": S, "Unsupported": U, "Grooming": G}
MARKERS = {"Start/End", "StartEnd"}
EXCLUDE_TYPES = {"Jumping", "_DEFAULT"}
RATERS = ["Jin", "Furkan", "Oliver"]
CLASS_NAMES = {O: "Other", S: "Supported", U: "Unsupported", G: "Grooming"}


def load_labels(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["from"] = pd.to_numeric(df["from"], errors="coerce")
    df["to"] = pd.to_numeric(df["to"], errors="coerce")
    return df


def drop_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows without an end time (the single NA row) are dropped and returned for logging."""
    bad = df["to"].isna() | df["from"].isna()
    return df[~bad].copy(), df[bad].copy()


def bout_frames(t_from: float, t_to: float, n_frames: int, fps: float = 25.0) -> tuple[int, int]:
    """Half-open frame range [a, b) covered by a bout: round(from*fps) .. round(to*fps)-1."""
    a = int(round(t_from * fps))
    b = int(round(t_to * fps))
    return max(a, 0), min(max(b, 0), n_frames)


def rater_trial_window(rows: pd.DataFrame, fps: float = 25.0, start_max_s: float = 10.0,
                       end_min_s: float = 590.0) -> tuple[int | None, int | None]:
    """First included frame = round(end of earliest marker with from<=10 s);
    first excluded frame = round(start of latest marker with from>=590 s). Stray markers ignored."""
    m = rows[rows["type"].isin(MARKERS)]
    start = m[m["from"] <= start_max_s]
    end = m[m["from"] >= end_min_s]
    s = int(round(start.sort_values("from").iloc[0]["to"] * fps)) if len(start) else None
    e = int(round(end.sort_values("from").iloc[-1]["from"] * fps)) if len(end) else None
    return s, e


def rater_masks(rows: pd.DataFrame, n_frames: int, fps: float = 25.0, drop_exact_1s: bool = False):
    """Boolean masks (3, n) for S/U/G covered by this rater, plus a Jumping/_DEFAULT mask."""
    cov = np.zeros((4, n_frames), dtype=bool)  # rows 1..3 used
    excl = np.zeros(n_frames, dtype=bool)
    types = rows["type"].to_numpy()
    t0 = rows["from"].to_numpy(dtype=float)
    t1 = rows["to"].to_numpy(dtype=float)
    for typ, a_s, b_s in zip(types, t0, t1):
        if typ in BEHAV:
            if drop_exact_1s and abs((b_s - a_s) - 1.0) < 1e-6:
                continue
            a, b = bout_frames(a_s, b_s, n_frames, fps)
            cov[BEHAV[typ], a:b] = True
        elif typ in EXCLUDE_TYPES:
            a, b = bout_frames(a_s, b_s, n_frames, fps)
            excl[a:b] = True
    return cov[1:], excl


def rater_votes(cov: np.ndarray) -> np.ndarray:
    """cov: (3, n) bool for S,U,G -> vote codes."""
    n_lab = cov.sum(0)
    vote = np.full(cov.shape[1], O, dtype=np.int8)
    one = n_lab == 1
    vote[one] = (np.argmax(cov[:, one], axis=0) + 1).astype(np.int8)
    vote[n_lab >= 2] = ABSTAIN
    return vote


def consensus(votes: np.ndarray) -> np.ndarray:
    """votes: (R, n) codes in {O,S,U,G,ABSTAIN}. Class with >=2 votes wins, else AMBIG."""
    counts = np.stack([(votes == c).sum(0) for c in (O, S, U, G)])  # (4, n)
    best = counts.argmax(0)
    gt = np.where(counts.max(0) >= 2, best, AMBIG).astype(np.int8)
    return gt


def video_labels(df: pd.DataFrame, vid: str, n_frames: int, fps: float = 25.0,
                 drop_exact_1s: bool = False) -> dict:
    """Everything needed for one video: per-rater votes, window, exclusion, consensus."""
    d = df[df["ID"] == vid]
    votes, excl_any, starts, ends, windows = [], np.zeros(n_frames, bool), [], [], {}
    for rater in RATERS:
        rows = d[d["Experimenter"] == rater]
        cov, excl = rater_masks(rows, n_frames, fps, drop_exact_1s)
        votes.append(rater_votes(cov))
        excl_any |= excl
        s, e = rater_trial_window(rows, fps)
        windows[rater] = (s, e)
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)
    votes = np.stack(votes)
    w0 = max(starts) if starts else 0
    w1 = min(ends) if ends else n_frames
    in_win = np.zeros(n_frames, bool)
    in_win[w0:w1] = True
    gt = consensus(votes)
    gt = np.where(in_win & ~excl_any, gt, EXCL).astype(np.int8)
    return {"votes": votes, "gt": gt, "window": (w0, w1), "rater_windows": windows,
            "excluded_jump_default": excl_any, "in_window": in_win}


def runs(seq: np.ndarray, value: int) -> list[tuple[int, int]]:
    """Half-open runs [a,b) where seq == value."""
    m = np.concatenate([[False], seq == value, [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
