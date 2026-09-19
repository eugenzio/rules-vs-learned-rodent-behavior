"""Convert ETH labels to frame arrays; dataset summary (Table 1) and inter-rater agreement.

Outputs: data/interim/labels/<vid>.parquet, results/labels_summary.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.labels import (ABSTAIN, AMBIG, EXCL, G, O, RATERS, S, U, drop_invalid, load_labels,  # noqa: E402
                       runs, video_labels)

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
CLS = {S: "Supported", U: "Unsupported", G: "Grooming", O: "Other"}


def f1(a, b, c):
    tp = np.sum((a == c) & (b == c)); fp = np.sum((a != c) & (b == c)); fn = np.sum((a == c) & (b != c))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def fleiss_kappa(votes):
    """votes: (R, n) with codes 0..3, frames with any abstain removed by caller."""
    k = 4
    n_r = votes.shape[0]
    counts = np.stack([(votes == c).sum(0) for c in range(k)], 1)  # (n, k)
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n_r) / (n_r * (n_r - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e)


def main():
    df, bad = drop_invalid(load_labels(ROOT / "data/raw/labels/AllLabDataOFT_final.csv"))
    meta = json.load(open(ROOT / "results/track_meta.json"))
    out_dir = ROOT / "data/interim/labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video, all_votes, all_gt = {}, [], []
    for i in CFG["data"]["eth"]["ids"]:
        vid = f"OFT_{i}"
        n = meta[vid]["n"]
        lv = video_labels(df, vid, n)
        lv1s = video_labels(df, vid, n, drop_exact_1s=True)
        w0, w1 = lv["window"]
        assert [w0, w1] == meta[vid]["window"], vid
        gt = lv["gt"][w0:w1]
        gt_amb_o = np.where(gt == AMBIG, O, gt)
        tab = pd.DataFrame({"frame": np.arange(w0, w1), "gt": gt, "gt_amb_other": gt_amb_o,
                            "gt_drop1s": lv1s["gt"][w0:w1]})
        for r, name in enumerate(RATERS):
            v = lv["votes"][r, w0:w1].copy()
            v[lv["excluded_jump_default"][w0:w1]] = -1
            tab[f"vote_{name}"] = v
        tab.to_parquet(out_dir / f"{vid}.parquet")
        per_video[vid] = {"window": [w0, w1], "n_window": int(w1 - w0),
                          "n_excluded_jump_default": int(lv["excluded_jump_default"][w0:w1].sum()),
                          "n_ambiguous": int((gt == AMBIG).sum()),
                          "counts": {CLS[c]: int((gt == c).sum()) for c in (O, S, U, G)},
                          "bouts": {CLS[c]: len(runs(gt, c)) for c in (S, U, G)}}
        all_votes.append(tab[[f"vote_{r}" for r in RATERS]].to_numpy().T)
        all_gt.append(gt)
    V = np.concatenate(all_votes, 1)
    GT = np.concatenate(all_gt)
    scored = (GT != EXCL)
    n_scored_or_amb = int(scored.sum())
    summ = {"dropped_rows": bad[["ID", "Experimenter", "from", "to", "type"]].astype(str).to_dict("records"),
            "n_videos": 20, "n_frames_in_windows": int(len(GT)),
            "n_excluded_jump_default": int((GT == EXCL).sum()),
            "n_ambiguous": int((GT == AMBIG).sum()),
            "pct_ambiguous_of_nonexcluded": float(100 * (GT == AMBIG).sum() / n_scored_or_amb),
            "n_scored": int(np.isin(GT, [O, S, U, G]).sum())}
    cls = {}
    for c in (S, U, G, O):
        lens = [b - a for vid in per_video for a, b in runs(
            pd.read_parquet(out_dir / f"{vid}.parquet")["gt"].to_numpy(), c)]
        cls[CLS[c]] = {"frames": int((GT == c).sum()), "pct_scored": float(100 * (GT == c).sum() / summ["n_scored"]),
                       "bouts": len(lens), "mean_bout_s": float(np.mean(lens) / 25) if lens else None,
                       "median_bout_s": float(np.median(lens) / 25) if lens else None}
    summ["classes"] = cls
    # ---- inter-rater (frames in window, not Jumping/_DEFAULT, no rater abstaining) ----
    ok = (V >= 0).all(0) & (V != ABSTAIN).all(0)
    summ["interrater_frames"] = int(ok.sum())
    summ["n_frames_with_any_abstain"] = int(((V == ABSTAIN).any(0) & (V >= 0).all(0)).sum())
    pw = {}
    for a, b in combinations(range(3), 2):
        x, y = V[a, ok], V[b, ok]
        pw[f"{RATERS[a]}-{RATERS[b]}"] = {CLS[c]: float(f1(x, y, c)) for c in (S, U, G)}
        pw[f"{RATERS[a]}-{RATERS[b]}"]["macro3"] = float(np.mean([pw[f"{RATERS[a]}-{RATERS[b]}"][CLS[c]] for c in (S, U, G)]))
    summ["pairwise_f1"] = pw
    summ["pairwise_f1_mean"] = {k: float(np.mean([pw[p][k] for p in pw])) for k in ("Supported", "Unsupported", "Grooming", "macro3")}
    summ["fleiss_kappa"] = float(fleiss_kappa(V[:, ok]))
    # per-rater totals
    summ["rater_seconds"] = {RATERS[r]: {CLS[c]: float((V[r] == c).sum() / 25) for c in (S, U, G)} for r in range(3)}
    summ["per_video"] = per_video
    json.dump(summ, open(ROOT / "results/labels_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_video",)}, indent=1))


if __name__ == "__main__":
    main()
