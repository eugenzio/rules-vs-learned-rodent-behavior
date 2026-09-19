"""DLC <-> MOG2 alignment check (label-free) for all ETH and Stockholm videos.

For each video: DLC body centre -> same homography -> arena cm; cross-correlate speed series
with the MOG2 centroid speed over lags -20..+20 frames. Report best lag, peak correlation and
position RMSE at that lag. Requirement (prereg / plan): peak normalised xcorr >= 0.8.
Writes results/align_check.json.
"""
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.calib import canvas_to_cm  # noqa: E402


def dlc_xy(csv, part):
    d = pd.read_csv(csv, header=[1, 2], index_col=0)
    return d[part]["x"].to_numpy(), d[part]["y"].to_numpy(), d[part]["likelihood"].to_numpy()


def to_cm(H, x, y):
    p = np.stack([x, y, np.ones_like(x)], 0)
    q = np.asarray(H) @ p
    return canvas_to_cm(np.stack([q[0] / q[2], q[1] / q[2]], 1))


def speed(xy):
    v = np.full(len(xy), np.nan)
    v[2:-2] = np.hypot(*(xy[4:] - xy[:-4]).T)
    return v


def best_lag(a, b, max_lag=20):
    """corr(a[t], b[t+lag]) for lag in [-max_lag, max_lag]; b is DLC, indexed by video frame."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        idx = np.arange(len(a))
        j = idx + lag
        ok = (j >= 0) & (j < len(b))
        x, y = a[idx[ok]], b[j[ok]]
        m = np.isfinite(x) & np.isfinite(y)
        out[lag] = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 else np.nan
    lag = max(out, key=lambda k: -np.inf if np.isnan(out[k]) else out[k])
    return lag, out[lag], out


def main():
    meta = json.load(open(ROOT / "results/track_meta.json"))
    man = json.load(open(ROOT / "results/manifest_download.json"))
    stk = {str(t): f for f in glob(str(ROOT / "data/raw/stockholm/tracking/*.csv"))
           for t in [re.search(r"Trial\s+(\d+)DLC", f).group(1)]}
    mapping = json.load(open(ROOT / "results/stockholm_mapping.json"))["label_file_to_trial"]
    res = {}
    for vid, m in sorted(meta.items()):
        if vid.startswith("OFT"):
            csv, part = ROOT / "data/raw/eth_dlc" / man["id_to_dlcfile"][vid], "bodycentre"
        else:
            csv, part = stk[str(mapping[vid.split("_")[1]])], "bodycenter"
        tr = pd.read_parquet(ROOT / f"data/interim/track/{vid}.parquet")
        n_video = m["n"]
        x, y, lk = dlc_xy(csv, part)
        dxy = to_cm(m["H"], x, y)
        dxy[lk < 0.9] = np.nan
        mog = np.full((n_video, 2), np.nan)
        mog[tr["frame"].to_numpy()] = tr[["cx", "cy"]].to_numpy()
        lag, r, curve = best_lag(speed(mog), speed(dxy))
        # position agreement at best lag
        idx = np.arange(n_video)
        j = idx + lag
        ok = (j >= 0) & (j < len(dxy))
        diff = np.hypot(*(mog[idx[ok]] - dxy[j[ok]]).T)
        res[vid] = {"dlc_rows": int(len(x)), "video_frames": int(n_video), "row_diff": int(len(x) - n_video),
                    "best_lag_frames": int(lag), "peak_speed_corr": float(r),
                    "corr_at_lag0": float(curve[0]), "median_pos_err_cm": float(np.nanmedian(diff)),
                    "pass": bool(r >= 0.8)}
        print(f"{vid}: rows DLC-video {len(x) - n_video:+d}  best lag {lag:+d}  r={r:.3f} (lag0 {curve[0]:.3f})"
              f"  median pos err {np.nanmedian(diff):.2f} cm")
    json.dump(res, open(ROOT / "results/align_check.json", "w"), indent=1)
    print("ALL PASS" if all(v["pass"] for v in res.values()) else "SOME FAIL")


if __name__ == "__main__":
    main()
