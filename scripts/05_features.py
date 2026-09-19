"""Raw per-frame features (+ DLC reference features for ETH). One parquet per video.

Outputs: data/interim/features/<vid>.parquet, results/feature_qc.json (unit/range assertions).
"""
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.dlc import dlc_features  # noqa: E402
from rr.features import RAW, raw_features  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
F = CFG["features"]

# physically plausible ranges (assertions on medians / 99th percentiles)
RANGES = {"speed": (0, 60), "area_ratio": (0.2, 3.0), "body_length_ratio": (0.2, 3.0), "aspect": (0.05, 1.0),
          "solidity": (0.3, 1.0), "wall_dist": (-10, 22), "head_ratio": (0.1, 1.5), "seg_quality": (0, 1.0001),
          "head_wall_dist": (-12, 22), "d_wall_min": (-12, 22)}


def main():
    meta = json.load(open(ROOT / "results/track_meta.json"))
    man = json.load(open(ROOT / "results/manifest_download.json"))
    out = ROOT / "data/interim/features"
    out.mkdir(parents=True, exist_ok=True)
    qc = {}
    for vid, m in sorted(meta.items()):
        tr = pd.read_parquet(ROOT / f"data/interim/track/{vid}.parquet")
        raw = raw_features(tr, m["n"], tuple(m["window"]), m["side"], CFG["fps_expected"],
                           F["diff_half_width_frames"], F["jitter_half_width_frames"],
                           int(CFG["tracker"]["ffill_max_s"] * CFG["fps_expected"]))
        if vid.startswith("OFT"):
            dl = dlc_features(ROOT / "data/raw/eth_dlc" / man["id_to_dlcfile"][vid], m["H"], tuple(m["window"]), m["side"])
            raw = raw.join(dl)
        raw.to_parquet(out / f"{vid}.parquet")
        q = {}
        for c in RAW:
            v = raw[c].to_numpy(float)
            q[c] = {"median": float(np.nanmedian(v)), "p01": float(np.nanpercentile(v, 1)),
                    "p99": float(np.nanpercentile(v, 99)), "nan_frac": float(np.isnan(v).mean())}
            if c in RANGES:
                lo, hi = RANGES[c]
                assert lo <= q[c]["median"] <= hi, (vid, c, q[c])
        q["missing_frac"] = float(raw["missing"].mean())
        qc[vid] = q
        print(vid, "speed med %.2f  area med %.2f  len med %.2f  headwall med %.1f  missing %.4f" % (
            q["speed"]["median"], q["area_ratio"]["median"], q["body_length_ratio"]["median"],
            q["head_wall_dist"]["median"], q["missing_frac"]))
    json.dump(qc, open(ROOT / "results/feature_qc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
