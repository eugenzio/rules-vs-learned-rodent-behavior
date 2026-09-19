"""Post-hoc diagnostic (not a model comparison): class-conditional medians of the rule features on
ETH consensus frames, and the fraction of each class meeting the fixed rear / groom conditions.
Output: results/diagnostics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.experiment import Dataset  # noqa: E402
from rr.rules import theta_m  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
NAMES = {0: "Other", 1: "Supported", 2: "Unsupported", 3: "Grooming"}
FEATS = ["body_length_ratio", "area_ratio", "head_wall_dist", "d_wall_min", "speed", "motion_energy", "aspect"]


def main():
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    d = Dataset(vids)
    f = CFG["rules"]["fixed"]
    rows = []
    for v in vids:
        r = d.raw[v].assign(y=d.gt[v], th=theta_m(d.raw[v]))
        rows.append(r)
    R = pd.concat(rows)
    R = R[R.y >= 0]
    out = {"median": {}, "iqr": {}, "frac_fixed_rear_condition": {}, "frac_fixed_groom_condition": {}}
    rear = (R.body_length_ratio < f["rear_length_ratio_lt"]) & (R.area_ratio < f["rear_area_ratio_lt"])
    groom = (R.speed < f["groom_speed_lt"]) & (R.area_ratio > f["groom_area_ratio_gt"]) & \
            (R.area_ratio < f["groom_area_ratio_lt"]) & (R.motion_energy > R.th)
    for c, name in NAMES.items():
        s = R[R.y == c]
        out["median"][name] = {k: float(s[k].median()) for k in FEATS}
        out["iqr"][name] = {k: [float(s[k].quantile(.25)), float(s[k].quantile(.75))] for k in FEATS}
        out["frac_fixed_rear_condition"][name] = float(rear[R.y == c].mean())
        out["frac_fixed_groom_condition"][name] = float(groom[R.y == c].mean())
    json.dump(out, open(ROOT / "results/diagnostics.json", "w"), indent=1)
    print(json.dumps(out["median"], indent=1))
    print(json.dumps(out["frac_fixed_rear_condition"], indent=1), json.dumps(out["frac_fixed_groom_condition"], indent=1))


if __name__ == "__main__":
    main()
