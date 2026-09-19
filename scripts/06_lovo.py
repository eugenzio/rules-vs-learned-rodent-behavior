"""Leave-one-video-out on the 20 ETH videos.

Methods: fixed_rules, tuned_rules (pre-registered grid), tuned_rules_wide (post-hoc exploratory,
see protocol.md), dt, rf, hgb, rf_rule.
Outputs: results/lovo/<method>.json, data/interim/preds/<method>/<vid>.npy
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.experiment import Dataset, lovo  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))

# Post-hoc exploratory grid (NOT pre-registered): same rule form, rear thresholds extended to 1.10
# after the class-conditional medians showed rearing silhouettes barely shrink (protocol.md).
WIDE = {"stage1": {"rear_length_ratio_lt": [round(0.60 + 0.05 * i, 2) for i in range(11)],
                   "rear_area_ratio_lt": [round(0.60 + 0.05 * i, 2) for i in range(11)],
                   "rear_min_frames": [4, 8, 12, 16]},
        "stage3": {"groom_speed_lt": [1, 2, 3, 4], "groom_area_ratio_gt": [0.5, 0.6, 0.7],
                   "groom_area_ratio_lt": [0.9, 1.0, 1.1], "groom_theta_m_percentile": [10, 30, 50, 70, 90],
                   "groom_min_frames": [12, 25, 50]}}


def cfg_for(method):
    if method != "tuned_rules_wide":
        return CFG
    c = copy.deepcopy(CFG)
    c["rules"]["tuned"]["stage1"] = WIDE["stage1"]
    c["rules"]["tuned"]["stage3"] = WIDE["stage3"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--gt", default="gt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    data = Dataset(vids, gt_key=a.gt)
    out = ROOT / "results/lovo"
    out.mkdir(parents=True, exist_ok=True)
    for m in a.methods:
        name = m + a.tag
        method = "tuned_rules" if m == "tuned_rules_wide" else m
        print(f"== {name}", flush=True)
        res, extra = lovo(method, data, cfg_for(m), pred_dir=ROOT / f"data/interim/preds/{name}",
                          log=lambda s: print(s, flush=True))
        json.dump({"method": name, "gt_key": a.gt, "per_video": res, "extra": extra},
                  open(out / f"{name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
