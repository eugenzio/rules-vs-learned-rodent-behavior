"""Error analysis on pooled LOVO predictions.

(1) Pooled 4x4 confusion matrices (RF, tuned rules, wide tuned rules, fixed rules).
(2) Supported<->Unsupported confusion rate among frames with gt in {S,U} and pred in {S,U},
    binned by centroid wall_dist and head_wall_dist (cm; bins (-inf,0),[0,2),[2,4),[4,8),[8,inf)),
    and corner (centroid within 8 cm of two walls) vs side vs centre.
Output: results/error_analysis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.metrics import confusion  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
METHODS = ["fixed_rules", "tuned_rules", "tuned_rules_wide", "rf", "hgb"]
BINS = [-np.inf, 0, 2, 4, 8, np.inf]
BIN_NAMES = ["<0", "0-2", "2-4", "4-8", ">=8"]


def main():
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    side = CFG["data"]["eth"]["arena_side_cm"]
    rows = []
    for v in vids:
        raw = pd.read_parquet(ROOT / f"data/interim/features/{v}.parquet")
        lab = pd.read_parquet(ROOT / f"data/interim/labels/{v}.parquet").set_index("frame")
        tr = pd.read_parquet(ROOT / f"data/interim/track/{v}.parquet").set_index("frame")
        d = pd.DataFrame({"gt": lab["gt"], "wall_dist": raw["wall_dist"], "head_wall_dist": raw["head_wall_dist"],
                          "cx": tr["cx"].reindex(raw.index), "cy": tr["cy"].reindex(raw.index)}, index=raw.index)
        for m in METHODS:
            d[m] = np.load(ROOT / f"data/interim/preds/{m}/{v}.npy")
        rows.append(d.assign(vid=v))
    D = pd.concat(rows)
    D = D[D["gt"] >= 0]
    near_x = (D.cx < 8) | (D.cx > side - 8)
    near_y = (D.cy < 8) | (D.cy > side - 8)
    D["zone"] = np.where(near_x & near_y, "corner", np.where(near_x | near_y, "side", "centre"))
    out = {"confusion": {m: confusion(D["gt"].to_numpy(), D[m].to_numpy()).tolist() for m in METHODS},
           "su_confusion": {}}
    su = D[D["gt"].isin([1, 2])]
    for m in METHODS:
        s = su[su[m].isin([1, 2])]
        err = s[m] != s["gt"]
        r = {}
        for col in ("wall_dist", "head_wall_dist"):
            b = pd.cut(s[col], BINS, labels=BIN_NAMES, right=False)
            g = err.groupby(b, observed=False)
            r[col] = {k: {"rate": float(g.mean()[k]), "n": int(g.size()[k])} for k in BIN_NAMES}
        gz = err.groupby(s["zone"])
        r["zone"] = {k: {"rate": float(gz.mean()[k]), "n": int(gz.size()[k])} for k in gz.size().index}
        r["overall"] = {"rate": float(err.mean()), "n": int(len(err))}
        out["su_confusion"][m] = r
    json.dump(out, open(ROOT / "results/error_analysis.json", "w"), indent=1)
    for m in ("tuned_rules", "rf"):
        print(m, {k: (round(v["rate"], 3), v["n"]) for k, v in out["su_confusion"][m]["zone"].items()},
              {k: round(v["rate"], 3) for k, v in out["su_confusion"][m]["head_wall_dist"].items()})


if __name__ == "__main__":
    main()
