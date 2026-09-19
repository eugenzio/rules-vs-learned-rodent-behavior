"""Ablations (RF unless stated), all LOVO on ETH.

windows    : raw + one window at a time, k in {0,5,12,25,50}
source     : silhouette_only (9 raw), dlc_reference (9 silhouette + 5 DLC)   [main = 14 contour-proxy]
sensitivity: labels gt_amb_other / gt_drop1s, for RF and tuned_rules
Outputs: results/ablations/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.experiment import Dataset, lovo  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
OUT = ROOT / "results/ablations"


def save(name, res, extra=None, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"name": name, "per_video": res, "extra": extra or {}, **kw}, open(OUT / f"{name}.json", "w"),
              indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["windows", "source", "sensitivity"])
    a = ap.parse_args()
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.which == "windows":
        d = Dataset(vids)
        for k in CFG["evaluation"]["ablations"]["windows_single"]:
            w = [] if k == 0 else [k]
            print(f"== rf windows {w}", flush=True)
            res, _ = lovo("rf", d, CFG, windows=w, log=log)
            save(f"rf_window_{k}", res, windows=w)
    elif a.which == "source":
        d = Dataset(vids)
        for src in ("silhouette_only", "dlc_reference"):
            print(f"== rf source {src}", flush=True)
            res, _ = lovo("rf", d, CFG, source=src, log=log)
            save(f"rf_source_{src}", res, source=src)
    else:
        for key in ("gt_amb_other", "gt_drop1s"):
            d = Dataset(vids, gt_key=key)
            for m in ("tuned_rules", "rf"):
                print(f"== {m} labels {key}", flush=True)
                res, extra = lovo(m, d, CFG, log=log)
                save(f"{m}_{key}", res, extra, gt_key=key)


if __name__ == "__main__":
    main()
