"""ONE-SHOT external test: train on all 20 ETH videos, evaluate once on the 2,999 labelled
Stockholm frames. Refuses to run on a dirty tree or if the lock exists (a rerun must be disclosed).

Outputs: results/stockholm/results.json, results/stockholm/LOCK
"""
import argparse
import copy
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.experiment import Dataset, feature_cols, make_model  # noqa: E402
from rr.features import design  # noqa: E402
from rr.metrics import bouts, confusion, scores  # noqa: E402
from rr.postproc import postprocess  # noqa: E402
from rr.rules import TunedRuleSearch, fixed_params, predict_rules  # noqa: E402
from rr.stats import block_bootstrap_f1  # noqa: E402
from rr.stockholm import load_manual_labels  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
OUT = ROOT / "results/stockholm"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-disclosed", action="store_true")
    a = ap.parse_args()
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("REFUSED: tracked files are dirty; commit first.")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "LOCK"
    if lock.exists() and not a.rerun_disclosed:
        sys.exit(f"REFUSED: one-shot already run ({lock.read_text().strip()}).")
    sha = git("rev-parse", "HEAD")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lovo", ROOT / "scripts/06_lovo.py")
    lovo_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lovo_mod)
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    eth = Dataset(vids)
    w = CFG["features"]["windows_frames"]
    pp = CFG["postprocess"]
    # ---- Stockholm inputs ----
    meta = json.load(open(ROOT / "results/track_meta.json"))
    stk = {}
    for k in ("1", "2", "3"):
        vid = f"STK_{k}"
        raw = pd.read_parquet(ROOT / f"data/interim/features/{vid}.parquet")
        lab = load_manual_labels(ROOT / f"data/raw/stockholm/batch2-{k}_manual_labeling.csv", meta[vid]["n"])
        assert raw.index.min() == 0 and len(raw) == meta[vid]["n"]
        stk[vid] = (raw, lab)
    # ---- fit on all ETH ----
    methods, preds, fitted = {}, {}, {}
    p_fixed = fixed_params(CFG)
    fitted["fixed_rules"] = p_fixed
    p_tuned, _ = TunedRuleSearch(eth.raw, eth.gt, CFG).fit(vids)
    fitted["tuned_rules"] = p_tuned
    cw = copy.deepcopy(CFG)
    cw["rules"]["tuned"]["stage1"] = lovo_mod.WIDE["stage1"]
    cw["rules"]["tuned"]["stage3"] = lovo_mod.WIDE["stage3"]
    p_wide, _ = TunedRuleSearch(eth.raw, eth.gt, cw).fit(vids)
    fitted["tuned_rules_wide"] = p_wide
    for v, (raw, _) in stk.items():
        for name, p in (("fixed_rules", p_fixed), ("tuned_rules", p_tuned), ("tuned_rules_wide", p_wide)):
            preds.setdefault(name, {})[v] = predict_rules(raw, p, pp)
    model_hash = {}
    for m in ("dt", "rf", "hgb", "rf_rule"):
        cols = feature_cols(m)
        X = np.concatenate([eth.X(v, cols, w)[eth.gt[v] >= 0] for v in vids])
        y = np.concatenate([eth.gt[v][eth.gt[v] >= 0] for v in vids])
        model = make_model(m, CFG).fit(X, y)
        b = io.BytesIO()
        joblib.dump(model, b, compress=3)
        model_hash[m] = hashlib.sha256(b.getvalue()).hexdigest()
        (ROOT / "data/interim/models").mkdir(parents=True, exist_ok=True)
        (ROOT / f"data/interim/models/{m}_all_eth.joblib").write_bytes(b.getvalue())
        for v, (raw, _) in stk.items():
            Xs = design(raw, cols, w).to_numpy(np.float32)
            preds.setdefault(m, {})[v] = postprocess(model.predict(Xs).astype(np.int8),
                                                     pp["mode_filter_frames"], pp["min_bout_frames"])
    # ---- score ----
    res = {"git_sha": sha, "model_sha256": model_hash, "fitted_rule_params": fitted,
           "n_labelled": int(sum(len(l) for _, l in stk.values())), "analyses": {}}
    for analysis in ("primary", "exclude_stand_and_sniff"):
        A = {}
        for m in preds:
            gts, prs, per_video = [], [], {}
            for v, (raw, lab) in stk.items():
                keep = ~lab["stand_and_sniff"] if analysis != "primary" else np.ones(len(lab), bool)
                g = lab["y"].to_numpy()[keep]
                p = preds[m][v][lab["frame"].to_numpy()[keep]]
                gts.append(g); prs.append(p)
                seg = preds[m][v][lab["frame"].min():lab["frame"].max() + 1]
                per_video[v] = {c: bouts(seg, i)[0] for i, c in ((1, "S"), (2, "U"), (3, "G"))}
            g, p = np.concatenate(gts), np.concatenate(prs)
            cm = confusion(g, p)
            sc = scores(cm)
            ci = {c: block_bootstrap_f1(g, p, i, block=25, n_boot=10000, seed=CFG["seed"])
                  for i, c in ((1, "Supported"), (2, "Unsupported"), (3, "Grooming"))}
            A[m] = {**sc, "cm": cm.tolist(), "ci95": ci, "pred_bouts_in_labelled_segments": per_video,
                    "n_pos": {c: int((g == i).sum()) for i, c in ((1, "Supported"), (2, "Unsupported"), (3, "Grooming"))}}
        human_bouts = {}
        for v, (raw, lab) in stk.items():
            human_bouts[v] = {c: bouts(lab["y"].to_numpy(), i)[0] for i, c in ((1, "S"), (2, "U"), (3, "G"))}
        A["_human_bouts_in_labelled_segments"] = human_bouts
        res["analyses"][analysis] = A
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
    lock.write_text(f"run at {datetime.now().isoformat(timespec='seconds')} on commit {sha}\n"
                    + ("RERUN (disclosed)\n" if a.rerun_disclosed else ""))
    for m in preds:
        s = res["analyses"]["primary"][m]
        print(f"{m:18s} macro3={s['macro3']:.3f} macroSU={s['macro_SU']:.3f} S={s['Supported_F1']:.3f} "
              f"U={s['Unsupported_F1']:.3f} G={s['Grooming_F1']:.3f}")


if __name__ == "__main__":
    main()
