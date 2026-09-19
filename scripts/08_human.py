"""Human ceiling (secondary, like-for-like) and bout-level agreement.

(1) Leave-one-rater-out: each rater vs the consensus of the other two (frames where both agree and
    neither abstains/excluded); every method's LOVO predictions scored against the same 3 references.
(2) Bout level per video: count and total duration per class. Human = median over 3 raters, each
    rater sequence passed through the same post-processing. Pearson r and median relative error.
Outputs: results/human_ceiling.json, results/bouts.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.labels import ABSTAIN, RATERS  # noqa: E402
from rr.metrics import bouts, confusion, scores  # noqa: E402
from rr.postproc import postprocess  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
METHODS = ["fixed_rules", "tuned_rules", "tuned_rules_wide", "dt", "rf", "hgb", "rf_rule"]
NAMES = {1: "Supported", 2: "Unsupported", 3: "Grooming"}


def main():
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    pp = CFG["postprocess"]
    labs = {v: pd.read_parquet(ROOT / f"data/interim/labels/{v}.parquet") for v in vids}
    preds = {m: {v: np.load(ROOT / f"data/interim/preds/{m}/{v}.npy") for v in vids}
             for m in METHODS if (ROOT / f"data/interim/preds/{m}").exists()}
    # ---------- (1) leave-one-rater-out ----------
    ceil = {"human": {}, **{m: {} for m in preds}}
    for r_i, r in enumerate(RATERS):
        others = [o for o in RATERS if o != r]
        cm_h = np.zeros((4, 4), int)
        cm_m = {m: np.zeros((4, 4), int) for m in preds}
        for v in vids:
            L = labs[v]
            a, b, me = L[f"vote_{others[0]}"].to_numpy(), L[f"vote_{others[1]}"].to_numpy(), L[f"vote_{r}"].to_numpy()
            ok = (a == b) & (a >= 0) & (a != ABSTAIN) & (me >= 0) & (me != ABSTAIN)
            ref = np.where(ok, a, -1)
            cm_h += confusion(ref, me)
            for m in preds:
                cm_m[m] += confusion(ref, preds[m][v])
        ceil["human"][r] = scores(cm_h)
        for m in preds:
            ceil[m][r] = scores(cm_m[m])
    summary = {k: {s: float(np.mean([ceil[k][r][s] for r in RATERS])) for s in
                   ("Supported_F1", "Unsupported_F1", "Grooming_F1", "macro3")} for k in ceil}
    json.dump({"per_reference": ceil, "mean_over_references": summary,
               "note": "pooled over videos per reference; reference = 2 other raters agreeing"},
              open(ROOT / "results/human_ceiling.json", "w"), indent=1, default=float)
    # ---------- (2) bout level ----------
    out = {"human_median": {}, "methods": {}}
    human = {c: {"count": [], "dur": []} for c in NAMES}
    for v in vids:
        L = labs[v]
        per_r = []
        for r in RATERS:
            s = L[f"vote_{r}"].to_numpy().copy()
            s[(s < 0) | (s == ABSTAIN)] = 0
            per_r.append(postprocess(s.astype(np.int8), pp["mode_filter_frames"], pp["min_bout_frames"]))
        for c in NAMES:
            cnt = [bouts(s, c)[0] for s in per_r]
            dur = [bouts(s, c)[1] for s in per_r]
            human[c]["count"].append(float(np.median(cnt)))
            human[c]["dur"].append(float(np.median(dur)))
    out["human_median"] = {NAMES[c]: human[c] for c in NAMES}
    for m in preds:
        res = {}
        for c in NAMES:
            pc = [bouts(preds[m][v], c)[0] for v in vids]
            pd_ = [bouts(preds[m][v], c)[1] for v in vids]
            hc, hd = np.array(human[c]["count"]), np.array(human[c]["dur"])
            rel_c = np.abs(np.array(pc) - hc) / np.maximum(hc, 1)
            rel_d = np.abs(np.array(pd_) - hd) / np.maximum(hd, 1)
            res[NAMES[c]] = {"count_r": float(pearsonr(pc, hc)[0]) if np.std(pc) > 0 else np.nan,
                             "dur_r": float(pearsonr(pd_, hd)[0]) if np.std(pd_) > 0 else np.nan,
                             "count_median_rel_err": float(np.median(rel_c)),
                             "dur_median_rel_err": float(np.median(rel_d)),
                             "pred_count": pc, "pred_dur": pd_}
        out["methods"][m] = res
    # rater-vs-other-raters bout r (each rater vs mean of other two)
    hr = {}
    for c in NAMES:
        rs = []
        for r_i in range(3):
            me, oth = [], []
            for v in vids:
                L = labs[v]
                seqs = []
                for r in RATERS:
                    s = L[f"vote_{r}"].to_numpy().copy()
                    s[(s < 0) | (s == ABSTAIN)] = 0
                    seqs.append(postprocess(s.astype(np.int8), pp["mode_filter_frames"], pp["min_bout_frames"]))
                me.append(bouts(seqs[r_i], c)[0])
                oth.append(np.mean([bouts(seqs[j], c)[0] for j in range(3) if j != r_i]))
            rs.append(pearsonr(me, oth)[0])
        hr[NAMES[c]] = {"count_r_each_rater_vs_others": [float(x) for x in rs], "mean": float(np.mean(rs))}
    out["human_rater_vs_others"] = hr
    json.dump(out, open(ROOT / "results/bouts.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))
    print(json.dumps({m: {c: round(out["methods"][m][c]["count_r"], 3) for c in out["methods"][m]} for m in out["methods"]}, indent=1))
    print(json.dumps(hr, indent=1))


if __name__ == "__main__":
    main()
