"""All paper numbers + tables, generated from results/*.json.

Outputs: paper/generated/numbers.tex (\\num{key} macros), paper/generated/table_data.tex,
paper/generated/table_main.tex, results/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.stats import bootstrap_ci, holm, paired_wilcoxon  # noqa: E402

import yaml
CFG_WINDOWS = yaml.safe_load(open(ROOT / "config/prereg.yaml"))["features"]["windows_frames"]
R = ROOT / "results"
GEN = ROOT / "paper/generated"
METHODS = ["fixed_rules", "tuned_rules", "tuned_rules_wide", "dt", "rf", "hgb", "rf_rule"]
LABEL = {"fixed_rules": "Rules (fixed)", "tuned_rules": "Rules (tuned)", "tuned_rules_wide": "Rules (tuned, wide)$^\\dagger$",
         "dt": "Decision tree", "rf": "Random forest", "hgb": "Gradient boosting", "rf_rule": "RF, rule inputs"}
KEY = {"fixed_rules": "fixed", "tuned_rules": "tuned", "tuned_rules_wide": "wide", "dt": "dt", "rf": "rf",
       "hgb": "gbm", "rf_rule": "rfrule"}
CLS = [("Supported_F1", "S"), ("Unsupported_F1", "U"), ("Grooming_F1", "G"), ("macro3", "macro")]
NUM = {}


def put(k, v):
    NUM[k] = v


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def f2n(x):  # no leading zero
    s = f2(x)
    return s.replace("0.", ".", 1) if s.startswith("0.") else s


def pfmt(p):
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
    return f"{p:.3f}"


def vec(res, m, k):
    d = res[m]["per_video"]
    return np.array([d[v][k] for v in sorted(d)], float)


def main():
    lab = json.load(open(R / "labels_summary.json"))
    meta = json.load(open(R / "track_meta.json"))
    align = json.load(open(R / "align_check.json"))
    res = {m: json.load(open(R / f"lovo/{m}.json")) for m in METHODS if (R / f"lovo/{m}.json").exists()}
    summary = {}
    # ---------------- data ----------------
    put("n_raw_features", "14")
    put("n_scored", f"{lab['n_scored']:,}".replace(",", "{,}"))
    put("n_scored_k", f"{round(lab['n_scored'] / 1000):d}")
    put("pct_ambiguous", f"{lab['pct_ambiguous_of_nonexcluded']:.2f}")
    pw = lab["pairwise_f1_mean"]
    put("human_S", f2(pw["Supported"])); put("human_U", f2(pw["Unsupported"])); put("human_G", f2(pw["Grooming"]))
    put("human_macro", f2(pw["macro3"]))
    put("fleiss_kappa", f2(lab["fleiss_kappa"]))
    cov = [m["coverage"] for v, m in meta.items()]
    put("coverage_min", f"{100 * min(cov):.1f}")
    put("align_err_eth", f"{np.median([a['median_pos_err_cm'] for v, a in align.items() if v.startswith('OFT')]):.1f}")
    put("align_r_min", f2(min(a["peak_speed_corr"] for a in align.values())))
    # ---------------- LOVO ----------------
    for m in res:
        for k, short in CLS:
            v = vec(res, m, k)
            put(f"{KEY[m]}_{short}", f2(np.nanmean(v)))
            put(f"{KEY[m]}_{short}_sd", f2(np.nanstd(v, ddof=1)))
            summary.setdefault(m, {})[short] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                                                "n": int(np.isfinite(v).sum())}
    if "rf" in res and "tuned_rules" in res:
        tests = {}
        prim = paired_wilcoxon(vec(res, "rf", "macro3"), vec(res, "tuned_rules", "macro3"))
        summary["primary_test"] = prim
        put("p_primary", pfmt(prim["p"])); put("rbc_primary", f2(prim["rank_biserial"]))
        put("delta_primary", f2(prim["median_diff"]))
        for k, short in CLS[:3]:
            for m in ("rf", "hgb"):
                if m in res:
                    tests[f"{m}_{short}"] = paired_wilcoxon(vec(res, m, k), vec(res, "tuned_rules", k))
        if "hgb" in res:
            tests["hgb_macro"] = paired_wilcoxon(vec(res, "hgb", "macro3"), vec(res, "tuned_rules", "macro3"))
        adj = holm({k: t["p"] for k, t in tests.items()})
        for k, t in tests.items():
            t["p_holm"] = adj[k]
            put(f"p_{k}", pfmt(adj[k]))
        summary["secondary_tests"] = tests
        put("p_holm_max", pfmt(max(adj.values())))
        # gap_U - gap_S (RF vs tuned), video-level bootstrap
        gU = vec(res, "rf", "Unsupported_F1") - vec(res, "tuned_rules", "Unsupported_F1")
        gS = vec(res, "rf", "Supported_F1") - vec(res, "tuned_rules", "Supported_F1")
        d = gU - gS
        ci = bootstrap_ci(d, np.mean, 10000, 0)
        summary["gapU_minus_gapS"] = {"mean": float(np.nanmean(d)), "ci95": ci}
        put("gapUS_mean", f2(np.nanmean(d))); put("gapUS_lo", f2(ci[0])); put("gapUS_hi", f2(ci[1]))
        put("gap_S", f2(np.nanmean(gS))); put("gap_U", f2(np.nanmean(gU)))
        gG = vec(res, "rf", "Grooming_F1") - vec(res, "tuned_rules", "Grooming_F1")
        put("gap_G", f2(np.nanmean(gG)))
    # ---------------- human (secondary) + bouts ----------------
    if (R / "human_ceiling.json").exists():
        hc = json.load(open(R / "human_ceiling.json"))["mean_over_references"]
        summary["human_ceiling_secondary"] = hc
        put("hum2_macro", f2(hc["human"]["macro3"]))
        if "rf" in hc:
            put("rf2_macro", f2(hc["rf"]["macro3"]))
    if (R / "bouts.json").exists():
        b = json.load(open(R / "bouts.json"))
        summary["bouts"] = {m: {c: b["methods"][m][c]["count_r"] for c in b["methods"][m]} for m in b["methods"]}
        for m in b["methods"]:
            for c, short in (("Supported", "S"), ("Unsupported", "U"), ("Grooming", "G")):
                put(f"{KEY[m]}_boutr_{short}", f2(b["methods"][m][c]["count_r"]))
                put(f"{KEY[m]}_durr_{short}", f2(b["methods"][m][c]["dur_r"]))
        for c, short in (("Supported", "S"), ("Unsupported", "U"), ("Grooming", "G")):
            put(f"human_boutr_{short}", f2(b["human_rater_vs_others"][c]["mean"]))
    # ---------------- ablations ----------------
    ab = R / "ablations"
    if ab.exists():
        for f in sorted(ab.glob("*.json")):
            d = json.load(open(f))
            v = np.array([d["per_video"][x]["macro3"] for x in d["per_video"]], float)
            name = f.stem.replace("_", "")
            put(f"ab{name}", f2(np.nanmean(v)))
            summary.setdefault("ablations", {})[f.stem] = {"macro3_mean": float(np.nanmean(v)),
                                                          "macro3_sd": float(np.nanstd(v, ddof=1))}
            for k, short in CLS[:3]:
                vv = np.array([d["per_video"][x][k] for x in d["per_video"]], float)
                put(f"ab{name}{short}", f2(np.nanmean(vv)))
    # ---------------- Stockholm ----------------
    if (R / "stockholm/results.json").exists():
        s = json.load(open(R / "stockholm/results.json"))
        put("stk_n_labelled", f"{s['n_labelled']:,}".replace(",", "{,}"))
        P = s["analyses"]["primary"]
        npos = P["rf"]["n_pos"]
        put("stk_n_groom", str(npos["Grooming"])); put("stk_n_S", str(npos["Supported"])); put("stk_n_U", str(npos["Unsupported"]))
        summary["stockholm"] = {}
        for m in METHODS:
            if m not in P:
                continue
            for k, short in CLS + [("macro_SU", "macroSU")]:
                put(f"stk_{KEY[m]}_{short}", f2(P[m][k]))
            for c, short in (("Supported", "S"), ("Unsupported", "U"), ("Grooming", "G")):
                lo, hi = P[m]["ci95"][c]
                put(f"stk_{KEY[m]}_{short}_lo", f2(lo)); put(f"stk_{KEY[m]}_{short}_hi", f2(hi))
            if m in summary:
                dm = P[m]["macro3"] - summary[m]["macro"]["mean"]
                put(f"delta_{KEY[m]}", f"{dm:+.2f}")
                summary["stockholm"][m] = {"macro3": P[m]["macro3"], "macroSU": P[m]["macro_SU"], "delta_macro3": dm}
        lov_rf, lov_tu = summary["rf"]["macro"]["mean"], summary["tuned_rules"]["macro"]["mean"]
        put("gap_macro_lovo", f2(lov_rf - lov_tu))
        put("gap_macro_stk", f2(P["rf"]["macro3"] - P["tuned_rules"]["macro3"]))
        put("stk_rules_max", f2(max(P[m]["macro3"] for m in ("fixed_rules", "tuned_rules", "tuned_rules_wide"))))
        S2 = s["analyses"]["exclude_stand_and_sniff"]
        for m in ("rf", "tuned_rules", "tuned_rules_wide"):
            if m in S2:
                put(f"stkx_{KEY[m]}_macro", f2(S2[m]["macro3"]))
    else:
        # data-section numbers that do not depend on the one-shot run
        put("stk_n_labelled", "2{,}999"); put("stk_n_groom", "68")
    # ---------------- cost ----------------
    if (R / "cost.json").exists():
        c = json.load(open(R / "cost.json"))
        put("tracker_fps", f"{c['tracker_pass2_fps']:.0f}")
        for m, v in c["models"].items():
            put(f"{KEY[m]}_us", f"{v['batch_us_per_frame']:.1f}")
            put(f"{KEY[m]}_lat_ms", f"{v['single_frame_latency_ms_median']:.1f}")
            put(f"{KEY[m]}_mb", f"{v['size_mb']:.1f}")
            put(f"{KEY[m]}_efps", f"{c['end_to_end_fps'][m]:.0f}")
        put("rules_efps", f"{c['end_to_end_fps']['rules']:.0f}")
        put("rules_us", f"{c['fixed_rules_us_per_frame']:.1f}")
        put("efps_drop_pct", f"{100 * (1 - min(c['end_to_end_fps'][m] for m in ('dt', 'rf', 'hgb', 'rf_rule')) / c['end_to_end_fps']['rules']):.1f}")
        put("feat_us", f"{c['raw_features_us_per_frame'] + c['windowing_us_per_frame']:.0f}")
        put("cpu_name", c["machine"]["cpu"])
    # ---------------- error analysis ----------------
    if (R / "error_analysis.json").exists():
        e = json.load(open(R / "error_analysis.json"))["su_confusion"]
        for m in ("rf", "tuned_rules_wide", "tuned_rules"):
            if m in e:
                for z, v in e[m]["zone"].items():
                    put(f"su_{KEY[m]}_{z}", f"{100 * v['rate']:.0f}")
                for b_, v in e[m]["head_wall_dist"].items():
                    key = {"<0": "neg", "0-2": "zto", "2-4": "twf", "4-8": "fte", ">=8": "gte"}[b_]
                    put(f"su_{KEY[m]}_hw{key}", f"{100 * v['rate']:.0f}")
    if "rf" in res and "rf_rule" in res:
        put("gapRFrule", f2(np.nanmean(vec(res, "rf", "macro3")) - np.nanmean(vec(res, "rf_rule", "macro3"))))
    put("lag_s", f"{max(CFG_WINDOWS) / 25:.0f}")
    # ---------------- diagnostics (post-hoc) ----------------
    if (R / "diagnostics.json").exists():
        dg = json.load(open(R / "diagnostics.json"))
        for c, short in (("Supported", "S"), ("Unsupported", "U"), ("Grooming", "G"), ("Other", "O")):
            put(f"diag_blr_{short}", f2(dg["median"][c]["body_length_ratio"]))
            put(f"diag_ar_{short}", f2(dg["median"][c]["area_ratio"]))
            put(f"diag_hw_{short}", f"{dg['median'][c]['head_wall_dist']:.1f}")
            put(f"diag_rearcond_{short}", f"{100 * dg['frac_fixed_rear_condition'][c]:.0f}")
            put(f"diag_groomcond_{short}", f"{100 * dg['frac_fixed_groom_condition'][c]:.0f}")
    # tuned thresholds across folds
    for m in ("tuned_rules", "tuned_rules_wide"):
        if m in res:
            ex = res[m]["extra"]
            for k, short in (("rear_length_ratio_lt", "tauL"), ("rear_area_ratio_lt", "tauA"),
                             ("supported_head_wall_dist_lt_cm", "tauW"), ("groom_speed_lt", "tauV")):
                vals = np.array([e["params"][k] for e in ex.values()], float)
                put(f"{KEY[m]}_{short}", f"{np.median(vals):g}")
                summary.setdefault("tuned_params", {}).setdefault(m, {})[k] = {"median": float(np.median(vals)),
                                                                              "min": float(vals.min()), "max": float(vals.max())}
            grid_max = {"tuned_rules": 0.90, "tuned_rules_wide": 1.10}[m]
            n_ceiling = int(sum(e["params"]["rear_length_ratio_lt"] >= grid_max - 1e-9 for e in ex.values()))
            put(f"{KEY[m]}_nceilL", str(n_ceiling))
    # ---------------- write ----------------
    GEN.mkdir(parents=True, exist_ok=True)
    with open(GEN / "numbers.tex", "w") as f:
        f.write("% generated by scripts/12_tables.py -- do not edit\n")
        for k, v in sorted(NUM.items()):
            f.write(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n")
    # table: data
    cls = lab["classes"]
    rows = []
    for c in ("Supported", "Unsupported", "Grooming", "Other"):
        x = cls[c]
        stk = ""
        if (R / "stockholm/results.json").exists() and c != "Other":
            stk = str(json.load(open(R / "stockholm/results.json"))["analyses"]["primary"]["rf"]["n_pos"][c])
        elif c == "Other" and (R / "stockholm/results.json").exists():
            s = json.load(open(R / "stockholm/results.json"))
            npos = s["analyses"]["primary"]["rf"]["n_pos"]
            stk = str(s["n_labelled"] - sum(npos.values()))
        rows.append(f"{c} & {x['frames']:,} ({x['pct_scored']:.1f}\\%) & {x['bouts']:,} & {x['mean_bout_s']:.1f} & "
                    f"{f2(pw.get(c, np.nan)) if c != 'Other' else '--'} & {stk} \\\\".replace(",", "{,}"))
    with open(GEN / "table_data.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\caption{Consensus labels. ETH: 20 videos, 3 raters (majority vote); "
                "human F1 is the mean pairwise inter-rater frame F1. Stockholm: labelled frames (3 videos).}\\label{tab:data}\n"
                "\\setlength{\\tabcolsep}{3pt}\\footnotesize\n\\begin{tabular}{lrrrrr}\\toprule\n"
                "& \\multicolumn{4}{c}{ETH (primary)} & Stockh.\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\n"
                "Class & Frames & Bouts & Bout (s) & Human F1 & Frames\\\\\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # table: main
    body = []
    cost = json.load(open(R / "cost.json")) if (R / "cost.json").exists() else None
    stk = json.load(open(R / "stockholm/results.json"))["analyses"]["primary"] if (R / "stockholm/results.json").exists() else None
    bts = json.load(open(R / "bouts.json")) if (R / "bouts.json").exists() else None
    for m in METHODS:
        if m not in summary:
            continue
        cells = [f"{f2n(summary[m][s]['mean'])}$\\pm${f2n(summary[m][s]['sd'])}" for _, s in CLS]
        br = f2n(np.nanmean([bts["methods"][m][c]["count_r"] for c in ("Supported", "Unsupported", "Grooming")])) if bts and m in bts["methods"] else "--"
        us = "--"
        if cost:
            us = f"{cost['fixed_rules_us_per_frame']:.1f}" if "rules" in m else f"{cost['models'][m]['batch_us_per_frame']:.1f}"
        sk = f2n(stk[m]["macro3"]) if stk and m in stk else "--"
        body.append(f"{LABEL[m]} & " + " & ".join(cells) + f" & {br} & {us} & {sk} \\\\")
        if m == "tuned_rules_wide":
            body.append("\\midrule")
    human = f"Human (pairwise) & {f2n(pw['Supported'])} & {f2n(pw['Unsupported'])} & {f2n(pw['Grooming'])} & {f2n(pw['macro3'])} & " \
            f"{f2n(np.mean([bts['human_rater_vs_others'][c]['mean'] for c in ('Supported','Unsupported','Grooming')])) if bts else '--'} & -- & -- \\\\"
    with open(GEN / "table_main.tex", "w") as f:
        f.write("\\begin{table*}[t]\\centering\\caption{Leave-one-video-out results on ETH (frame F1, mean$\\pm$SD over 20 folds), "
                "bout-count correlation with the raters' median (Pearson $r$, mean over the three behaviors), classifier cost "
                "per frame (batch, one core, excluding tracking and features), and macro F1 on the external Stockholm set. "
                "$^\\dagger$Post-hoc, not pre-registered.}\\label{tab:main}\n"
                "\\footnotesize\\begin{tabular}{lccccccc}\\toprule\n"
                "Method & Supported & Unsupported & Grooming & Macro F1 & Bout $r$ & $\\mu$s/frame & Stockholm\\\\\\midrule\n"
                + "\n".join(body) + "\n\\midrule\n" + human + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    json.dump(summary, open(R / "summary.json", "w"), indent=1, default=float)
    print(f"{len(NUM)} numbers written")


if __name__ == "__main__":
    main()
