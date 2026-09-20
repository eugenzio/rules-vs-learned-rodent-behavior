"""Figures (vector PDF) from results/*.json. Style follows the author's earlier plots
(6.5 pt sans, 0.5 pt spines, y-grid only) and a validated categorical palette
(blue #2a78d6, orange #eb6834, aqua #1baf7a; all-pairs CVD dE 9.2).
Color encodes method family, identical in every figure: rules = orange, learned = blue,
RF on rule inputs = aqua; post-hoc wide-grid rules = orange + hatch. Human ceiling = neutral dashed.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "results"
FIG = ROOT / "paper/figs"
CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METHODS = ["fixed_rules", "tuned_rules", "tuned_rules_wide", "dt", "rf", "hgb", "rf_rule"]
SHORT = {"fixed_rules": "Rules\nfixed", "tuned_rules": "Rules\ntuned", "tuned_rules_wide": "Rules\nwide$^\\dagger$",
         "dt": "DT", "rf": "RF", "hgb": "GBM", "rf_rule": "RF\nrule-in"}
COLOR = {"fixed_rules": ORANGE, "tuned_rules": ORANGE, "tuned_rules_wide": ORANGE, "dt": BLUE, "rf": BLUE,
         "hgb": BLUE, "rf_rule": AQUA}
HATCH = {"tuned_rules_wide": "////"}

plt.rcParams.update({"font.size": 6.5, "font.family": "sans-serif", "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.6})


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def per_video(m, k):
    d = json.load(open(R / f"lovo/{m}.json"))["per_video"]
    return np.array([d[v][k] for v in sorted(d)], float)


def fig2():
    lab = json.load(open(R / "labels_summary.json"))["pairwise_f1_mean"]
    panels = [("Supported_F1", "Supported rear", lab["Supported"]), ("Unsupported_F1", "Unsupported rear", lab["Unsupported"]),
              ("Grooming_F1", "Grooming", lab["Grooming"]), ("macro3", "Macro F1 (3 behaviors)", lab["macro3"])]
    ms = [m for m in METHODS if (R / f"lovo/{m}.json").exists()]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=True)
    for ax, (k, title, human) in zip(axes, panels):
        data = [per_video(m, k)[np.isfinite(per_video(m, k))] for m in ms]
        bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=0.8), whiskerprops=dict(color=INK2, lw=0.5),
                        capprops=dict(color=INK2, lw=0.5),
                        flierprops=dict(marker="o", ms=2.2, mfc="none", mec=INK2, mew=0.4))
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(COLOR[m]); patch.set_edgecolor("white"); patch.set_linewidth(0.8)
            patch.set_alpha(0.9)
            if m in HATCH:
                patch.set_hatch(HATCH[m])
        for i, (m, d) in enumerate(zip(ms, data)):   # individual folds
            x = np.random.default_rng(i).uniform(-0.18, 0.18, len(d)) + i + 1
            ax.scatter(x, d, s=1.3, color=INK, alpha=0.35, lw=0, zorder=3)
        ax.axhline(human, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.text(0.6, human, "human", color=INK2, fontsize=5.5, va="bottom", ha="left")
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([SHORT[m] for m in ms], fontsize=5.2)
        ax.set_title(title, fontsize=6.5, color=INK, pad=3)
        ax.set_ylim(-0.02, 1.0)
        style(ax)
    axes[0].set_ylabel("Frame F1 (20 LOVO folds)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig2_lovo.pdf")
    plt.close(fig)


def fig3():
    s = json.load(open(R / "stockholm/results.json"))["analyses"]["primary"] if (R / "stockholm/results.json").exists() else {}
    ms = [m for m in METHODS if (R / f"lovo/{m}.json").exists()]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75), gridspec_kw={"width_ratios": [2.1, 1.3, 1.1]})
    # (a) dumbbell LOVO -> Stockholm
    ax = axes[0]
    for i, m in enumerate(ms):
        lv = np.nanmean(per_video(m, "macro3"))
        ax.plot([i, i], [lv, s[m]["macro3"]] if m in s else [lv, lv], color=COLOR[m], lw=1.2, alpha=0.6)
        ax.scatter(i, lv, s=14, color=COLOR[m], zorder=3, edgecolor="white", lw=0.5)
        if m in s:
            ax.scatter(i, s[m]["macro3"], s=14, facecolor="white", edgecolor=COLOR[m], lw=1.0, zorder=3)
    ax.scatter([], [], s=14, color=INK2, label="ETH, leave-one-video-out")
    ax.scatter([], [], s=14, facecolor="white", edgecolor=INK2, lw=1.0, label="Stockholm (external)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", handletextpad=0.2)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([SHORT[m] for m in ms], fontsize=5.2)
    ax.set_ylabel("Macro F1 (3 behaviors)"); ax.set_ylim(0, 1.0)
    ax.set_title("(a) Lab shift", fontsize=6.5, loc="left")
    style(ax)
    # (b) window ablation (RF)
    ax = axes[1]
    ks = CFG["evaluation"]["ablations"]["windows_single"]
    vals = []
    for k in ks:
        f = R / f"ablations/rf_window_{k}.json"
        vals.append(np.nanmean([v["macro3"] for v in json.load(open(f))["per_video"].values()]) if f.exists() else np.nan)
    secs = [k / 25 for k in ks]
    ax.plot(secs, vals, color=BLUE, lw=1.2, marker="o", ms=3.2, mec="white", mew=0.5)
    main = np.nanmean(per_video("rf", "macro3"))
    ax.axhline(main, color=BLUE, lw=0.8, ls=(0, (1, 1.6)))
    ax.text(2.0, main + 0.008, "all three windows", color=BLUE, fontsize=5.5, va="bottom", ha="right")
    lab = json.load(open(R / "labels_summary.json"))["pairwise_f1_mean"]["macro3"]
    ax.axhline(lab, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(2.0, lab, "human", color=INK2, fontsize=5.5, va="bottom", ha="right")
    ax.set_xlabel("Window half-width (s)"); ax.set_ylim(0.3, 0.85); ax.set_xticks(secs)
    ax.set_xticklabels([f"{x:g}" for x in secs])
    ax.set_title("(b) Temporal context (RF)", fontsize=6.5, loc="left")
    style(ax)
    # (c) feature source (RF)
    ax = axes[2]
    src = [("rf_source_silhouette_only", "Silhouette\nonly (9)"), (None, "+ contour\nhead (14)"),
           ("rf_source_dlc_reference", "+ DLC\npose (14)$^\\ddagger$")]
    vals = []
    for f, _ in src:
        if f is None:
            vals.append(main)
        else:
            p = R / f"ablations/{f}.json"
            vals.append(np.nanmean([v["macro3"] for v in json.load(open(p))["per_video"].values()]) if p.exists() else np.nan)
    bars = ax.bar(range(3), vals, color=[BLUE, BLUE, "#ffffff"], edgecolor=[BLUE, BLUE, BLUE], lw=0.8, width=0.6)
    bars[2].set_hatch("....")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.5, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in src], fontsize=5.2)
    ax.set_ylim(0, 0.9)
    ax.set_title("(c) Feature source (RF)", fontsize=6.5, loc="left")
    style(ax)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig3_shift_ablation.pdf")
    plt.close(fig)


def fig1(examples):
    """examples: list of (label, rectified BGR crop with overlay) + full frames."""
    raw_img, rect_img, crops = examples
    fig = plt.figure(figsize=(7.16, 1.55))
    gs = fig.add_gridspec(1, 8, width_ratios=[1.45, 1.05, 0.75, 0.75, 0.75, 0.75, 0.15, 1.6], wspace=0.08)
    img_axes = []
    ax = fig.add_subplot(gs[0]); ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    img_axes.append((ax, "Raw frame"))
    ax = fig.add_subplot(gs[1]); ax.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    img_axes.append((ax, "Rectified (10 px/cm)"))
    for j, (name, img) in enumerate(crops):
        ax = fig.add_subplot(gs[2 + j]); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
        img_axes.append((ax, name))
    fig.canvas.draw()
    top = max(a.get_position().y1 for a, _ in img_axes)
    for a, t in img_axes:   # one baseline for every panel title
        p = a.get_position()
        fig.text((p.x0 + p.x1) / 2, top + 0.03, t, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax = fig.add_subplot(gs[7]); ax.axis("off")
    boxes = [(0.86, "MOG2 silhouette + head proxy\n14 features / frame"),
             (0.52, "Windows $\\pm$0.2, 0.5, 1 s\n(mean, SD, min, max): 182"),
             (0.14, "Threshold rules  |  DT / RF / GBM\nsame post-processing")]
    for y, t in boxes:
        ax.text(0.5, y, t, ha="center", va="center", fontsize=5.8, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f4f3ef", ec=GRID, lw=0.6), transform=ax.transAxes)
    for y0, y1 in ((0.73, 0.64), (0.40, 0.29)):
        ax.annotate("", (0.5, y1), (0.5, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2))
    fig.savefig(FIG / "fig1_pipeline.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def make_fig1():
    """Pick one mid-bout frame per class from OFT_5 consensus labels; re-render with the tracker."""
    from rr.calib import homography
    from rr.track import track_video
    meta = json.load(open(R / "track_meta.json"))
    vid = "OFT_5"
    m = meta[vid]
    lab = pd.read_parquet(ROOT / f"data/interim/labels/{vid}.parquet")
    picks = {}
    for code, name in ((1, "Supported"), (2, "Unsupported"), (3, "Grooming"), (0, "Other")):
        g = (lab["gt"] == code).to_numpy().astype(int)
        # longest run
        d = np.diff(np.r_[0, g, 0]); s_, e_ = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        j = np.argmax(e_ - s_)
        picks[name] = int(lab["frame"].iloc[(s_[j] + e_[j]) // 2])
    H, size = homography(np.array(m["corners"]), m["side"])
    qc = ROOT / "data/interim/fig1"
    qc.mkdir(parents=True, exist_ok=True)
    track_video(ROOT / f"data/raw/eth_videos/{vid}.mp4", H, size, tuple(m["window"]), m["side"], CFG["tracker"],
                qc_frames=set(picks.values()), qc_dir=qc, tag="fig")
    tr = pd.read_parquet(ROOT / f"data/interim/track/{vid}.parquet").set_index("frame")
    crops = []
    for name, f in picks.items():
        img = cv2.imread(str(qc / f"fig_f{f:05d}.png"))
        cx, cy = ((tr.loc[f, ["cx", "cy"]].to_numpy() + 8) * 10).astype(int)
        h = 75
        crop = img[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
        crops.append((name, crop))
    cap = cv2.VideoCapture(str(ROOT / f"data/raw/eth_videos/{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, picks["Supported"])
    _, raw = cap.read()
    rect = cv2.imread(str(qc / f"fig_f{picks['Supported']:05d}.png"))
    json.dump(picks, open(R / "fig1_frames.json", "w"))
    fig1((raw, rect, crops))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["fig1", "fig2", "fig3"]
    if "fig1" in which:
        make_fig1()
    if "fig2" in which:
        fig2()
    if "fig3" in which:
        fig3()
    print("figures:", which)
