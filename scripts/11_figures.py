"""Figures (vector PDF) from results/*.json.

Style targets an IEEE two-column page: Times (the body font of IEEEtran) so the
figures read as part of the typeset page rather than as pasted-in plots, 7 pt text,
0.6 pt black rules, outward ticks, no gridlines and no drop shadows or fills that
carry no information.

Palette is the Okabe-Ito colorblind-safe set, the de-facto standard for scientific
figures: vermillion #D55E00 = threshold rules, blue #0072B2 = learned models,
bluish green #009E73 = RF restricted to the rules' inputs. Validated with the
dataviz palette checker (all six checks pass; worst adjacent CVD dE 8.6 tritan,
18.0 protan). Colour encodes method family and is identical in every figure;
the post-hoc wide grid is the same vermillion with a hatch, and the human ceiling
is always a neutral black dashed line.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "results"
FIG = ROOT / "paper/figs"
CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))

INK, INK2, MUTED = "#000000", "#3d3d3d", "#808080"
BLUE, ORANGE, AQUA = "#0072B2", "#D55E00", "#009E73"   # Okabe-Ito
METHODS = ["fixed_rules", "tuned_rules", "tuned_rules_wide", "dt", "rf", "hgb", "rf_rule"]
SHORT = {"fixed_rules": "Rules\nfixed", "tuned_rules": "Rules\ntuned", "tuned_rules_wide": "Rules\nwide\u2020",
         "dt": "DT", "rf": "RF", "hgb": "GBM", "rf_rule": "RF\nrule-in"}
COLOR = {"fixed_rules": ORANGE, "tuned_rules": ORANGE, "tuned_rules_wide": ORANGE, "dt": BLUE, "rf": BLUE,
         "hgb": BLUE, "rf_rule": AQUA}
HATCH = {"tuned_rules_wide": "/////"}
SERIF = ["Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman", "Times", "STIX Two Text", "DejaVu Serif"]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": SERIF,
                     "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold", "axes.linewidth": 0.6,
                     "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "xtick.direction": "out", "ytick.direction": "out",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.labelcolor": INK, "text.color": INK,
                     "legend.frameon": False, "legend.handletextpad": 0.5,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.5})


def style(ax):
    """Two-spine frame, no grid: the plain convention in IEEE and most journals."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, pad=2)


def per_video(m, k):
    d = json.load(open(R / f"lovo/{m}.json"))["per_video"]
    return np.array([d[v][k] for v in sorted(d)], float)


def fig2():
    lab = json.load(open(R / "labels_summary.json"))["pairwise_f1_mean"]
    panels = [("Supported_F1", "Supported rear", lab["Supported"]), ("Unsupported_F1", "Unsupported rear", lab["Unsupported"]),
              ("Grooming_F1", "Grooming", lab["Grooming"]), ("macro3", "Macro F1 (3 behaviors)", lab["macro3"])]
    ms = [m for m in METHODS if (R / f"lovo/{m}.json").exists()]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.08), sharey=True)
    for ax, (k, title, human) in zip(axes, panels):
        data = [per_video(m, k)[np.isfinite(per_video(m, k))] for m in ms]
        # every fold is plotted individually, so fliers are off: drawing them too
        # would show the same outlying video twice.
        bp = ax.boxplot(data, widths=0.58, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=INK2, lw=0.6),
                        capprops=dict(color=INK2, lw=0.6))
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(COLOR[m]); patch.set_alpha(0.45)
            patch.set_edgecolor(INK2); patch.set_linewidth(0.6)
            if m in HATCH:
                patch.set_hatch(HATCH[m])
        for i, (m, d) in enumerate(zip(ms, data)):   # one marker per held-out video
            x = np.random.default_rng(i).uniform(-0.15, 0.15, len(d)) + i + 1
            ax.plot(x, d, ls="none", marker="o", ms=1.6, mfc="none", mec=INK2, mew=0.35,
                    alpha=0.75, zorder=3)
        ax.axhline(human, color=INK2, lw=0.6, ls=(0, (2.5, 2)), zorder=2)
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([SHORT[m] for m in ms], fontsize=6.0)
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        ax.set_ylim(-0.03, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlim(0.4, len(ms) + 0.6)
        style(ax)
    axes[0].set_ylabel("Frame F1 (20 LOVO folds)")
    handles = [Patch(fc=ORANGE, ec=INK2, alpha=0.45, lw=0.6, label="Threshold rules"),
               Patch(fc=BLUE, ec=INK2, alpha=0.45, lw=0.6, label="Learned models"),
               Patch(fc=AQUA, ec=INK2, alpha=0.45, lw=0.6, label="RF, rule inputs only"),
               Line2D([], [], color=INK2, lw=0.6, ls=(0, (2.5, 2)), label="Human pairwise ceiling")]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               fontsize=6.5, columnspacing=1.6, handlelength=1.5)
    fig.tight_layout(w_pad=0.7, rect=(0, 0, 1, 0.932))
    fig.savefig(FIG / "fig2_lovo.pdf")
    plt.close(fig)


def fig3():
    s = json.load(open(R / "stockholm/results.json"))["analyses"]["primary"] if (R / "stockholm/results.json").exists() else {}
    ms = [m for m in METHODS if (R / f"lovo/{m}.json").exists()]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.98), gridspec_kw={"width_ratios": [2.1, 1.3, 1.1]})
    lab = json.load(open(R / "labels_summary.json"))["pairwise_f1_mean"]["macro3"]
    main = np.nanmean(per_video("rf", "macro3"))
    # (a) LOVO -> Stockholm, one connector per method
    ax = axes[0]
    for i, m in enumerate(ms):
        lv = np.nanmean(per_video(m, "macro3"))
        if m in s:
            ax.plot([i, i], [lv, s[m]["macro3"]], color=COLOR[m], lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(i, s[m]["macro3"], marker="o", ms=3.6, mfc="white", mec=COLOR[m], mew=0.9, zorder=3)
        ax.plot(i, lv, marker="o", ms=3.6, mfc=COLOR[m], mec=COLOR[m], mew=0.9, zorder=3)
    ax.axhline(lab, color=INK2, lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ax.annotate("human", (len(ms) - 0.45, lab), xytext=(0, 1.5), textcoords="offset points",
                fontsize=6.5, va="bottom", ha="right", color=INK)
    handles = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=INK2, mec=INK2, label="ETH (leave-one-video-out)"),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=INK2, mew=0.9,
                      label="Stockholm (external)")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", handletextpad=0.4,
              borderpad=0.1, labelspacing=0.3)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([SHORT[m] for m in ms], fontsize=6.5)
    ax.set_xlim(-0.6, len(ms) - 0.4)
    ax.set_ylabel("Macro F1 (3 behaviors)"); ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("(a) Laboratory shift", fontsize=8, loc="left", pad=3)
    style(ax)
    # (b) window ablation (RF); y is zoomed -- stated in the caption
    ax = axes[1]
    ks = CFG["evaluation"]["ablations"]["windows_single"]
    vals = []
    for k in ks:
        f = R / f"ablations/rf_window_{k}.json"
        vals.append(np.nanmean([v["macro3"] for v in json.load(open(f))["per_video"].values()]) if f.exists() else np.nan)
    secs = [k / 25 for k in ks]
    ax.axhline(lab, color=INK2, lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ax.annotate("human", (2.05, lab), xytext=(0, 1.5), textcoords="offset points",
                fontsize=6.5, va="bottom", ha="right", color=INK)
    ax.axhline(main, color=MUTED, lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    ax.annotate("all three windows", (2.05, main), xytext=(0, 1.5), textcoords="offset points",
                fontsize=6.5, va="bottom", ha="right", color=INK2)
    ax.plot(secs, vals, color=BLUE, lw=0.9, marker="o", ms=3.0, mfc="white", mec=BLUE, mew=0.9, zorder=3)
    ax.set_xlabel("Window half-width (s)"); ax.set_ylim(0.30, 0.88); ax.set_xticks(secs)
    ax.set_xticklabels([f"{x:g}" for x in secs], fontsize=6.5)
    ax.set_yticks(np.arange(0.3, 0.81, 0.1))
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(-0.12, 2.12)
    ax.set_title("(b) Temporal context (RF)", fontsize=8, loc="left", pad=3)
    style(ax)
    # (c) feature source (RF)
    ax = axes[2]
    src = [("rf_source_silhouette_only", "Silhouette\nonly (9)"), (None, "+ contour\nhead (14)"),
           ("rf_source_dlc_reference", "+ DLC\npose (14)\u2021")]
    vals = []
    for f, _ in src:
        if f is None:
            vals.append(main)
        else:
            p = R / f"ablations/{f}.json"
            vals.append(np.nanmean([v["macro3"] for v in json.load(open(p))["per_video"].values()]) if p.exists() else np.nan)
    bars = ax.bar(range(3), vals, facecolor=BLUE, edgecolor=INK2, lw=0.6, width=0.62, zorder=2)
    for b in bars:
        b.set_alpha(0.45)
    bars[2].set_facecolor("white"); bars[2].set_alpha(1.0); bars[2].set_hatch("....")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in src], fontsize=6.5)
    ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.62, 2.62)
    ax.set_title("(c) Feature source (RF)", fontsize=8, loc="left", pad=3)
    style(ax)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig3_shift_ablation.pdf")
    plt.close(fig)


def fig1(examples):
    """examples: list of (label, rectified BGR crop with overlay) + full frames."""
    raw_img, rect_img, crops = examples
    # Saved at exactly \textwidth and WITHOUT bbox_inches="tight": a tight box would
    # crop the canvas to ~5.6 in, and LaTeX would then scale it back up to 7.16 in,
    # printing this figure's type ~28% larger than Fig. 2 and Fig. 3.
    fig = plt.figure(figsize=(7.16, 1.56))
    gs = fig.add_gridspec(1, 8, width_ratios=[1.45, 1.05, 0.75, 0.75, 0.75, 0.75, 0.22, 1.95],
                          wspace=0.08, left=0.004, right=0.996, top=0.845, bottom=0.01)
    img_axes = []
    ax = fig.add_subplot(gs[0]); ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    img_axes.append((ax, "Raw frame"))
    ax = fig.add_subplot(gs[1]); ax.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    img_axes.append((ax, "Rectified, 10 px/cm"))
    for j, (name, img) in enumerate(crops):
        ax = fig.add_subplot(gs[2 + j]); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
        img_axes.append((ax, name))
    for a, _ in img_axes:      # hang every panel from a common top edge
        a.set_anchor("N")
    fig.canvas.draw()
    top = max(a.get_position().y1 for a, _ in img_axes)
    for a, t in img_axes:   # one baseline for every panel title
        p = a.get_position()
        fig.text((p.x0 + p.x1) / 2, top + 0.025, t, ha="center", va="bottom", fontsize=7.5, color=INK)
    # Stage boxes: square corners, drawn as real patches spanning the column so the
    # frame can never fall outside the saved bounding box (text bboxes could).
    ax = fig.add_subplot(gs[7]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [(0.845, "MOG2 silhouette + head proxy\n14 features / frame"),
             (0.500, "Windows ±5, 12, 25 frames\n(mean, SD, min, max): 182"),
             (0.155, "Threshold rules  |  DT / RF / GBM\nsame post-processing")]
    bh = 0.27
    for y, t in boxes:
        ax.add_patch(Rectangle((0.0, y - bh / 2), 1.0, bh, transform=ax.transAxes,
                               facecolor="white", edgecolor=INK2, lw=0.6, zorder=1, clip_on=False))
        ax.text(0.5, y, t, ha="center", va="center", fontsize=6.8, color=INK, zorder=2,
                linespacing=1.35, transform=ax.transAxes)
    for y0, y1 in ((0.845 - bh / 2, 0.500 + bh / 2), (0.500 - bh / 2, 0.155 + bh / 2)):
        ax.annotate("", (0.5, y1), (0.5, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.28",
                                    lw=0.6, color=INK2, shrinkA=0, shrinkB=0))
    fig.savefig(FIG / "fig1_pipeline.pdf", dpi=600)   # 600 dpi: the video frames are rasters
    plt.close(fig)


def make_fig1():
    """Pick one mid-bout frame per class from OFT_5 consensus labels; re-render with the tracker.

    The per-frame overlays are cached in data/interim/fig1; when they are already
    there the tracker is not re-run, so the figure can be restyled cheaply.
    """
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
    qc = ROOT / "data/interim/fig1"
    qc.mkdir(parents=True, exist_ok=True)
    if all((qc / f"fig_f{f:05d}.png").exists() for f in picks.values()):
        print("fig1: reusing cached overlays", flush=True)
    else:
        H, size = homography(np.array(m["corners"]), m["side"])
        track_video(ROOT / f"data/raw/eth_videos/{vid}.mp4", H, size, tuple(m["window"]), m["side"], CFG["tracker"],
                    qc_frames=set(picks.values()), qc_dir=qc, tag="fig")
    tr = pd.read_parquet(ROOT / f"data/interim/track/{vid}.parquet").set_index("frame")
    crops = []
    for name, f in picks.items():
        img = cv2.imread(str(qc / f"fig_f{f:05d}.png"))
        cx, cy = ((tr.loc[f, ["cx", "cy"]].to_numpy() + 8) * 10).astype(int)
        h = 100
        y0 = int(np.clip(cy - h, 0, img.shape[0] - 2 * h))
        x0 = int(np.clip(cx - h, 0, img.shape[1] - 2 * h))
        crop = img[y0:y0 + 2 * h, x0:x0 + 2 * h]
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
