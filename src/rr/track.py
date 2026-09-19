"""CPU-only silhouette tracker on a rectified metric canvas.

Segmentation constants and the head/tail disambiguation (`HeadTail`) are copied from the
author's own MIT-licensed tracker (Copyright (c) 2025 Eugene Cha; tracker.py,
`RodentTracker._determine_head_tail`). Two pre-registered changes:
  1. frames are warped to a 10 px/cm top-down canvas before segmentation;
  2. MOG2 is pre-trained on every 50th in-trial frame, then applied with learningRate=0,
     so a still animal is not absorbed into the background.
Output: one row per frame with raw per-frame geometry (cm on the arena frame).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from .calib import canvas_to_cm, signed_wall_dist


class HeadTail:
    """Movement-direction head/tail choice (copied logic; thresholds in canvas px)."""

    def __init__(self, min_move_px: float = 2.0, history: int = 5):
        self.min_move = min_move_px
        self.history_size = history
        self.hist: list[tuple[float, float]] = []
        self.prev_c = (np.nan, np.nan)
        self.prev_head = (np.nan, np.nan)

    def choose(self, e1, e2, c):
        if not np.isnan(self.prev_c[0]):
            dx, dy = c[0] - self.prev_c[0], c[1] - self.prev_c[1]
            if np.hypot(dx, dy) > self.min_move:
                self.hist.append((dx, dy))
                if len(self.hist) > self.history_size:
                    self.hist.pop(0)
        self.prev_c = c
        if len(self.hist) >= 2:
            adx = np.mean([m[0] for m in self.hist])
            ady = np.mean([m[1] for m in self.hist])
            d1 = (e1[0] - c[0]) * adx + (e1[1] - c[1]) * ady
            d2 = (e2[0] - c[0]) * adx + (e2[1] - c[1]) * ady
            head = e1 if d1 > d2 else e2
        elif not np.isnan(self.prev_head[0]):
            head = e1 if np.hypot(e1[0] - self.prev_head[0], e1[1] - self.prev_head[1]) < \
                np.hypot(e2[0] - self.prev_head[0], e2[1] - self.prev_head[1]) else e2
        else:
            head = e1
        self.prev_head = head
        return head

    def reset_centroid(self):
        self.prev_c = (np.nan, np.nan)


COLS = ["frame", "found", "cx", "cy", "area", "major", "minor", "angle", "solidity",
        "head_x", "head_y", "tip_dist", "d_wall_min", "seg_quality", "me_raw",
        "bx", "by", "bw", "bh"]


def _segment(bgs, blurred, cfg, lr):
    fg = bgs.apply(blurred, learningRate=lr)
    _, fg = cv2.threshold(fg, cfg["shadow_threshold"], 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"], cfg["morph_kernel"]))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=cfg["morph_iterations"])


def track_video(video_path, H, canvas_size, window, side_cm, cfg, ppcm=10.0, margin_cm=8.0,
                qc_frames=(), qc_dir=None, tag=""):
    """window: (f0, f1) frames to track (background pre-training uses the same window)."""
    cv2.setNumThreads(1)
    f0, f1 = window
    size = (canvas_size, canvas_size)
    ksz = (cfg["blur_ksize"], cfg["blur_ksize"])
    bgs = cv2.createBackgroundSubtractorMOG2(history=cfg["mog2_history"],
                                             varThreshold=cfg["mog2_var_threshold"],
                                             detectShadows=cfg["mog2_detect_shadows"])
    # ---- pass 1: background pre-training on every Nth in-trial frame ----
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = cfg["background_pretrain_every_n"]
    i, n_bg = 0, 0
    while True:
        ok = cap.grab()
        if not ok or i >= f1:
            break
        if i >= f0 and (i - f0) % step == 0:
            _, fr = cap.retrieve()
            w = cv2.warpPerspective(fr, H, size)
            bgs.apply(cv2.GaussianBlur(w, ksz, 0), learningRate=-1)
            n_bg += 1
        i += 1
    cap.release()
    bg_img = bgs.getBackgroundImage()
    # ---- pass 2: tracking with frozen background ----
    cap = cv2.VideoCapture(str(video_path))
    ht = HeadTail(min_move_px=cfg["head_min_move_cm"] * ppcm, history=cfg["head_history"])
    rows, prev_gray, i = [], None, 0
    t0 = time.perf_counter()
    n_proc = 0
    while True:
        ok = cap.grab()
        if not ok or i >= f1:
            break
        if i < f0:
            i += 1
            continue
        _, fr = cap.retrieve()
        w = cv2.warpPerspective(fr, H, size)
        gray = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        mask = _segment(bgs, cv2.GaussianBlur(w, ksz, 0), cfg, cfg["tracking_learning_rate"])
        rows.append(_measure(i, mask, gray, prev_gray, ht, cfg, side_cm, ppcm, margin_cm))
        if i in qc_frames and qc_dir is not None:
            _qc_overlay(w, mask, rows[-1], qc_dir / f"{tag}_f{i:05d}.png", ppcm, margin_cm)
        prev_gray = gray
        n_proc += 1
        i += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    df = pd.DataFrame(rows, columns=COLS)
    meta = {"n_total_frames_cv2": n_total, "n_tracked": n_proc, "n_background_frames": n_bg,
            "pass2_seconds": elapsed, "pass2_fps": n_proc / elapsed if elapsed else np.nan,
            "coverage": float(df["found"].mean()) if len(df) else 0.0}
    return df, bg_img, meta


def _measure(i, mask, gray, prev_gray, ht, cfg, side_cm, ppcm, margin_cm):
    nan = np.nan
    out = [i, False] + [nan] * (len(COLS) - 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in cnts if cv2.contourArea(c) >= cfg["min_contour_area_px"]]
    if not valid:
        ht.reset_centroid()
        return out
    c = max(valid, key=cv2.contourArea)
    area_px = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] == 0 or len(c) < 5:
        ht.reset_centroid()
        return out
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    (ex, ey), (d1, d2), ang = cv2.fitEllipse(c)
    # OpenCV RotatedRect: axis d1 lies along `ang`, d2 along ang+90. Use the true major axis
    # (the original code assumed d2 is always the major axis).
    if d1 >= d2:
        major, minor, ax_deg = d1, d2, ang
    else:
        major, minor, ax_deg = d2, d1, ang - 90.0
    a = np.radians(ax_deg)
    dx, dy = (major / 2) * np.cos(a), (major / 2) * np.sin(a)
    e1, e2 = (cx + dx, cy + dy), (cx - dx, cy - dy)
    head_end = ht.choose(e1, e2, (cx, cy))
    # head point: farthest contour point on the head side of the centroid
    pts = c[:, 0, :].astype(float)
    hv = np.array([head_end[0] - cx, head_end[1] - cy])
    nv = np.linalg.norm(hv)
    rel = pts - [cx, cy]
    dist = np.hypot(rel[:, 0], rel[:, 1])
    if nv > 0:
        side = rel @ (hv / nv) > 0
        j = np.argmax(np.where(side, dist, -1))
    else:
        j = np.argmax(dist)
    head = pts[j]
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    all_area = float(sum(cv2.contourArea(k) for k in cnts))
    bx, by, bw, bh = cv2.boundingRect(c)
    me = nan
    if prev_gray is not None:
        me = float(cv2.absdiff(gray[by:by + bh, bx:bx + bw], prev_gray[by:by + bh, bx:bx + bw]).mean())
    pts_cm = canvas_to_cm(pts, ppcm, margin_cm)
    c_cm = canvas_to_cm([[cx, cy]], ppcm, margin_cm)[0]
    h_cm = canvas_to_cm([head], ppcm, margin_cm)[0]
    return [i, True, c_cm[0], c_cm[1], area_px / ppcm ** 2, major / ppcm, minor / ppcm, ax_deg % 180.0,
            area_px / ha if ha > 0 else nan, h_cm[0], h_cm[1], dist[j] / ppcm,
            float(signed_wall_dist(pts_cm, side_cm).min()),
            area_px / all_area if all_area > 0 else nan,
            me, bx, by, bw, bh]


def _qc_overlay(w, mask, row, path, ppcm, margin_cm):
    img = w.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    if row[1]:
        c = ((np.array(row[2:4]) + margin_cm) * ppcm).astype(int)
        h = ((np.array(row[9:11]) + margin_cm) * ppcm).astype(int)
        cv2.circle(img, tuple(c), 4, (255, 0, 0), -1)
        cv2.circle(img, tuple(h), 4, (0, 0, 255), -1)
    cv2.imwrite(str(path), img)
