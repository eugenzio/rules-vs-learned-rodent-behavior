"""Compute cost on this machine (run with no other heavy jobs).

Per-frame costs (single process): tracker pass 2 (decode+warp+MOG2+features geometry), background
pre-training (pass 1, per-video setup), raw+windowed features, model inference (batch and single
frame with n_jobs=1), post-processing. End-to-end fps per method. Model sizes (joblib, compress=3).
Output: results/cost.json
"""
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.experiment import Dataset, feature_cols, make_model  # noqa: E402
from rr.features import design, raw_features  # noqa: E402
from rr.postproc import postprocess  # noqa: E402
from rr.rules import fixed_params, predict_rules  # noqa: E402
from rr.track import track_video  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))


def main():
    meta = json.load(open(ROOT / "results/track_meta.json"))
    vid = "OFT_5"
    m = meta[vid]
    f0 = m["window"][0]
    win = (f0, f0 + 3000)
    # ---- tracker: pass 1 + pass 2 on a 3000-frame window (single process) ----
    t = time.perf_counter()
    tr, _, mm = track_video(ROOT / f"data/raw/eth_videos/{vid}.mp4", np.array(m["H"]), m["canvas"], win,
                            m["side"], CFG["tracker"])
    total = time.perf_counter() - t
    pass2_us = 1e6 / mm["pass2_fps"]
    pass1_s = total - mm["pass2_seconds"]   # decode 3000 frames + 60 background updates
    # ---- features ----
    trf = pd.read_parquet(ROOT / f"data/interim/track/{vid}.parquet")
    t = time.perf_counter()
    raw = raw_features(trf, m["n"], tuple(m["window"]), m["side"])
    raw_us = 1e6 * (time.perf_counter() - t) / len(raw)
    t = time.perf_counter()
    X = design(raw, feature_cols("rf"), CFG["features"]["windows_frames"]).to_numpy(np.float32)
    win_us = 1e6 * (time.perf_counter() - t) / len(raw)
    pp = CFG["postprocess"]
    t = time.perf_counter()
    postprocess(np.zeros(len(raw), np.int8), pp["mode_filter_frames"], pp["min_bout_frames"])
    post_us = 1e6 * (time.perf_counter() - t) / len(raw)
    t = time.perf_counter()
    predict_rules(raw, fixed_params(CFG), pp)
    rules_us = 1e6 * (time.perf_counter() - t) / len(raw)
    # ---- models: fit on all ETH (as in cross-lab) and time inference ----
    vids = [f"OFT_{i}" for i in CFG["data"]["eth"]["ids"]]
    d = Dataset(vids)
    models = {}
    for name in ("dt", "rf", "hgb", "rf_rule"):
        cols = feature_cols(name)
        Xtr = np.concatenate([d.X(v, cols, CFG["features"]["windows_frames"])[d.gt[v] >= 0] for v in vids])
        ytr = np.concatenate([d.gt[v][d.gt[v] >= 0] for v in vids])
        mdl = make_model(name, CFG).fit(Xtr, ytr)
        if hasattr(mdl, "n_jobs"):
            mdl.set_params(n_jobs=1)
        Xte = design(raw, cols, CFG["features"]["windows_frames"]).to_numpy(np.float32)
        t = time.perf_counter()
        mdl.predict(Xte)
        batch_us = 1e6 * (time.perf_counter() - t) / len(Xte)
        lat = []
        for i in range(200):
            t = time.perf_counter()
            mdl.predict(Xte[i:i + 1])
            lat.append(time.perf_counter() - t)
        b = io.BytesIO()
        joblib.dump(mdl, b, compress=3)
        models[name] = {"batch_us_per_frame": batch_us, "single_frame_latency_ms_median": 1e3 * float(np.median(lat)),
                        "size_mb": b.tell() / 1e6, "n_features": Xte.shape[1]}
    feat_us = raw_us + win_us
    e2e = {}
    for name, v in models.items():
        e2e[name] = 1e6 / (pass2_us + feat_us + v["batch_us_per_frame"] + post_us)
    e2e["rules"] = 1e6 / (pass2_us + raw_us + rules_us)
    out = {"machine": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                             text=True).stdout.strip(),
                       "cores": subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip(),
                       "python": platform.python_version(), "gpu_used": False},
           "video": vid, "resolution": "928x576", "canvas_px": m["canvas"],
           "tracker_pass2_us_per_frame": pass2_us, "tracker_pass2_fps": 1e6 / pass2_us,
           "tracker_3000frame_window_total_s": total, "pass1_setup_s_per_3000_frames": pass1_s,
           "raw_features_us_per_frame": raw_us, "windowing_us_per_frame": win_us,
           "postprocess_us_per_frame": post_us, "fixed_rules_us_per_frame": rules_us,
           "models": models, "end_to_end_fps": e2e,
           "note": "single process; end-to-end = tracker pass 2 + features + model + post-processing; "
                   "centred +-1 s windows imply 1 s output latency; background pre-training (pass 1) is a per-video setup step"}
    json.dump(out, open(ROOT / "results/cost.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
