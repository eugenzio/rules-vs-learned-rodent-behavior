"""Track ETH (and, with --stockholm, Stockholm) videos. Label-blind: uses only trial-window markers.

Outputs: data/interim/track/<vid>.parquet, results/qc/<vid>_bg.png, results/qc/<vid>_f*.png,
results/track_meta.json (merged).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rr.calib import dlc_corners, homography  # noqa: E402
from rr.labels import drop_invalid, load_labels, video_labels  # noqa: E402
from rr.track import track_video  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config/prereg.yaml"))
OUT = ROOT / "data/interim/track"
QC = ROOT / "results/qc"


def eth_jobs():
    man = json.load(open(ROOT / "results/manifest_download.json"))
    df, _ = drop_invalid(load_labels(ROOT / "data/raw/labels/AllLabDataOFT_final.csv"))
    jobs = []
    for i in CFG["data"]["eth"]["ids"]:
        vid = f"OFT_{i}"
        path = ROOT / f"data/raw/eth_videos/{vid}.mp4"
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        win = video_labels(df, vid, n)["window"]           # markers only (label-blind wrt behaviour)
        corners, cqc = dlc_corners(ROOT / "data/raw/eth_dlc" / man["id_to_dlcfile"][vid],
                                   CFG["data"]["eth"]["corner_likelihood_min"])
        jobs.append(dict(vid=vid, path=str(path), corners=corners.tolist(), corner_qc=cqc,
                         window=[int(win[0]), int(win[1])], side=CFG["data"]["eth"]["arena_side_cm"], n=n))
    return jobs


def stockholm_jobs():
    corners = json.load(open(ROOT / "results/stockholm_corners.json"))
    mapping = json.load(open(ROOT / "results/stockholm_mapping.json"))
    jobs = []
    for k, t in mapping["label_file_to_trial"].items():
        path = ROOT / f"data/raw/stockholm/videos/new_Trial     {t}.mp4"
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        jobs.append(dict(vid=f"STK_{k}", path=str(path), corners=corners[str(t)]["corners"], corner_qc={},
                         window=[0, n], side=CFG["data"]["stockholm"]["arena_side_cm"], n=n))
    return jobs


def run(job):
    cal = CFG["calibration"]
    H, size = homography(np.array(job["corners"]), job["side"], cal["canvas_px_per_cm"], cal["margin_cm"])
    f0, f1 = job["window"]
    qc_frames = set(int(x) for x in np.linspace(f0 + 100, f1 - 100, 6))
    QC.mkdir(parents=True, exist_ok=True)
    df, bg, meta = track_video(job["path"], H, size, (f0, f1), job["side"], CFG["tracker"],
                               cal["canvas_px_per_cm"], cal["margin_cm"], qc_frames, QC, job["vid"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{job['vid']}.parquet")
    cv2.imwrite(str(QC / f"{job['vid']}_bg.png"), bg)
    meta.update({k: job[k] for k in ("vid", "window", "side", "n", "corners")})
    meta["median_major_cm"] = float(df.loc[df.found, "major"].median())
    meta["median_area_cm2"] = float(df.loc[df.found, "area"].median())
    meta["H"] = H.tolist()
    meta["canvas"] = size
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stockholm", action="store_true")
    a = ap.parse_args()
    jobs = stockholm_jobs() if a.stockholm else eth_jobs()
    if a.only:
        jobs = [j for j in jobs if j["vid"] in a.only]
    with ProcessPoolExecutor(a.workers) as ex:
        metas = list(ex.map(run, jobs))
    path = ROOT / "results/track_meta.json"
    allm = json.load(open(path)) if path.exists() else {}
    for m, j in zip(metas, jobs):
        m["corner_qc"] = j["corner_qc"]
        allm[m["vid"]] = m
        print(f"{m['vid']}: coverage {m['coverage']:.3f}  fps {m['pass2_fps']:.0f}  "
              f"L_ref {m['median_major_cm']:.2f} cm  A_ref {m['median_area_cm2']:.1f} cm2  win {m['window']}")
    json.dump(allm, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
