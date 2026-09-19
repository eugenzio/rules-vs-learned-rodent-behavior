"""Download all raw inputs with integrity checks.

ETH videos + Stockholm files: md5 checked against the Zenodo API.
DLCAnalyzer labels + DLC CSVs: fetched at a pinned commit, sha256 recorded.
Writes results/manifest_download.json.
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DLC_SHA = "d6f9532d191e388a32bef2063f7fd4067946dcec"  # only commit touching the label file
DLC_OUT_SHA = "78cf2233706dda43d6fd7aa12a0114a8743625aa"  # only commit touching data/OFT/Output_DLC
ETH_IDS = [5, 6, 11, 12, 14, 15, 16, 23, 24, 38, 39, 41, 43, 44, 49, 50, 51, 52, 54, 58]
STOCKHOLM_FILES = [
    "batch2_videos.rar", "batch2_tracking.rar",
    "batch2-1_manual_labeling.csv", "batch2-2_manual_labeling.csv", "batch2-3_manual_labeling.csv",
    "batch2_1_unsupervised_labeling.csv", "batch2_2_unsupervised_labeling.csv", "batch2_3_unsupervised_labeling.csv",
]


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, md5=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5 and digest(dest, "md5") == md5:
        return "cached"
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if md5 and digest(tmp, "md5") != md5:
        tmp.unlink()
        raise RuntimeError(f"md5 mismatch: {dest.name}")
    tmp.rename(dest)
    return "downloaded"


def zenodo_files(record):
    d = get_json(f"https://zenodo.org/api/records/{record}")
    return {f["key"]: f for f in d["files"]}


def main():
    manifest = {"dlcanalyzer_labels_commit": DLC_SHA, "dlcanalyzer_dlc_commit": DLC_OUT_SHA, "files": {}}
    # --- DLCAnalyzer labels (pinned) ---
    lab_url = f"https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/{DLC_SHA}/data/OFT/Labels/AllLabDataOFT_final.csv"
    lab = RAW / "labels" / "AllLabDataOFT_final.csv"
    fetch(lab_url, lab)
    manifest["files"][str(lab.relative_to(ROOT))] = {"sha256": digest(lab, "sha256"), "url": lab_url}
    # map ID -> DLCFile from the label file itself
    import csv
    rows = list(csv.DictReader(open(lab, newline=""), delimiter=";"))
    id2dlc = {}
    for r in rows:
        id2dlc.setdefault(r["ID"], set()).add(r["DLCFile"])
    assert all(len(v) == 1 for v in id2dlc.values()), "ID->DLCFile not 1:1"
    id2dlc = {k: next(iter(v)) for k, v in id2dlc.items()}
    manifest["id_to_dlcfile"] = id2dlc
    # --- ETH videos ---
    zf = zenodo_files(3608658)
    for i in ETH_IDS:
        name = f"OFT_{i}.mp4"
        meta = zf[name]
        md5 = meta["checksum"].split(":", 1)[1]
        dest = RAW / "eth_videos" / name
        status = fetch(meta["links"]["self"], dest, md5)
        manifest["files"][str(dest.relative_to(ROOT))] = {"md5": md5, "size": meta["size"], "status": status}
        print(name, status, flush=True)
    # --- DLC CSVs for the 20 labelled videos (pinned commit) ---
    for oid, fn in sorted(id2dlc.items()):
        url = f"https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/{DLC_OUT_SHA}/data/OFT/Output_DLC/{fn}"
        dest = RAW / "eth_dlc" / fn
        if not dest.exists():
            fetch(url, dest)
        manifest["files"][str(dest.relative_to(ROOT))] = {"sha256": digest(dest, "sha256"), "id": oid}
        print(oid, fn[:20], "ok", flush=True)
    # --- Stockholm ---
    zs = zenodo_files(14382973)
    for name in STOCKHOLM_FILES:
        meta = zs[name]
        md5 = meta["checksum"].split(":", 1)[1]
        dest = RAW / "stockholm" / name
        status = fetch(meta["links"]["self"], dest, md5)
        manifest["files"][str(dest.relative_to(ROOT))] = {"md5": md5, "size": meta["size"], "status": status}
        print(name, status, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(manifest, open(ROOT / "results" / "manifest_download.json", "w"), indent=1)
    print("ALL DOWNLOADS VERIFIED")


if __name__ == "__main__":
    sys.exit(main())
