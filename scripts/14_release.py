"""Package release artifacts into release/ (committed to the public repo).

- per-frame tracker features for all 23 videos (our own derived data; DLC-derived columns excluded
  because they derive from GPL-3.0 files), float32 parquet (zstd)
- LOVO predictions for every method (post-processed class codes), one npz per method
- Stockholm one-shot predictions are in results/stockholm/results.json (confusions only)
Consensus labels are NOT redistributed: regenerate with scripts/01_labels.py from the pinned source.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"


def main():
    (REL / "features").mkdir(parents=True, exist_ok=True)
    (REL / "predictions").mkdir(parents=True, exist_ok=True)
    manifest = {"features": {}, "predictions": {}}
    for f in sorted((ROOT / "data/interim/features").glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[[c for c in d.columns if not c.startswith("dlc_")]].astype(np.float32)
        out = REL / "features" / f.name
        d.to_parquet(out, compression="zstd")
        manifest["features"][f.stem] = {"frames": [int(d.index.min()), int(d.index.max()) + 1],
                                        "columns": list(d.columns), "bytes": out.stat().st_size}
    for mdir in sorted((ROOT / "data/interim/preds").iterdir()):
        arrs = {p.stem: np.load(p) for p in sorted(mdir.glob("*.npy"))}
        np.savez_compressed(REL / "predictions" / f"{mdir.name}.npz", **arrs)
        manifest["predictions"][mdir.name] = sorted(arrs)
    manifest["codes"] = {"0": "Other", "1": "Supported rear", "2": "Unsupported rear", "3": "Grooming"}
    manifest["frame_index"] = "0-based video frame; predictions cover the trial window in results/track_meta.json"
    json.dump(manifest, open(REL / "manifest.json", "w"), indent=1)
    tot = sum(p.stat().st_size for p in REL.rglob("*") if p.is_file())
    print(f"release/: {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
