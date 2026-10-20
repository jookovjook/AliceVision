"""Compare AliceVision tag36h11 features (<viewId>.tag36h11.feat/.desc) with the Python reference detections.

    python check_tag_features.py FEATURES_DIR REFERENCE.json

.feat = text lines "x y scale orientation"; .desc = uint64 count + count x 256 one-hot bytes, index = tagId*4 + corner.
viewId = frame index (our convention). Reports detection agreement and the corner position difference in pixels.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(feat_dir, vid):
    f = Path(feat_dir) / f"{vid}.tag36h11.feat"
    if not f.exists():
        return None
    pts = np.loadtxt(f, ndmin=2)
    raw = (Path(feat_dir) / f"{vid}.tag36h11.desc").read_bytes()
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    desc = np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256)
    idx = desc.argmax(1)
    assert (desc.max(1) == 255).all() and ((desc > 0).sum(1) == 1).all(), "descriptor is not one-hot"
    return {int(i): p[:2] for i, p in zip(idx, pts)}


def main():
    feat_dir, ref = sys.argv[1], json.load(open(sys.argv[2]))["frames"]
    both = only_ref = only_av = 0
    d = []
    for fr, dets in ref.items():
        av = load(feat_dir, int(fr))
        if av is None:
            continue
        refc = {t["id"] * 4 + k: np.array(c) for t in dets if t["id"] < 64 for k, c in enumerate(t["corners"])}
        for i in set(refc) | set(av):
            if i in refc and i in av:
                both += 1; d.append(np.linalg.norm(refc[i] - av[i]))
            elif i in refc:
                only_ref += 1
            else:
                only_av += 1
    d = np.array(d)
    print(f"corners in both: {both}, only Python: {only_ref}, only AliceVision: {only_av}")
    print(f"position difference px: median {np.median(d):.4f}, p99 {np.percentile(d, 99):.4f}, max {d.max():.4f}")


if __name__ == "__main__":
    main()
