"""Pictures of AprilTag detections: tag-by-frame presence chart + annotated sample frames.

    python tags.py DETECTIONS.json FRAMES_DIR OUT_PREFIX [--samples 000004,000500,...]

Frames are shown rotated 90 deg clockwise (the phone was held in portrait; the video is stored landscape).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

GROUPS = [(0, 13, "frame tags 0-13", "tab:blue"), (20, 43, "screw tags 20-43", "tab:orange"), (44, 47, "knob tags 44-47", "tab:green")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("--samples", default="")
    a = ap.parse_args()
    D = json.load(open(a.dets))["frames"]
    names = sorted(D)
    fidx = np.array([int(n) for n in names])
    ids = list(range(0, 14)) + list(range(20, 48))
    M = np.full((len(ids), len(names)), np.nan)
    for j, n in enumerate(names):
        for d in D[n]:
            if d["id"] in ids:
                M[ids.index(d["id"]), j] = d["margin"]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    for lo, hi, lab, col in GROUPS:
        for t in range(lo, hi + 1):
            r = ids.index(t)
            xs = fidx[~np.isnan(M[r])]
            ax.scatter(xs, np.full(len(xs), r), s=6, color=col, marker="s")
    cnt = (~np.isnan(M)).sum(1)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{t} ({c})" for t, c in zip(ids, cnt)], fontsize=7)
    for lo, hi, lab, col in GROUPS:
        ax.text(fidx.max() * 1.01, ids.index(lo) + (hi - lo) / 2, lab, color=col, va="center", fontsize=9)
    ax.set_xlabel("frame"); ax.invert_yaxis()
    ax.set_title(f"AprilTag 36h11 detections in {len(names)} frames (label: tag id (frames seen)); "
                 f"missing: {[t for t, c in zip(ids, cnt) if c == 0]}")
    plt.tight_layout(); plt.savefig(a.out + "_presence.png", dpi=110); plt.close()

    samples = a.samples.split(",") if a.samples else [names[i] for i in np.linspace(0, len(names) - 1, 4).astype(int)]
    fig, axs = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 6))
    for ax, n in zip(np.atleast_1d(axs), samples):
        img = cv2.imread(str(Path(a.frames) / f"{n}.jpg"))
        for d in D.get(n, []):
            c = np.array(d["corners"])
            col = (255, 128, 0) if d["id"] <= 13 else (0, 140, 255) if d["id"] <= 43 else (0, 200, 0)
            cv2.polylines(img, [c.astype(np.int32)], True, col, 3)
            cv2.circle(img, tuple(c[0].astype(int)), 6, (0, 0, 255), -1)  # layout corner 0 (TL)
            cv2.putText(img, str(d["id"]), tuple((c.mean(0) + [-15, 10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ax.imshow(img[..., ::-1]); ax.set_axis_off(); ax.set_title(f"frame {n}: {len(D.get(n, []))} tags", fontsize=10)
    plt.suptitle("blue: frame tags, orange: screw tags, green: knob tags; red dot = corner 0 (top-left as printed)")
    plt.tight_layout(); plt.savefig(a.out + "_samples.png", dpi=90); plt.close()
    print("wrote", a.out + "_presence.png", a.out + "_samples.png")


if __name__ == "__main__":
    main()
