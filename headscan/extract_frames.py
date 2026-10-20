"""Extract every rgb.mp4 frame of a Stray Scanner capture and pick keyframes.

    python extract_frames.py CAPTURE_DIR OUT_DIR [--window 8]

Writes OUT_DIR/all/<frame>.jpg (frame = odometry.csv 'frame' index, 6 digits, no auto-rotation so the
frames match the ARKit landscape intrinsics), OUT_DIR/keyframes/<frame>.jpg (symlinks: sharpest frame of
each window of --window frames, by variance of the Laplacian) and OUT_DIR/keyframes.csv.
"""
import argparse
import csv
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


def sharpness(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=8)
    a = ap.parse_args()
    cap, out = Path(a.capture), Path(a.out)
    frames = [r["frame"].strip() for r in csv.DictReader(open(cap / "odometry.csv"), skipinitialspace=True)]
    alld = out / "all"
    alld.mkdir(parents=True, exist_ok=True)
    if len(list(alld.glob("*.jpg"))) != len(frames):
        subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-i", str(cap / "rgb.mp4"), "-vsync", "0",
                        "-q:v", "2", "-start_number", "0", str(alld / "%06d.jpg")], check=True)
    jpgs = sorted(alld.glob("*.jpg"))
    assert len(jpgs) == len(frames), f"{len(jpgs)} video frames vs {len(frames)} odometry rows"
    assert [p.stem for p in jpgs] == frames, "frame names do not match odometry.csv"
    h, w = cv2.imread(str(jpgs[0])).shape[:2]
    print(f"{len(jpgs)} frames {w}x{h}")

    sh = np.array([sharpness(p) for p in jpgs])
    kd = out / "keyframes"
    kd.mkdir(exist_ok=True)
    for p in kd.glob("*.jpg"):
        p.unlink()
    rows = []
    for s in range(0, len(jpgs), a.window):
        i = s + int(np.argmax(sh[s:s + a.window]))
        os.symlink(os.path.relpath(jpgs[i], kd), kd / jpgs[i].name)
        rows.append((frames[i], sh[i], np.median(sh[s:s + a.window])))
    with open(out / "keyframes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "sharpness", "window_median_sharpness"])
        wr.writerows(rows)
    np.save(out / "sharpness_all.npy", sh)
    print(f"{len(rows)} keyframes (window {a.window}); sharpness median all {np.median(sh):.1f}, "
          f"keyframes {np.median([r[1] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
