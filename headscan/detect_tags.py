"""Detect AprilTag 36h11 in frames (Python reference detector, pupil-apriltags = AprilRobotics apriltag 3).

    python detect_tags.py FRAMES_DIR OUT.json [--decimate 1.0] [--stride 1] [--max-hamming 1] [--jobs 60]
    python detect_tags.py --check-sheet tag_sheet.png      # corner-order sanity check on the printed sheet

Corners are written in the layout order of frame_tags_layout.json (OpenCV TL, TR, BR, BL as seen with the tag
upright, i.e. as printed on tag_sheet.png). For an upright tag pupil-apriltags returns them as TR, TL, BL, BR in image
coordinates (checked on tag_sheet.png, where every tag is upright), so layout corner k = det.corners[[1, 0, 3, 2][k]];
--check-sheet re-verifies this.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_DET = None


def detector(decimate):
    global _DET
    if _DET is None:
        from pupil_apriltags import Detector
        _DET = Detector(families="tag36h11", nthreads=1, quad_decimate=decimate, quad_sigma=0.0,
                        refine_edges=1, decode_sharpening=0.25)
    return _DET


def detect(path, decimate=1.0, max_hamming=1):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    out = []
    for d in detector(decimate).detect(g):
        if d.hamming > max_hamming:
            continue
        c = np.asarray(d.corners)[[1, 0, 3, 2]]  # TR,TL,BL,BR -> TL,TR,BR,BL
        out.append({"id": int(d.tag_id), "corners": c.round(3).tolist(), "center": np.asarray(d.center).round(3).tolist(),
                    "hamming": int(d.hamming), "margin": round(float(d.decision_margin), 2)})
    return path.stem, out


def _job(args):
    return detect(*args)


def check_sheet(path):
    stem, dets = detect(Path(path), 1.0, 0)
    ok = 0
    for d in dets:
        c = np.array(d["corners"])
        tl, tr, br, bl = c
        good = tl[0] < tr[0] and tl[1] < bl[1] and br[0] > bl[0] and br[1] > tr[1]
        ok += good
    print(f"sheet: {len(dets)} tags, ids {sorted(d['id'] for d in dets)}; corner order TL,TR,BR,BL correct for {ok}/{len(dets)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--decimate", type=float, default=1.0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-hamming", type=int, default=1); ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--check-sheet")
    a = ap.parse_args()
    if a.check_sheet:
        return check_sheet(a.check_sheet)
    imgs = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".jpg", ".png"))[::a.stride]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(_job, [(p, a.decimate, a.max_hamming) for p in imgs], chunksize=4))
    json.dump({"family": "tag36h11", "corner_order": "TL,TR,BR,BL (layout order)", "decimate": a.decimate,
               "frames": res}, open(a.out, "w"))
    ids = [d["id"] for v in res.values() for d in v]
    u, n = np.unique(ids, return_counts=True)
    print(f"{len(imgs)} frames, {len(ids)} detections, {sum(1 for v in res.values() if v)} frames with >= 1 tag")
    print("per id:", dict(zip(u.tolist(), n.tolist())))


if __name__ == "__main__":
    main()
