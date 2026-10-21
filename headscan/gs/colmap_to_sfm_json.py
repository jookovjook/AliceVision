"""Minimal AliceVision sfmData JSON (views + poses + one pinhole intrinsic) from a COLMAP text model, so
headscan/viz/render.py can frame figures when only the GS export is at hand (same cameras as sfm_device.sfm).

    python colmap_to_sfm_json.py SPARSE_DIR OUT.sfm
"""
import json
import sys

import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main(sparse, out):
    cam = [l.split() for l in open(f"{sparse}/cameras.txt") if l.strip() and not l.startswith("#")][0]
    w, h, fx, fy, cx, cy = int(cam[2]), int(cam[3]), *map(float, cam[4:8])
    intr = {"intrinsicId": "1", "width": str(w), "height": str(h), "sensorWidth": str(w), "sensorHeight": str(h),
            "focalLength": str(fx), "pixelRatio": str(fx / fy), "principalPoint": [str(cx - w / 2), str(cy - h / 2)],
            "type": "pinhole", "distortionParams": []}
    views, poses = [], []
    for l in open(f"{sparse}/images.txt"):
        e = l.split()
        if len(e) < 10 or l.startswith("#"):
            continue
        R = qvec2rotmat(np.array(e[1:5], float)); t = np.array(e[5:8], float)  # world -> camera
        vid = e[0]
        views.append({"viewId": vid, "poseId": vid, "intrinsicId": "1", "path": e[9], "width": str(w), "height": str(h)})
        poses.append({"poseId": vid, "pose": {"transform": {"rotation": [str(v) for v in R.flatten(order="F")],
                                                            "center": [str(v) for v in -R.T @ t]}, "locked": "1"}})
    json.dump({"version": ["1", "2", "6"], "views": views, "intrinsics": [intr], "poses": poses, "structure": []},
              open(out, "w"), indent=1)
    print(f"wrote {out}: {len(views)} views")


if __name__ == "__main__":
    main(*sys.argv[1:3])
