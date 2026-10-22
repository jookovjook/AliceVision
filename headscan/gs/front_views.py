"""Front-of-head subset of a GS export (gs_export.py): only the views that look at the face, for a face-only Stochastic
Solids run.

    python front_views.py EXPORT_DIR OUT_DIR [--target 0 0.08 -0.03] [--max-angle 35] [--min-y 0.0] [--scale 1]

A view is kept when its camera centre is in front of the ear axis (device y > --min-y, metres) and its optical axis
points within --max-angle degrees of the target (face centre, device metres). OUT_DIR gets images/ (symlinks),
sparse/0/{cameras,images,points3D}.txt (same camera model, filtered views, all points) and front_views.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export"); ap.add_argument("out")
    ap.add_argument("--target", type=float, nargs=3, default=[0.0, 0.08, -0.03])
    ap.add_argument("--max-angle", type=float, default=35.0); ap.add_argument("--min-y", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    E, O = Path(a.export), Path(a.out)
    (O / "images").mkdir(parents=True, exist_ok=True); (O / "sparse/0").mkdir(parents=True, exist_ok=True)
    lines = [l for l in open(E / "sparse/0/images.txt") if l.strip() and not l.startswith("#")]
    tgt = np.array(a.target) * a.scale
    keep, total = [], 0
    for l in lines:
        p = l.split(); total += 1
        q = np.array([float(v) for v in p[1:5]]); t = np.array([float(v) for v in p[5:8]])
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()   # world -> camera
        C = -R.T @ t; ax = R.T @ np.array([0, 0, 1.0]); d = tgt - C; d /= np.linalg.norm(d)
        ang = np.degrees(np.arccos(np.clip(ax @ d, -1, 1)))
        if C[1] / a.scale > a.min_y and ang < a.max_angle:
            keep.append(l); name = p[9]
            dst = O / "images" / name
            if not dst.exists():
                os.symlink(os.path.realpath(E / "images" / name), dst)
    with open(O / "sparse/0/images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (front views)\n")
        for l in keep:
            f.write(l.rstrip("\n") + "\n\n")
    for n in ("cameras.txt", "points3D.txt"):
        dst = O / "sparse/0" / n
        if not dst.exists():
            os.symlink(os.path.realpath(E / "sparse/0" / n), dst)
    info = {"views_total": total, "views_kept": len(keep), "target_m": a.target, "max_angle_deg": a.max_angle, "min_y_m": a.min_y}
    json.dump(info, open(O / "front_views.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
