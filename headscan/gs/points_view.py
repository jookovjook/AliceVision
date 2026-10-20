"""Coloured point clouds (e.g. Gaussian centres) from the same cameras as headscan/viz/render.py compare.

    python points_view.py SFM OUT.png LABEL=PLY [LABEL=PLY ...] [--crop 0.4 --min-opacity 0.5 --point-size 2]

PLYs are in the SfM/device frame (no AliceVision OBJ flip). Rows = clouds, columns = 6 azimuths around the head.
"""
import argparse
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.environ.get("HEADSCAN_VIZ", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from render import basis, head_frame, load_sfm, look_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sfm"); ap.add_argument("out"); ap.add_argument("clouds", nargs="+")
    ap.add_argument("--crop", type=float, default=0.4); ap.add_argument("--nviews", type=int, default=6)
    ap.add_argument("--view-dist", type=float, default=0.6); ap.add_argument("--min-opacity", type=float, default=0.5)
    ap.add_argument("--point-size", type=float, default=2.0); ap.add_argument("--title", default="")
    a = ap.parse_args()
    import pyrender
    from plyfile import PlyData
    _, cams, _, _ = load_sfm(a.sfm)
    ctr, up, dist = head_frame(cams)
    e1, e2 = basis(up)
    azs = np.radians(np.arange(0, 360, 360 / a.nviews))
    r = pyrender.OffscreenRenderer(440, 520, point_size=a.point_size)
    rows = [c.split("=", 1) for c in a.clouds]
    fig, axs = plt.subplots(len(rows), len(azs), figsize=(2.3 * len(azs), 2.75 * len(rows)), squeeze=False)
    for ri, (lab, path) in enumerate(rows):
        el = PlyData.read(path)["vertex"]
        p = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float32)
        c = np.stack([el["red"], el["green"], el["blue"]], 1).astype(np.float32) / 255
        keep = np.linalg.norm(p - ctr, axis=1) < a.crop * dist
        if "opacity" in el.data.dtype.names:
            keep &= np.asarray(el["opacity"]) >= a.min_opacity
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1.0] * 3)
        scene.add(pyrender.Mesh.from_points(p[keep], colors=c[keep]))
        cn = scene.add(pyrender.PerspectiveCamera(yfov=np.radians(45)))
        for ai, az in enumerate(azs):
            rd = a.view_dist * dist
            eye = ctr + rd * (np.cos(az) * e1 + np.sin(az) * e2) * np.cos(np.radians(10)) + rd * np.sin(np.radians(10)) * up
            scene.set_pose(cn, look_at(eye, ctr, up))
            img, _ = r.render(scene)
            axs[ri, ai].imshow(img); axs[ri, ai].set_xticks([]); axs[ri, ai].set_yticks([])
            for sp in axs[ri, ai].spines.values(): sp.set_visible(False)
            if ai == 0:
                axs[ri, ai].set_ylabel(f"{lab}\n{keep.sum() / 1e6:.2f} M points", fontsize=9)
    fig.suptitle(a.title); plt.tight_layout(); plt.savefig(a.out, dpi=95); print("wrote", a.out)


if __name__ == "__main__":
    main()
