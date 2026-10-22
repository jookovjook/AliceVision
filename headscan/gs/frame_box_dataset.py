"""Crop a GS export (gs_export.py) to a device-frame box: masked images + LiDAR initial points, for a fast Stochastic
Solids run on the frame and what it holds (head / table top) only.

    python frame_box_dataset.py EXPORT_DIR OUT_DIR --lo X Y Z --hi X Y Z [--scale 1] [--views-every 1] [--no-mask]
        [--init lidar|sfm|both] [--voxel 0.002] [--dilate 24]

Box corners in device metres (the export may be scaled: --scale = its units per metre, export_check.json).
Mask per view: pixels whose ray crosses the box, minus those whose LiDAR depth (EXPORT/depth, z x scale in mm)
lands outside the box (room, shoulders), opened and dilated by --dilate px (LiDAR is low-res and confidence-2 only, so
pixels without depth are kept); everything else is set to black, the trainer's background, so no Gaussians are spent
on the room. Initial points: in-box LiDAR points of every view,
voxel-downsampled (--voxel metres), coloured from the image (--init lidar), or the export's SfM points in the box
(sfm), or both. Writes OUT_DIR/images, OUT_DIR/sparse/0/{cameras,images,points3D}.txt and box_dataset.json.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_model(sparse):
    cam = open(sparse / "cameras.txt").read().split("\n")
    c = [l.split() for l in cam if l and not l.startswith("#")][0]
    W, H, f, cx, cy = int(c[2]), int(c[3]), float(c[4]), float(c[6]), float(c[7])
    imgs = []
    for l in open(sparse / "images.txt"):
        p = l.split()
        if len(p) == 10 and not l.startswith("#"):
            q = [float(v) for v in p[1:5]]; t = np.array([float(v) for v in p[5:8]])
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            imgs.append((p[9], R, t, l.strip()))
    return (W, H, f, cx, cy), imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export"); ap.add_argument("out")
    ap.add_argument("--lo", type=float, nargs=3, required=True); ap.add_argument("--hi", type=float, nargs=3, required=True)
    ap.add_argument("--scale", type=float, default=None, help="export units per metre (default: export_check.json)")
    ap.add_argument("--views-every", type=int, default=1); ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--init", choices=["lidar", "sfm", "both"], default="lidar")
    ap.add_argument("--voxel", type=float, default=0.002); ap.add_argument("--dilate", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=300000)
    ap.add_argument("--capture", default=None, help="Stray capture: mask with its raw LiDAR (all confidences)")
    a = ap.parse_args()
    t0 = time.time()
    E, O = Path(a.export), Path(a.out)
    (O / "images").mkdir(parents=True, exist_ok=True); (O / "sparse/0").mkdir(parents=True, exist_ok=True)
    s = a.scale or json.load(open(E / "export_check.json")).get("scene_scale", 1.0)
    lo, hi = np.array(a.lo) * s, np.array(a.hi) * s
    (W, H, f, cx, cy), imgs = read_model(E / "sparse/0")
    imgs = imgs[::a.views_every]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(u + 0.5 - cx) / f, (v + 0.5 - cy) / f, np.ones_like(u, float)], -1)   # COLMAP pixel centres
    k = np.ones((9, 9), np.uint8); kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1,) * 2)
    P, C, kept = [], [], []
    for n, (name, R, t, line) in enumerate(imgs):
        img = cv2.imread(str(E / "images" / name))
        z = cv2.imread(str(E / "depth" / name), -1).astype(np.float32) / 1000.0     # export units
        X = (ray * z[..., None]) @ R + (-R.T @ t)                                     # camera -> world: R^T (x - t)
        inb = (z > 0) & np.all((X >= lo) & (X <= hi), -1)
        if a.no_mask:
            m = np.ones((H, W), bool)
        else:
            # keep a pixel when its ray crosses the box, unless LiDAR puts the surface it sees outside the box
            # (room behind, shoulders below); pixels without LiDAR (hair, thin frame parts) are kept
            d = ray @ R; o = -R.T @ t
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tn = np.nanmax(np.minimum(t1, t2), -1); tf = np.nanmin(np.maximum(t1, t2), -1)
            hit = (tf > np.maximum(tn, 0))
            zo, io = z, inb
            if a.capture:   # raw LiDAR, every confidence, nearest sample (distortion ignored: the mask is dilated anyway)
                zr = cv2.imread(str(Path(a.capture) / "depth" / name), -1).astype(np.float32) / 1000.0 * s
                zo = zr[np.clip(((v + (1440 - H) / 2) * zr.shape[0] / 1440).astype(int), 0, zr.shape[0] - 1),
                        np.clip(((u + (1920 - W) / 2) * zr.shape[1] / 1920).astype(int), 0, zr.shape[1] - 1)]
                Xo = (ray * zo[..., None]) @ R + (-R.T @ t)
                io = np.all((Xo >= lo) & (Xo <= hi), -1)
            outside = cv2.morphologyEx(((zo > 0) & ~io).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = (hit & (outside == 0)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)                          # drop specks left between LiDAR samples
            m = (cv2.dilate(m, kd) > 0) & hit
        out = img * m[..., None]
        cv2.imwrite(str(O / "images" / name), out)
        kept.append(float(m.mean()))
        if a.init in ("lidar", "both"):
            sel = inb.copy(); sel[::2, :] = False; sel[:, ::2] = False
            P.append(X[sel]); C.append(img[sel][:, ::-1])
    pts, col = (np.concatenate(P), np.concatenate(C)) if P else (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
    if len(pts):
        key = np.floor(pts / (a.voxel * s)).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        cnt = np.bincount(inv.ravel()); P_ = np.zeros((len(first), 3)); C_ = np.zeros((len(first), 3))
        np.add.at(P_, inv.ravel(), pts); np.add.at(C_, inv.ravel(), col.astype(float))
        pts, col = P_ / cnt[:, None], (C_ / cnt[:, None]).astype(np.uint8)
        if len(pts) > a.max_points:
            pick = np.random.default_rng(0).choice(len(pts), a.max_points, replace=False); pts, col = pts[pick], col[pick]
    if a.init in ("sfm", "both"):
        sp = [l.split() for l in open(E / "sparse/0/points3D.txt") if l.strip() and not l.startswith("#")]
        X = np.array([[float(q) for q in p[1:4]] for p in sp]); Cc = np.array([[int(q) for q in p[4:7]] for p in sp], np.uint8)
        m = np.all((X >= lo) & (X <= hi), -1)
        pts, col = np.concatenate([pts, X[m]]), np.concatenate([col, Cc[m]])
    (O / "sparse/0/cameras.txt").write_text(open(E / "sparse/0/cameras.txt").read())
    (O / "sparse/0/images.txt").write_text("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world->camera)\n" +
                                           "".join(f"{line}\n\n" for _, _, _, line in imgs))
    (O / "sparse/0/points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR (tracks omitted)\n" + "".join(
        f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0.5\n" for i, (p, c) in enumerate(zip(pts, col))))
    info = {"export": str(E), "views": len(imgs), "box_lo_m": a.lo, "box_hi_m": a.hi, "scale": s, "init": a.init,
            "points": int(len(pts)), "mask": not a.no_mask, "kept_pixel_fraction": float(np.mean(kept)), "seconds": time.time() - t0}
    json.dump(info, open(O / "box_dataset.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
