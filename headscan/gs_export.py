"""Export the metric device-frame reconstruction for Gaussian-splatting tools (COLMAP + nerfstudio formats).

    python gs_export.py DEVICE_SFM CAPTURE OUT_DIR [--k 0.99384] [--margin 4] [--scale 1]

Images are re-sampled ONCE from the recorded frames onto an ideal, CENTRED pinhole (square pixels, f = the refined
SfM focal) with the same camera model the poses were refined with (AliceVision pinhole + radialk3 about the
principal point), cropped to the largest centred size whose pixels all map inside the recorded frame. So every tool
gets: no distortion, principal point at the image centre, poses in the device frame in METRES (origin = ear-axis
midpoint, x right, y forward, z up; scale from the frame-tag layout).

OUT_DIR/
  images/<frame>.png            undistorted 8-bit PNG (lossless; frames decoded once from rgb.mp4 as JPEG q2)
  sparse/0/{cameras,images,points3D}.txt    COLMAP text model (PINHOLE, R_cw/t_cw, SfM landmarks with colours)
  transforms.json               nerfstudio (OpenGL camera axes), depth_file_path + mask/confidence paths
  depth/<frame>.png             LiDAR z-depth resampled to the image grid, uint16 mm, x k (device-frame units),
                                confidence 2 only, eroded 2 depth px at edges, 0 = none
  depth_conf/<frame>.jpg        AGS-Mesh confidence image (black = trusted, white = ignore)
  points3D.ply                  same landmarks as PLY (nerfstudio init)
  export_check.json             reprojection of the SfM observations through the export (should match the SfM)

--scale S multiplies every world coordinate (camera centres, points, depth) by S, for trainers with absolute
distance limits (the 3DGS rasterizer culls everything nearer than 0.2 units); divide their outputs by S.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

W, H, DW, DH = 1920, 1440, 256, 192


def load(sfm):
    d = json.load(open(sfm))
    K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], ([float(v) for v in i.get("distortionParams", [])] + [0, 0, 0])[:3])
    P = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                       np.array([float(x) for x in p["pose"]["transform"]["center"]])) for p in d["poses"]}
    V = {v["viewId"]: v for v in d["views"] if v.get("poseId") in P}
    return d, K, P, V


def distort(x, k):
    r2 = (x ** 2).sum(-1, keepdims=True)
    return x * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)


def undistort_pts(xd, k, it=20):
    x = xd.copy()
    for _ in range(it):
        r2 = (x ** 2).sum(-1, keepdims=True)
        x = xd / (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return x


def src_map(Wn, Hn, f, K):
    """For each target pixel (OpenCV index coords, centre at (Wn-1)/2), the source pixel in the recorded frame."""
    fx, fy, cx, cy, k = K
    jj, ii = np.meshgrid(np.arange(Wn, dtype=np.float64), np.arange(Hn, dtype=np.float64))
    xu = np.stack([(jj - (Wn - 1) / 2) / f, (ii - (Hn - 1) / 2) / f], -1)
    xd = distort(xu, k)
    return fx * xd[..., 0] + cx, fy * xd[..., 1] + cy


def largest_crop(f, K, margin):
    best = None
    for Wn in range(1920, 1400, -8):
        Hn = int(round(Wn * 0.75 / 8)) * 8
        u, v = src_map(Wn, Hn, f, K)
        border = np.concatenate([u[0], u[-1], u[:, 0], u[:, -1]]), np.concatenate([v[0], v[-1], v[:, 0], v[:, -1]])
        if border[0].min() >= margin and border[0].max() <= W - 1 - margin and border[1].min() >= margin and border[1].max() <= H - 1 - margin:
            best = (Wn, Hn); break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sfm"); ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--k", type=float, default=0.99384); ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="world units per device-frame metre in the export")
    a = ap.parse_args()
    out, cap = Path(a.out), Path(a.capture)
    for s in ("images", "sparse/0", "depth", "depth_conf"):
        (out / s).mkdir(parents=True, exist_ok=True)
    d, K, P, V = load(a.sfm)
    assert len(K) == 1, "expects one shared intrinsic"
    Kc = next(iter(K.values()))
    f = Kc[1]
    Wn, Hn = largest_crop(f, Kc, a.margin)
    cxn, cyn = (Wn - 1) / 2, (Hn - 1) / 2  # OpenCV index coords; COLMAP/nerfstudio cx = Wn/2 (pixel centres at +0.5)
    mu, mv = src_map(Wn, Hn, f, Kc)
    mu32, mv32 = mu.astype(np.float32), mv.astype(np.float32)
    # LiDAR grid lookup (recorded depth pixel under every target pixel)
    dj = np.round((mu + 0.5) * DW / W - 0.5).astype(int).clip(0, DW - 1); di = np.round((mv + 0.5) * DH / H - 0.5).astype(int).clip(0, DH - 1)
    print(f"target pinhole {Wn}x{Hn}, f = {f:.2f} px, centred; source pp ({Kc[2]:.1f}, {Kc[3]:.1f}), k = {np.round(Kc[4], 4).tolist()}")
    cams_txt = [f"1 PINHOLE {Wn} {Hn} {f:.6f} {f:.6f} {Wn / 2:.6f} {Hn / 2:.6f}"]
    imgs_txt, frames = [], []
    for n, (vid, v) in enumerate(sorted(V.items(), key=lambda t: int(t[1]["frameId"])), 1):
        fr = int(v["frameId"]); name = f"{fr:06d}.png"
        img = cv2.imread(v["path"])
        cv2.imwrite(str(out / "images" / name), cv2.remap(img, mu32, mv32, cv2.INTER_CUBIC))
        R, C = P[v["poseId"]]                     # R = c_R_w (OpenCV camera)
        C = C * a.scale
        t = -R @ C
        q = Rotation.from_matrix(R).as_quat()     # x, y, z, w
        imgs_txt += [f"{n} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {name}", ""]
        z = cv2.imread(str(cap / "depth" / f"{fr:06d}.png"), -1).astype(np.float32)
        c = cv2.imread(str(cap / "confidence" / f"{fr:06d}.png"), -1)
        good = (c == 2) & (z > 0)
        good = cv2.erode(good.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # 2 px from any invalid / low-confidence pixel
        spread = cv2.dilate(z, np.ones((3, 3))) - cv2.erode(z, np.ones((3, 3)))
        good &= spread < 10.0
        zt = np.where(good[di, dj], z[di, dj] * a.k * a.scale, 0).astype(np.uint16)
        cv2.imwrite(str(out / "depth" / name), zt)
        cv2.imwrite(str(out / "depth_conf" / name.replace(".png", ".jpg")), np.where(zt > 0, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = C
        frames.append({"file_path": f"images/{name}", "depth_file_path": f"depth/{name}",
                       "confidence_path": f"depth_conf/{name.replace('.png', '.jpg')}",
                       "transform_matrix": (c2w @ np.diag([1, -1, -1, 1])).tolist(), "colmap_im_id": n, "frame": fr})
    # landmarks with colours from their first observing view
    pts, by_view = [], {}
    for l in d.get("structure", []):   # tag-only poses (tag_poses.py) have no landmarks
        X = np.array([float(x) for x in l["X"]])
        obs = [o for o in l["observations"] if o["observationId"] in V]
        if len(obs) < 2:
            continue
        pts.append((X, obs, l["descType"]))
        by_view.setdefault(obs[0]["observationId"], []).append(len(pts) - 1)
    col = np.zeros((len(pts), 3), np.uint8)
    errs = []
    for vid, ids in by_view.items():
        img = cv2.imread(V[vid]["path"])[..., ::-1]
        for i in ids:
            o = pts[i][1][0]; u, vv = (int(np.clip(round(float(x)), 0, s - 1)) for x, s in zip(o["x"], (W, H)))
            col[i] = img[vv, u]
    # export check: observations through the export (undistort to the target pinhole) vs projection of X
    rng = np.random.default_rng(0)
    for i in rng.choice(len(pts), min(20000, len(pts)), replace=False):
        X, obs, _ = pts[i]
        for o in obs[:3]:
            R, C = P[V[o["observationId"]]["poseId"]]
            xc = R @ (X - C); proj = f * xc[:2] / xc[2] + [cxn, cyn]
            fx, fy, cx, cy, k = Kc
            xu = undistort_pts(np.array([[(float(o["x"][0]) - cx) / fx, (float(o["x"][1]) - cy) / fy]]), k)[0]
            errs.append(np.linalg.norm(f * xu + [cxn, cyn] - proj))
    errs = np.array(errs) if errs else np.full(1, np.nan)
    pts_txt = []
    with open(out / "points3D.ply", "w") as fp:
        fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i, (X, obs, _) in enumerate(pts):
            X = X * a.scale
            fp.write(f"{X[0]:.6f} {X[1]:.6f} {X[2]:.6f} {col[i][0]} {col[i][1]} {col[i][2]}\n")
            pts_txt.append(f"{i + 1} {X[0]:.6f} {X[1]:.6f} {X[2]:.6f} {col[i][0]} {col[i][1]} {col[i][2]} 0.5")  # no tracks: 2D points are not exported
    (out / "sparse/0/cameras.txt").write_text(f"# PINHOLE, centred, undistorted, device frame x {a.scale:g} (units per metre)\n" + "\n".join(cams_txt) + "\n")
    (out / "sparse/0/images.txt").write_text("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world->camera)\n" + "\n".join(imgs_txt) + "\n")
    (out / "sparse/0/points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR (tracks omitted)\n" + "\n".join(pts_txt) + "\n")
    json.dump({"camera_model": "OPENCV", "fl_x": f, "fl_y": f, "cx": Wn / 2, "cy": Hn / 2, "w": Wn, "h": Hn,
               "k1": 0, "k2": 0, "p1": 0, "p2": 0, "frames": frames, "ply_file_path": "points3D.ply",
               "depth_unit_scale_factor": 0.001,
               "scene_scale": a.scale,
               "note": "device frame (ear-axis origin, x right, y forward, z up), metres x scene_scale; OpenGL camera axes"},
              open(out / "transforms.json", "w"), indent=1)
    chk = {"target": [Wn, Hn], "f": f, "views": len(frames), "points": len(pts), "scene_scale": a.scale,
           "reproj_px": {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)), "n": int(len(errs))},
           "lidar_valid_fraction": float(np.mean([(cv2.imread(str(out / "depth" / fr["file_path"].split("/")[1]), -1) > 0).mean() for fr in frames[::10]]))}
    json.dump(chk, open(out / "export_check.json", "w"), indent=1)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
