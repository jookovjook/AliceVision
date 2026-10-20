"""Stray Scanner capture -> AliceVision sfmData JSON (views + intrinsics [+ ARKit poses]).

    python stray_to_sfm.py CAPTURE_DIR FRAMES_DIR OUT.sfm [--poses] [--lock-poses]
        [--intrinsics shared|perframe] [--distortion none|lut] [--lock-intrinsics] [--version 1.2.14]

Conventions (verified in scanner/StrayScanner/Helpers/OdometryEncoder.swift): odometry.csv stores the camera-to-world
pose with q = q_WA * q_AC, q_AC = 180 deg about x, i.e. ARKit's camera (x right, y up, z back) is ALREADY turned into
the OpenCV camera (x right, y down, z forward) = AliceVision's camera. So AliceVision c_R_w = R_WC^T and center = xyz;
the JSON 'rotation' is c_R_w in Eigen column-major order = R_WC row-major. World = ARKit world (metres, y up).
Intrinsics: fx, fy, cx, cy per frame in rgb pixels (1920x1440). principalPoint is stored as an offset from the image
centre, focal in mm on a virtual 36 mm sensor. Distortion 'lut': radialk3 fitted to Apple's lensDistortionLookupTable
(distorted = centre + (1 + m(r_u)) (p_u - centre), r_u = undistorted radius in reference pixels); AliceVision's
radialk3 is centred on the principal point, Apple's on the lens distortion centre (we report the error of that).
Measured on capture 6ba4b7d84f (ARKit poses + intrinsics locked, 248 keyframes): median reprojection residual
1.47 px with 'lut-inv', 1.53 px with no distortion, 1.80 px with 'lut' as documented; RGB-only self-calibration agrees
with 'lut-inv' (pincushion, ~+1.5 % at the frame edge). So the recorded frames follow the INVERSE table: default lut-inv.
viewId = poseId = frameId = frame index (same as cameraInit --viewIdMethod filename on <frame>.jpg).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

W, H, SW = 1920, 1440, 36.0


def read_csv(p):
    return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(p))]


def fit_lut_radialk3(cap, f_px, cx, cy, table="distortion"):
    """Fit k1..k3 (normalised by focal, centred on the principal point) to the Apple LUT; returns k, rms/max px error."""
    cal = read_csv(cap / "calibration.csv")[0]
    rw, rh = float(cal["reference_width"]), float(cal["reference_height"])
    s = W / rw
    dcx, dcy = float(cal["distortion_center_x"]) * s, float(cal["distortion_center_y"]) * s
    lut = np.fromfile(cap / table / "000000.bin", dtype="<f4").astype(float)
    rmax = max(np.hypot(x - dcx, y - dcy) for x in (0, W) for y in (0, H))  # in video px (LUT spans to the far corner)
    r_lut = np.linspace(0, rmax, len(lut))

    def apple(pu):  # undistorted -> distorted, video px
        d = pu - [dcx, dcy]
        r = np.linalg.norm(d, axis=1)
        return [dcx, dcy] + (1 + np.interp(r, r_lut, lut))[:, None] * d

    # sample a pixel grid of undistorted points, fit radialk3 about the principal point in normalised coords
    g = np.stack(np.meshgrid(np.linspace(0, W, 49), np.linspace(0, H, 37)), -1).reshape(-1, 2)
    pd = apple(g)
    xu = (g - [cx, cy]) / f_px
    xd = (pd - [cx, cy]) / f_px
    r2 = (xu ** 2).sum(1)
    # xd = xu (1 + k1 r2 + k2 r2^2 + k3 r2^3)  -> linear in k
    A = np.stack([np.concatenate([xu[:, 0] * r2 ** p, xu[:, 1] * r2 ** p]) for p in (1, 2, 3)], 1)
    b = np.concatenate([xd[:, 0] - xu[:, 0], xd[:, 1] - xu[:, 1]])
    k = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = xu * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)[:, None] * f_px + [cx, cy]
    err = np.linalg.norm(pred - pd, axis=1)
    disp = np.linalg.norm(pd - g, axis=1)
    return k, float(np.sqrt((err ** 2).mean())), float(err.max()), float(disp.max()), (dcx, dcy)


def intrinsic(iid, fx, fy, cx, cy, dist, locked, serial):
    return {
        "intrinsicId": str(iid), "width": str(W), "height": str(H),
        "sensorWidth": str(SW), "sensorHeight": str(SW * H / W), "serialNumber": serial,
        "type": "pinhole", "initializationMode": "calibrated",
        "initialFocalLength": str(fy * SW / max(W, H)), "focalLength": str(fy * SW / max(W, H)),
        "pixelRatio": str(fy / fx), "pixelRatioLocked": "true",
        "scaleLocked": str(locked).lower(), "offsetLocked": str(locked).lower(),
        "principalPoint": [str(cx - W / 2.0), str(cy - H / 2.0)],
        "distortionType": "radialk3" if dist is not None else "none",
        "distortionParams": [str(v) for v in (dist if dist is not None else [])],
        "distortionLocked": str(locked).lower(), "distortionInitializationMode": "calibrated" if dist is not None else "none",
        "undistortionType": "none", "undistortionOffset": ["0", "0"], "undistortionParams": [],
        "locked": str(locked).lower(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("frames"); ap.add_argument("out")
    ap.add_argument("--poses", action="store_true"); ap.add_argument("--lock-poses", action="store_true")
    ap.add_argument("--intrinsics", choices=["shared", "perframe"], default="shared")
    ap.add_argument("--distortion", choices=["none", "lut", "lut-inv"], default="lut-inv",
                    help="lut: Apple forward table as documented (undistorted -> recorded); lut-inv: the inverse table used the same way (tests the opposite direction)")
    ap.add_argument("--lock-intrinsics", action="store_true")
    ap.add_argument("--version", default="1.2.14")
    a = ap.parse_args()
    cap, frames = Path(a.capture), Path(a.frames)
    odo = {r["frame"]: r for r in read_csv(cap / "odometry.csv")}
    imgs = sorted(p for p in frames.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    sel = [odo[p.stem] for p in imgs]
    med = {k: float(np.median([float(r[k]) for r in sel])) for k in ("fx", "fy", "cx", "cy")}
    dist = None
    if a.distortion in ("lut", "lut-inv"):
        k, rms, mx, disp, dc = fit_lut_radialk3(cap, med["fy"], med["cx"], med["cy"],
                                                "distortion" if a.distortion == "lut" else "inverse_distortion")
        dist = list(k)
        print(f"LUT -> radialk3 k = {np.round(k, 6).tolist()}  fit rms {rms:.3f} px, max {mx:.3f} px "
              f"(distortion up to {disp:.1f} px; Apple centre {dc[0]:.1f},{dc[1]:.1f} vs principal point {med['cx']:.1f},{med['cy']:.1f})")
    views, intr, poses = [], [], []
    if a.intrinsics == "shared":
        intr.append(intrinsic(1, med["fx"], med["fy"], med["cx"], med["cy"], dist, a.lock_intrinsics, "stray_iphone_wide"))
    for p, r in zip(imgs, sel):
        fi = int(r["frame"]); vid = fi
        iid = 1 if a.intrinsics == "shared" else vid
        if a.intrinsics == "perframe":
            intr.append(intrinsic(iid, float(r["fx"]), float(r["fy"]), float(r["cx"]), float(r["cy"]), dist,
                                  a.lock_intrinsics, f"stray_iphone_wide_{fi:06d}"))
        views.append({"viewId": str(vid), "poseId": str(vid), "frameId": str(fi), "intrinsicId": str(iid),
                      "path": str(p.resolve()), "width": str(W), "height": str(H), "metadata": {}})
        if a.poses:
            R_wc = Rotation.from_quat([float(r[c]) for c in ("qx", "qy", "qz", "qw")]).as_matrix()
            c_R_w = R_wc.T
            poses.append({"poseId": str(vid), "pose": {"transform": {
                "rotation": [repr(float(v)) for v in c_R_w.flatten(order="F")],
                "center": [r["x"], r["y"], r["z"]]}, "locked": "1" if a.lock_poses else "0"}})
    d = {"version": a.version.split("."), "views": views, "intrinsics": intr}
    if poses:
        d["poses"] = poses
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(views)} views, {len(intr)} intrinsics, {len(poses)} poses "
          f"(locked poses {a.lock_poses}, locked intrinsics {a.lock_intrinsics})")


if __name__ == "__main__":
    main()
