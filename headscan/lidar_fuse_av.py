"""Fuse iPhone LiDAR depth with AliceVision MVS depth maps in aliceVision_meshing (no C++ change).

    python lidar_fuse_av.py DENSE_DIR DEVICE_SFM LIDAR_SCALED_SFM CAPTURE OUT_DIR --k 0.9938 [--mode fused|lidar]

Every posed keyframe gets a twin "LiDAR view": viewId + 1000000, same poseId, a 256x192 undistorted pinhole intrinsic
(made by aliceVision_intrinsicsTransforming --scaleFactor 0.1333 on the dense sfm, so AliceVision's own pixel
convention is used), a 256x192 undistorted colour image, and a depth map <viewId>_depthMap.exr = Euclidean ray length
(what meshing expects, not z) of the LiDAR depth resampled from the recorded (distorted) depth map with the device-sfm
camera model, times k (LiDAR metres -> device frame), -1 where invalid (confidence < 2 or a 3x3 spread > 1 cm).
simMap = -1 (strong). mode 'fused': MVS views + depth maps (DENSE_DIR/depthFiltered, symlinked) + LiDAR twins;
mode 'lidar': LiDAR twins only. Writes OUT_DIR/sfm.sfm, OUT_DIR/depth/, OUT_DIR/images_lidar/.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

DW, DH, W, H = 256, 192, 1920, 1440
OFF = 1000000


def intr_px(i):
    w, h = float(i["width"]), float(i["height"])
    fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
    pp = [float(v) for v in i["principalPoint"]]
    return fx, fy, w / 2 + pp[0], h / 2 + pp[1], [float(v) for v in i.get("distortionParams", [])] + [0, 0, 0]


def write_exr(path, a, dtype):
    hdr = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(hdr, {"Y": a.astype(dtype)}) as f:
        f.write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dense"); ap.add_argument("device_sfm"); ap.add_argument("lidar_sfm"); ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--k", type=float, required=True); ap.add_argument("--mode", choices=["fused", "lidar"], default="fused")
    ap.add_argument("--conf", type=int, default=2)
    a = ap.parse_args()
    dense, out, cap = Path(a.dense), Path(a.out), Path(a.capture)
    (out / "depth").mkdir(parents=True, exist_ok=True); (out / "images_lidar").mkdir(exist_ok=True)
    D = json.load(open(dense / "sfm_pinhole.sfm"))
    L = json.load(open(a.lidar_sfm))
    dev = json.load(open(a.device_sfm))
    dist_intr = {i["intrinsicId"]: intr_px(i) for i in dev["intrinsics"]}  # recorded (distorted) camera model
    dev_view_intr = {v["viewId"]: v["intrinsicId"] for v in dev["views"]}
    Lint = {i["intrinsicId"]: i for i in L["intrinsics"]}
    posed = {p["poseId"] for p in D["poses"]}

    views, intrinsics = ([], []) if a.mode == "lidar" else (list(D["views"]), list(D["intrinsics"]))
    for i in L["intrinsics"]:
        j = copy.deepcopy(i); j["intrinsicId"] = str(int(i["intrinsicId"]) + OFF); j["serialNumber"] = "lidar_" + j.get("serialNumber", "")
        intrinsics.append(j)
    n = 0
    for v in L["views"]:
        if v["poseId"] not in posed:
            continue
        vid = int(v["viewId"]) + OFF
        fx, fy, cx, cy, _ = intr_px(Lint[v["intrinsicId"]])                 # 256x192 undistorted pinhole
        Fx, Fy, Cx, Cy, k = dist_intr[dev_view_intr[v["viewId"]]]           # 1920x1440 recorded camera
        jj, ii = np.meshgrid(np.arange(DW, dtype=np.float64), np.arange(DH, dtype=np.float64))
        xu = np.stack([(jj - cx) / fx, (ii - cy) / fy], -1)
        r2 = (xu ** 2).sum(-1, keepdims=True)
        xd = xu * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
        u, vv = Fx * xd[..., 0] + Cx, Fy * xd[..., 1] + Cy                   # recorded rgb pixel
        mj = np.round(u * DW / W).astype(int); mi = np.round(vv * DH / H).astype(int)
        inb = (mj >= 0) & (mj < DW) & (mi >= 0) & (mi < DH)
        fr = int(v["frameId"])
        z = cv2.imread(str(cap / "depth" / f"{fr:06d}.png"), -1).astype(np.float32) / 1000.0
        c = cv2.imread(str(cap / "confidence" / f"{fr:06d}.png"), -1)
        spread = cv2.dilate(z, np.ones((3, 3))) - cv2.erode(z, np.ones((3, 3)))
        bad = (c < a.conf) | (spread > 0.01) | (z <= 0)
        zz = np.full((DH, DW), -1.0, np.float32)
        ok = inb.copy(); ok[inb] = ~bad[mi[inb], mj[inb]]
        zz[ok] = z[mi[ok], mj[ok]] * a.k * np.sqrt(1 + r2[..., 0][ok])     # z -> Euclidean ray length
        write_exr(out / "depth" / f"{vid}_depthMap.exr", zz, np.float32)
        write_exr(out / "depth" / f"{vid}_simMap.exr", np.where(zz > 0, -1.0, 1.0), np.float16)
        rgb = cv2.imread(v["path"])
        small = cv2.resize(rgb, (DW, DH), interpolation=cv2.INTER_AREA)
        und = cv2.remap(small, (u * DW / W).astype(np.float32), (vv * DH / H).astype(np.float32), cv2.INTER_LINEAR)
        ip = out / "images_lidar" / f"{vid}.jpg"; cv2.imwrite(str(ip), und)
        w = copy.deepcopy(v); w["viewId"] = str(vid); w["intrinsicId"] = str(int(v["intrinsicId"]) + OFF); w["path"] = str(ip.resolve())
        views.append(w); n += 1
    if a.mode == "fused":
        for f in (dense / "depthFiltered").iterdir():
            dst = out / "depth" / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    S = dict(D); S["views"] = views; S["intrinsics"] = intrinsics
    json.dump(S, open(out / "sfm.sfm", "w"))
    print(f"{a.mode}: {n} LiDAR views (+ {0 if a.mode == 'lidar' else len(D['views'])} MVS views), k {a.k} -> {out}")


if __name__ == "__main__":
    main()
