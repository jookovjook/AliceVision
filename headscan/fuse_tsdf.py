"""Volumetric (TSDF) fusion of AliceVision MVS depth maps, optionally with iPhone LiDAR depth, + per-vertex colour.

    python fuse_tsdf.py DENSE_DIR DEVICE_SFM OUT.ply [--capture CAP --lidar-weight 0.3 --k 0.9938]
                        [--voxel 0.0015 --trunc 0.006 --half 0.22] [--volume V.npz [--from-volume]]

MVS: DENSE_DIR/depthFiltered/<viewId>_depthMap.exr = Euclidean ray length in the undistorted pinhole camera of
DENSE_DIR/sfm_pinhole.sfm (<= 0 invalid); simMap weights the sample (w = clip(-sim, 0.2, 1)).
LiDAR (optional): depth/<frame>.png z-depth in the recorded (distorted) camera of DEVICE_SFM, times k, weight
--lidar-weight per view, own truncation (3 x --trunc) because it is ~1 cm noisy.
Signed distance along the ray, truncated, running weighted mean; marching cubes only over cubes whose 8 corners are
observed (weight >= --min-weight), so the back of the truncation band does not become a second surface; faces wound
to point into free space (+T); components with < 2 % of the largest dropped; colour = mean of the recorded frames in
which the vertex is visible (MVS depth agrees within 1 cm).
The volume is a cube of half-size --half around the head box centre (DENSE_DIR/head_box.json).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR


def cams(sfm):
    d = json.load(open(sfm))
    K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], ([float(v) for v in i.get("distortionParams", [])] + [0, 0, 0])[:3])
    P = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                       np.array([float(x) for x in p["pose"]["transform"]["center"]])) for p in d["poses"]}
    return {v["viewId"]: (v, K[v["intrinsicId"]], P[v["poseId"]]) for v in d["views"] if v.get("poseId") in P}


def exr(path):
    with OpenEXR.File(str(path)) as f:
        return list(f.channels().values())[0].pixels.astype(np.float32)


def project(G, R, C, fx, fy, cx, cy, k=None):
    Xc = (G - C.astype(np.float32)) @ R.T.astype(np.float32)
    z = Xc[:, 2]
    ok = z > 0.05
    x = Xc[ok, :2] / z[ok, None]
    if k is not None and any(k):
        r2 = (x ** 2).sum(1, keepdims=True); x = x * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return np.flatnonzero(ok), fx * x[:, 0] + cx, fy * x[:, 1] + cy, Xc[ok]


def integrate(T, Wt, idx, sdf, w, trunc):
    good = np.isfinite(sdf) & (sdf > -trunc)
    idx, s, w = idx[good], np.minimum(sdf[good], trunc) / trunc, w[good]
    w0 = Wt[idx]
    T[idx] = (T[idx] * w0 + s * w) / (w0 + w)
    Wt[idx] = w0 + w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dense"); ap.add_argument("device_sfm"); ap.add_argument("out")
    ap.add_argument("--capture"); ap.add_argument("--lidar-weight", type=float, default=0.0); ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, default=0.0015); ap.add_argument("--trunc", type=float, default=0.006)
    ap.add_argument("--half", type=float, default=0.22); ap.add_argument("--min-weight", type=float, default=3.0)
    ap.add_argument("--volume", help="save the fused grid (T, W) to this .npz; with --from-volume, mesh it instead")
    ap.add_argument("--from-volume", action="store_true")
    a = ap.parse_args()
    import trimesh
    from skimage.measure import marching_cubes
    dense = Path(a.dense)
    ctr = np.array(json.load(open(dense / "head_box.json"))["centre"], np.float32)
    n = int(round(2 * a.half / a.voxel))
    ax = ((np.arange(n) + 0.5) * a.voxel - a.half).astype(np.float32)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + ctr
    T = np.ones(len(G), np.float32); Wt = np.zeros(len(G), np.float32)
    pin = cams(dense / "sfm_pinhole.sfm"); dev = cams(a.device_sfm)
    print(f"grid {n}^3 = {len(G) / 1e6:.1f} M voxels of {a.voxel * 1000:.1f} mm; {len(pin)} MVS views", flush=True)
    if a.from_volume:
        z = np.load(a.volume)
        assert int(z["n"]) == n and float(z["voxel"]) == a.voxel and np.allclose(z["ctr"], ctr), "volume grid differs"
        T, Wt, a.lidar_weight = z["T"], z["W"], float(z["lidar_weight"])
    for vid, (v, (fx, fy, cx, cy, _), (R, C)) in ({} if a.from_volume else pin).items():
        f = dense / "depthFiltered" / f"{vid}_depthMap.exr"
        if not f.exists():
            continue
        D = exr(f); S = exr(dense / "depthFiltered" / f"{vid}_simMap.exr")
        h, w = D.shape; s = w / float(v["width"])
        idx, u, vv, Xc = project(G, R, C, fx * s, fy * s, cx * s, cy * s)
        j, i = np.round(u).astype(np.int32), np.round(vv).astype(np.int32)
        inb = (j >= 0) & (j < w) & (i >= 0) & (i < h)
        idx, j, i, Xc = idx[inb], j[inb], i[inb], Xc[inb]
        d = D[i, j]; valid = d > 0
        integrate(T, Wt, idx[valid], d[valid] - np.linalg.norm(Xc[valid], axis=1),
                  np.clip(-S[i[valid], j[valid]], 0.2, 1.0), a.trunc)
    if a.lidar_weight > 0 and not a.from_volume:
        cap = Path(a.capture)
        for vid, (v, (fx, fy, cx, cy, k), (R, C)) in dev.items():
            z = cv2.imread(str(cap / "depth" / f"{int(v['frameId']):06d}.png"), -1).astype(np.float32) / 1000.0
            c = cv2.imread(str(cap / "confidence" / f"{int(v['frameId']):06d}.png"), -1)
            spread = cv2.dilate(z, np.ones((3, 3))) - cv2.erode(z, np.ones((3, 3)))
            z[(c < 2) | (spread > 0.01) | (z <= 0)] = np.nan
            idx, u, vv, Xc = project(G, R, C, fx, fy, cx, cy, k)
            j = np.round(u * 256 / 1920).astype(np.int32); i = np.round(vv * 192 / 1440).astype(np.int32)
            inb = (j >= 0) & (j < 256) & (i >= 0) & (i < 192)
            idx, j, i, Xc = idx[inb], j[inb], i[inb], Xc[inb]
            integrate(T, Wt, idx, z[i, j] * a.k - Xc[:, 2], np.full(len(idx), a.lidar_weight, np.float32), 3 * a.trunc)
    if a.volume and not a.from_volume:
        np.savez(a.volume, T=T, W=Wt, n=n, voxel=a.voxel, half=a.half, ctr=ctr, lidar_weight=a.lidar_weight)
    T3 = T.reshape(n, n, n).copy(); seen = Wt.reshape(n, n, n) >= a.min_weight
    T3[~seen] = 1.0
    verts, faces, _, _ = marching_cubes(T3, level=0.0)  # verts in voxel index coordinates
    # Keep only zero crossings between two observed voxels (each vertex sits on the grid edge floor(v)..ceil(v)).
    # Voxels more than trunc behind a surface are never integrated; meshing them as +1 (empty) put a second,
    # inward-facing surface ~trunc inside every wall (the old outputs had it on every ray out of the head).
    lo_, hi_ = np.floor(verts).astype(np.int64), np.ceil(verts).astype(np.int64)
    ok_v = seen[tuple(lo_.T)] & seen[tuple(hi_.T)]
    nf0 = len(faces); faces = faces[ok_v[faces].all(1)]
    # wind every face so its normal points to +T (free space): checked against the volume, not assumed
    from scipy.ndimage import map_coordinates
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    c0 = tri.mean(1)
    out_ = (map_coordinates(T3, (c0 + nrm).T, order=1) > map_coordinates(T3, (c0 - nrm).T, order=1)).mean()
    if out_ < 0.5:
        faces = faces[:, ::-1]
    print(f"marching cubes: {nf0} faces, {len(faces)} between observed voxels; "
          f"{max(out_, 1 - out_):.3f} of them face free space after winding", flush=True)
    verts = verts * a.voxel
    verts = (verts - a.half + a.voxel / 2 + ctr).astype(np.float32)
    m = trimesh.Trimesh(verts, faces, process=False)
    parts = m.split(only_watertight=False)
    big = max(len(p.faces) for p in parts)
    m = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.02 * big])
    # colour: mean over recorded frames where the vertex is visible (MVS depth agrees within 1 cm)
    V = m.vertices.astype(np.float32); acc = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
    for vid, (v, (fx, fy, cx, cy, _), (R, C)) in pin.items():
        f = dense / "depthFiltered" / f"{vid}_depthMap.exr"
        if not f.exists():
            continue
        D = exr(f); h, w = D.shape; s = w / float(v["width"])
        idx, u, vv, Xc = project(V, R, C, fx * s, fy * s, cx * s, cy * s)
        j, i = np.round(u).astype(int), np.round(vv).astype(int)
        inb = (j >= 0) & (j < w) & (i >= 0) & (i < h)
        idx, j, i, Xc = idx[inb], j[inb], i[inb], Xc[inb]
        vis = np.abs(D[i, j] - np.linalg.norm(Xc, axis=1)) < 0.01
        dv = dev[vid]; (_, (Fx, Fy, Cx, Cy, k), _) = dv
        ii, U, VV, _ = project(V[idx[vis]], R, C, Fx, Fy, Cx, Cy, k)
        img = cv2.imread(v["path"])[..., ::-1]
        U, VV = np.clip(np.round(U).astype(int), 0, img.shape[1] - 1), np.clip(np.round(VV).astype(int), 0, img.shape[0] - 1)
        sel = idx[vis][ii]
        np.add.at(acc, sel, img[VV, U]); np.add.at(cnt, sel, 1)
    col = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], 128).astype(np.uint8)
    m.visual.vertex_colors = np.c_[col, np.full(len(col), 255, np.uint8)]
    m.export(a.out)
    print(f"wrote {a.out}: {len(m.faces)} faces, {len(parts)} components before filtering, "
          f"lidar weight {a.lidar_weight}, coloured vertices {np.mean(cnt > 0):.3f}")


if __name__ == "__main__":
    main()
