"""LiDAR depth (Stray Scanner) against the SfM, and LiDAR-only TSDF fusion.

    python lidar.py compare SFM.sfm CAPTURE OUT_PREFIX            # LiDAR z vs SfM landmark z: scale k, noise
    python lidar.py tsdf    SFM.sfm CAPTURE OUT.ply [--k 1.0] [--voxel 0.0025] [--crop 0.22] [--conf 2]

Depth maps are 256x192 uint16 mm, camera z-depth, aligned to the RECORDED (distorted) rgb frame. A landmark is
projected with the SfM camera model (pinhole + radialk3) into the rgb frame and the depth pixel under it is read
(nearest, confidence >= --conf, 3x3 spread < 1 cm). k = median(z_sfm / z_lidar) is the factor that brings LiDAR
metres into SfM units (device frame: layout metres). For the TSDF, every depth map is first resampled onto an
fused (numpy TSDF, voxels projected with the same model) after scaling by k.
Device frame (step 4): origin = ear-axis midpoint, x right, y forward, z up; the head is near (0, 0, 0.06).
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEAD = np.array([0.0, 0.0, 0.06])
W, H, DW, DH = 1920, 1440, 256, 192


def load(sfm):
    d = json.load(open(sfm))
    K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], [float(v) for v in i["distortionParams"]] + [0, 0, 0])
    P = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                       np.array([float(x) for x in p["pose"]["transform"]["center"]])) for p in d["poses"]}
    V = {v["viewId"]: v for v in d["views"] if v.get("poseId") in P}
    return d, K, P, V


def distort(x, k):
    r2 = (x ** 2).sum(-1, keepdims=True)
    return x * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)


def read_depth(cap, frame, conf_min):
    z = cv2.imread(str(cap / "depth" / f"{frame:06d}.png"), -1).astype(np.float32) / 1000.0
    c = cv2.imread(str(cap / "confidence" / f"{frame:06d}.png"), -1)
    spread = cv2.dilate(z, np.ones((3, 3))) - cv2.erode(z, np.ones((3, 3)))
    z[(c < conf_min) | (spread > 0.01) | (z <= 0)] = np.nan
    return z


def compare(a):
    d, K, P, V = load(a.sfm)
    cap = Path(a.capture)
    obs_by_view = {}
    for l in d["structure"]:
        if len(l["observations"]) < 3:
            continue
        X = np.array([float(x) for x in l["X"]])
        for o in l["observations"]:
            obs_by_view.setdefault(o["observationId"], []).append(X)
    rows = []  # frame, z_sfm, z_lidar, near_head
    for vid, v in V.items():
        if vid not in obs_by_view:
            continue
        R, C = P[v["poseId"]]; fx, fy, cx, cy, k = K[v["intrinsicId"]]
        X = np.array(obs_by_view[vid]); Xc = (X - C) @ R.T
        ok = Xc[:, 2] > 0.05
        X, Xc = X[ok], Xc[ok]
        xd = distort(Xc[:, :2] / Xc[:, 2:], k)
        u, vv = fx * xd[:, 0] + cx, fy * xd[:, 1] + cy
        j = np.round((u + 0.5) * DW / W - 0.5).astype(int); i = np.round((vv + 0.5) * DH / H - 0.5).astype(int)
        inb = (j >= 0) & (j < DW) & (i >= 0) & (i < DH)
        z = read_depth(cap, int(v["frameId"]), a.conf)
        zl = np.full(len(X), np.nan); zl[inb] = z[i[inb], j[inb]]
        near = np.linalg.norm(X - HEAD, axis=1) < 0.2
        f = int(v["frameId"])
        rows += [(f, zs, zz, n) for zs, zz, n in zip(Xc[:, 2], zl, near) if np.isfinite(zz)]
    A = np.array(rows)
    head = A[A[:, 3] == 1]; bg = A[A[:, 3] == 0]
    k_head = float(np.median(head[:, 1] / head[:, 2])); k_bg = float(np.median(bg[:, 1] / bg[:, 2])) if len(bg) else float("nan")
    res = (head[:, 1] - k_head * head[:, 2]) * 1000
    mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
    fr = np.unique(head[:, 0]); kf = [np.median(head[head[:, 0] == f, 1] / head[head[:, 0] == f, 2]) for f in fr]
    s = {"k_head": k_head, "k_background": k_bg, "n_head": int(len(head)), "n_background": int(len(bg)),
         "head_residual_mm": {"median": float(np.median(res)), "robust_sigma": mad, "p5": float(np.percentile(res, 5)),
                              "p95": float(np.percentile(res, 95))}}
    json.dump(s, open(a.out + ".json", "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(fr, kf, ".", ms=3); ax[0].axhline(k_head, color="k", lw=1)
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("z_sfm / z_lidar (head points)")
    ax[0].set_title(f"scale LiDAR -> device frame: k = {k_head:.4f} (background {k_bg:.4f})")
    ax[1].hist(np.clip(res, -30, 30), bins=120, color="tab:blue")
    ax[1].set_xlabel("SfM z - k * LiDAR z, mm (head points)")
    ax[1].set_title(f"LiDAR vs SfM on the head: robust sigma {mad:.1f} mm, n = {len(head)}")
    hb = ax[2].hexbin(head[:, 2], head[:, 1] - k_head * head[:, 2], gridsize=60, bins="log", extent=(0.2, 1.0, -0.03, 0.03))
    ax[2].set_xlabel("LiDAR z, m"); ax[2].set_ylabel("residual, m"); ax[2].set_title("residual vs distance")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=110)
    print(json.dumps(s, indent=1))


def tsdf(a):
    """Plain numpy TSDF (Open3D 0.20's TSDF returns empty meshes in this environment): voxels are projected with the
    SfM camera model straight into the recorded (distorted) depth maps; colour from the full-resolution frames."""
    import trimesh
    from skimage.measure import marching_cubes
    d, K, P, V = load(a.sfm)
    cap = Path(a.capture)
    n = int(round(2 * a.crop / a.voxel))
    ax = (np.arange(n) + 0.5) * a.voxel - a.crop
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32) + HEAD.astype(np.float32)
    T = np.ones(len(G), np.float32); Wt = np.zeros(len(G), np.float32); Col = np.zeros((len(G), 3), np.float32)
    mu = a.trunc
    for vid, v in sorted(V.items(), key=lambda t: int(t[1]["frameId"])):
        R, C = P[v["poseId"]]; fx, fy, cx, cy, k = K[v["intrinsicId"]]
        Xc = (G - C.astype(np.float32)) @ R.T.astype(np.float32)
        zc = Xc[:, 2]
        ok = zc > 0.1
        xd = distort(Xc[ok, :2] / zc[ok, None], k)
        u, vv = fx * xd[:, 0] + cx, fy * xd[:, 1] + cy
        j = np.round((u + 0.5) * DW / W - 0.5).astype(np.int32); i = np.round((vv + 0.5) * DH / H - 0.5).astype(np.int32)
        inb = (j >= 0) & (j < DW) & (i >= 0) & (i < DH)
        idx = np.flatnonzero(ok)[inb]
        z = read_depth(cap, int(v["frameId"]), a.conf) * a.k
        zm = z[i[inb], j[inb]]
        sdf = zm - zc[idx]
        good = np.isfinite(sdf) & (sdf > -mu)
        idx, sdf = idx[good], np.minimum(sdf[good], mu) / mu
        w0 = Wt[idx]
        T[idx] = (T[idx] * w0 + sdf) / (w0 + 1)
        near = np.abs(sdf) < 1
        if near.any():
            rgb = cv2.imread(v["path"])[..., ::-1].astype(np.float32)
            uu = np.clip(np.round(u[inb][good][near]).astype(int), 0, W - 1); vq = np.clip(np.round(vv[inb][good][near]).astype(int), 0, H - 1)
            ci = idx[near]
            Col[ci] = (Col[ci] * w0[near, None] + rgb[vq, uu]) / (w0[near, None] + 1)
        Wt[idx] = w0 + 1
    T3 = T.reshape(n, n, n); W3 = Wt.reshape(n, n, n)
    T3[W3 < a.min_weight] = 1.0
    verts, faces, normals, _ = marching_cubes(T3, level=0.0, spacing=(a.voxel,) * 3)
    verts = verts - a.crop + a.voxel / 2 + HEAD
    vi = np.clip(np.round((verts - HEAD + a.crop) / a.voxel - 0.5).astype(int), 0, n - 1)
    col = Col.reshape(n, n, n, 3)[vi[:, 0], vi[:, 1], vi[:, 2]]
    m = trimesh.Trimesh(verts, faces[:, ::-1], vertex_colors=np.c_[col, np.full(len(col), 255)].astype(np.uint8), process=False)
    parts = m.split(only_watertight=False)
    m = max(parts, key=lambda p: len(p.faces)) if len(parts) else m
    m.export(a.out)
    print(f"wrote {a.out}: {len(m.faces)} faces (largest component) from {len(V)} depth maps "
          f"(k {a.k}, voxel {a.voxel * 1000:.1f} mm, trunc {mu * 1000:.0f} mm, min weight {a.min_weight})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare"); c.add_argument("sfm"); c.add_argument("capture"); c.add_argument("out"); c.add_argument("--conf", type=int, default=2)
    t = sp.add_parser("tsdf"); t.add_argument("sfm"); t.add_argument("capture"); t.add_argument("out")
    t.add_argument("--k", type=float, default=1.0); t.add_argument("--voxel", type=float, default=0.0025)
    t.add_argument("--crop", type=float, default=0.22); t.add_argument("--conf", type=int, default=2)
    t.add_argument("--trunc", type=float, default=0.012); t.add_argument("--min-weight", type=float, default=5)
    a = ap.parse_args()
    compare(a) if a.cmd == "compare" else tsdf(a)


if __name__ == "__main__":
    main()
