"""Tag-corner landmarks for a pose-only sfmData (tag_poses.py has cameras but no structure).

    python tag_triangulate.py POSES.sfm FEAT_TAGS_DIR OUT.sfm [--min-views 3] [--max-px 3]

Every tag36h11 corner index (tagId*4 + corner) seen in >= --min-views views is triangulated with the fixed cameras
(linear DLT on undistorted rays, then robust refinement), views reprojecting worse than --max-px are dropped. Writes
OUT.sfm = POSES.sfm + 'structure' in AliceVision's layout (descType tag36h11, observations with observationId = viewId
and featureId = row in <viewId>.tag36h11.feat), which is what cad_fit.py's marker_corners() reads.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def read_tags(feat_dir, vid):
    """2-D corners and marker indices (tagId*4 + corner) of one view (same as tag_poses.py, without its cv2 import)."""
    f = Path(feat_dir) / f"{vid}.tag36h11.feat"
    if not f.exists() or f.stat().st_size == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    pts = np.loadtxt(f, ndmin=2)
    raw = (Path(feat_dir) / f"{vid}.tag36h11.desc").read_bytes()
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    return pts[:, :2], np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256).argmax(1)


def undistort(xd, k, it=20):
    x = xd.copy()
    for _ in range(it):
        r2 = (x ** 2).sum(-1, keepdims=True)
        x = xd / (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sfm"); ap.add_argument("feat"); ap.add_argument("out")
    ap.add_argument("--min-views", type=int, default=3); ap.add_argument("--max-px", type=float, default=3.0)
    a = ap.parse_args()
    d = json.load(open(a.sfm))
    I = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        I[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], np.array(([float(v) for v in i.get("distortionParams", [])] + [0, 0, 0])[:3]))
    P = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                       np.array([float(x) for x in p["pose"]["transform"]["center"]])) for p in d["poses"]}
    obs = {}   # marker index -> [(viewId, featureId, ray origin C, ray dir (world), R, C, K, uv)]
    for v in d["views"]:
        if v.get("poseId") not in P:
            continue
        uv, idx = read_tags(a.feat, v["viewId"])
        if not len(uv):
            continue
        R, C = P[v["poseId"]]; fx, fy, cx, cy, k = I[v["intrinsicId"]]
        xn = undistort(np.c_[(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy], k)
        dirs = np.c_[xn, np.ones(len(xn))] @ R           # camera -> world (R = c_R_w)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for j, (m, dw) in enumerate(zip(idx, dirs)):
            obs.setdefault(int(m), []).append((v["viewId"], j, C, dw, R, (fx, fy, cx, cy, k), uv[j]))
    struct, errs = [], []

    def proj(X, o):
        R, C = o[4], o[2]; fx, fy, cx, cy, k = o[5]
        xc = R @ (X - C); xn = xc[:2] / xc[2]; r2 = xn @ xn
        xd = xn * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
        return np.array([fx * xd[0] + cx, fy * xd[1] + cy])

    for m, os_ in sorted(obs.items()):
        if len(os_) < a.min_views:
            continue
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o in os_:                                     # least-squares closest point to all rays
            Mo = np.eye(3) - np.outer(o[3], o[3]); A += Mo; b += Mo @ o[2]
        X = np.linalg.solve(A, b)
        X = least_squares(lambda X: np.concatenate([proj(X, o) - o[6] for o in os_]), X, loss="soft_l1", f_scale=1.0).x
        e = np.array([np.linalg.norm(proj(X, o) - o[6]) for o in os_]); keep = [o for o, ei in zip(os_, e) if ei < a.max_px]
        if len(keep) < a.min_views:
            continue
        errs += list(e[e < a.max_px])
        struct.append({"landmarkId": str(len(struct)), "descType": "tag36h11", "color": [str(m), "0", "0"],
                       "X": [f"{v:.9f}" for v in X],
                       "observations": [{"observationId": str(o[0]), "featureId": str(o[1]), "x": [f"{o[6][0]:.4f}", f"{o[6][1]:.4f}"]} for o in keep]})
    d["structure"] = struct
    json.dump(d, open(a.out, "w"), indent=1)
    print(f"{len(struct)} tag-corner landmarks of {len(obs)} seen; reprojection median {np.median(errs):.2f} px")


if __name__ == "__main__":
    main()
