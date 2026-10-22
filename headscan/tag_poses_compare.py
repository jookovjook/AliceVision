"""Compare tag-only poses (tag_poses.py) with the SfM device-frame poses (sfm_device.sfm), both in the device frame.

    python tag_poses_compare.py REF.sfm OUT_PREFIX NAME=TEST.sfm [NAME=TEST.sfm ...] [--title T]

Per variant: camera-centre and rotation differences on the common views as they are (both are device frames, so this
includes any gauge difference) and after a similarity fit (shape of the trajectory only), the similarity scale, and
the focal lengths. Writes OUT_PREFIX.json and OUT_PREFIX.png (per-view differences along the capture).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_poses import umeyama  # noqa: E402


def load(path):
    d = json.load(open(path))
    P = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                       np.array([float(x) for x in p["pose"]["transform"]["center"]]) * 1000.0) for p in d["poses"]}
    i = d["intrinsics"][0]
    f = float(i["focalLength"]) * max(float(i["width"]), float(i["height"])) / float(i["sensorWidth"])
    return {int(v["viewId"]): P[v["poseId"]] for v in d["views"] if v.get("poseId") in P}, f


def cam_model(path):
    """viewId -> (fx, fy, cx, cy, radialk3)."""
    d = json.load(open(path)); K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], np.array(([float(v) for v in i.get("distortionParams", [])] + [0, 0, 0])[:3]))
    return {int(v["viewId"]): K[v["intrinsicId"]] for v in d["views"]}


def tracks(path, n, seed=0):
    """n random SIFT landmarks with >= 3 observations: list of (viewIds, pixel coords)."""
    s = [l for l in json.load(open(path))["structure"] if l["descType"] != "tag36h11" and len(l["observations"]) >= 3]
    rng = np.random.default_rng(seed)
    return [([int(o["observationId"]) for o in s[j]["observations"]], np.array([[float(q) for q in o["x"]] for o in s[j]["observations"]]))
            for j in rng.choice(len(s), min(n, len(s)), replace=False)]


def retriangulate(poses, cam, trk):
    """Triangulate every track with the given poses (fixed) + intrinsics, return per-observation reprojection (px)."""
    from tag_poses import distort, undistort
    errs = []
    for vids, uv in trk:
        keep = [j for j, v in enumerate(vids) if v in poses]
        if len(keep) < 3:
            continue
        xu = np.array([undistort(uv[j:j + 1], np.array(cam[vids[j]][:2]), np.array(cam[vids[j]][2:4]), cam[vids[j]][4])[0] for j in keep])
        A = []
        for j, (x, y) in zip(keep, xu):
            R, C = poses[vids[j]]; P = np.c_[R, -R @ C]
            A += [x * P[2] - P[0], y * P[2] - P[1]]
        X = np.linalg.svd(np.array(A))[2][-1]; X = X[:3] / X[3]
        for _ in range(5):   # Gauss-Newton on the normalised reprojection
            J, r = [], []
            for j, (x, y) in zip(keep, xu):
                R, C = poses[vids[j]]; Xc = R @ (X - C)
                r += [Xc[0] / Xc[2] - x, Xc[1] / Xc[2] - y]
                J += [(R[0] - Xc[0] / Xc[2] * R[2]) / Xc[2], (R[1] - Xc[1] / Xc[2] * R[2]) / Xc[2]]
            X = X - np.linalg.lstsq(np.array(J), np.array(r), rcond=None)[0]
        for j in keep:
            R, C = poses[vids[j]]; Xc = R @ (X - C)
            if Xc[2] <= 0:
                continue
            fx, fy, cx, cy, k = cam[vids[j]]
            xd = distort(Xc[None, :2] / Xc[2], k[None])[0]
            errs.append(np.hypot(fx * xd[0] + cx - uv[j, 0], fy * xd[1] + cy - uv[j, 1]))
    return np.array(errs)


def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("out"); ap.add_argument("tests", nargs="+"); ap.add_argument("--title", default="")
    ap.add_argument("--tracks", type=int, default=3000, help="SIFT landmarks of REF re-triangulated with each pose set (0: off)")
    a = ap.parse_args()
    ref, fref = load(a.ref)
    summ = {"ref": a.ref, "ref_posed": len(ref), "ref_focal_px": fref, "variants": {}}
    trk = tracks(a.ref, a.tracks) if a.tracks else None
    if trk:
        e = retriangulate(ref, cam_model(a.ref), trk)
        summ["ref_sift_reproj_px"] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "n": len(e)}
        print(f"ref: SIFT tracks re-triangulated, reprojection median {np.median(e):.2f} px (p90 {np.percentile(e, 90):.2f}, {len(e)} obs)")
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    for spec in a.tests:
        name, path = spec.split("=", 1)
        T, f = load(path)
        ids = sorted(set(T) & set(ref))
        X = np.array([T[i][1] for i in ids]); Y = np.array([ref[i][1] for i in ids])
        dc = np.linalg.norm(X - Y, axis=1); dr = np.array([ang(T[i][0], ref[i][0]) for i in ids])
        s, R, t = umeyama(X, Y)
        ds = np.linalg.norm(s * X @ R.T + t - Y, axis=1)
        drs = np.array([ang(T[i][0] @ R.T, ref[i][0]) for i in ids])
        st = lambda v: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}
        summ["variants"][name] = {"path": path, "posed": len(T), "common": len(ids), "focal_px": f,
                                  "centre_mm": st(dc), "rotation_deg": st(dr), "sim_scale": float(s),
                                  "centre_after_sim_mm": st(ds), "rotation_after_sim_deg": st(drs),
                                  "sim_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                                  "sim_translation_mm": float(np.linalg.norm(t))}
        if trk:
            e = retriangulate(T, cam_model(path), trk)
            summ["variants"][name]["sift_reproj_px"] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "n": len(e)}
            print(f"{name:10s} SIFT tracks re-triangulated: median {np.median(e):.2f} px (p90 {np.percentile(e, 90):.2f}, {len(e)} obs)")
        ax[0].plot(ids, dc, ".-", ms=3, lw=0.8, label=f"{name}: median {np.median(dc):.1f} mm")
        ax[1].plot(ids, dr, ".-", ms=3, lw=0.8, label=f"{name}: median {np.median(dr):.2f} deg")
        print(f"{name:10s} {len(ids)} views  centre {np.median(dc):6.2f} mm (p90 {np.percentile(dc, 90):5.2f})  rot {np.median(dr):.3f} deg"
              f"  | after sim: scale {s:.4f}, centre {np.median(ds):.2f} mm, rot {np.median(drs):.3f} deg | f {f:.1f} vs {fref:.1f}")
    ax[0].set_ylabel("camera centre difference, mm"); ax[1].set_ylabel("camera rotation difference, deg")
    for x in ax:
        x.set_xlabel("frame"); x.legend(fontsize=8); x.grid(alpha=0.3)
    fig.suptitle(a.title or "tag-only poses vs SfM sfm_device poses (device frame, no alignment)")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=110)
    json.dump(summ, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
