"""Compare camera poses of two sfmData JSON files on their common viewIds.

    python compare_poses.py TEST.json REF.json OUT_PREFIX [--title T]

Fits a similarity (Umeyama, scale + rotation + translation) TEST -> REF on the camera centres, then reports
the scale, the centre residuals (in REF units, e.g. ARKit metres -> mm) and the rotation residuals per view.
Writes OUT_PREFIX.png and OUT_PREFIX.json.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cams(path):
    d = json.load(open(path))
    poses = {}
    for p in d.get("poses", []):
        t = p["pose"]["transform"]
        poses[p["poseId"]] = (np.array([float(x) for x in t["rotation"]]).reshape(3, 3, order="F"),
                              np.array([float(x) for x in t["center"]]))
    out = {}
    for v in d["views"]:
        if v.get("poseId") in poses:
            out[int(v["viewId"])] = poses[v["poseId"]]
    return out, len(d["views"])


def umeyama(X, Y):
    """s, R, t with Y ~ s R X + t."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / Xc.var(0).sum()
    return s, R, my - s * R @ mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test"); ap.add_argument("ref"); ap.add_argument("out"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    T, nT = cams(a.test)
    Rf, nR = cams(a.ref)
    ids = sorted(set(T) & set(Rf))
    X = np.array([T[i][1] for i in ids]); Y = np.array([Rf[i][1] for i in ids])
    s, R, t = umeyama(X, Y)
    # one robust refit without outliers (> 3x median residual)
    res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    keep = res < max(3 * np.median(res), 1e-9)
    s, R, t = umeyama(X[keep], Y[keep])
    res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    # rotation residual: camera axes of TEST mapped into REF world vs REF camera axes
    ang = []
    for i in ids:
        Rt_w = T[i][0] @ R.T  # c_R_w of TEST expressed in REF world
        dR = Rt_w @ Rf[i][0].T
        ang.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    ang = np.array(ang)
    mm = res * 1000.0
    summ = {"test": a.test, "ref": a.ref, "test_posed": len(T), "test_views": nT, "ref_posed": len(Rf),
            "common": len(ids), "inliers": int(keep.sum()), "scale_test_to_ref": float(s),
            "centre_residual_mm": {"median": float(np.median(mm)), "rms": float(np.sqrt((mm ** 2).mean())),
                                    "p95": float(np.percentile(mm, 95)), "max": float(mm.max())},
            "rotation_residual_deg": {"median": float(np.median(ang)), "p95": float(np.percentile(ang, 95)),
                                      "max": float(ang.max())}}
    json.dump(summ, open(a.out + ".json", "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ids, mm, ".-", ms=3); ax[0].set_xlabel("frame"); ax[0].set_ylabel("camera centre residual, mm")
    ax[0].set_title(f"centres after similarity fit: median {np.median(mm):.1f} mm, p95 {np.percentile(mm, 95):.1f}")
    ax[1].plot(ids, ang, ".-", ms=3, color="tab:orange"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("deg")
    ax[1].set_title(f"rotation residual: median {np.median(ang):.2f} deg, p95 {np.percentile(ang, 95):.2f}")
    Yp = (s * (R @ X.T)).T + t
    ax[2].plot(Y[:, 0], Y[:, 2], "-", lw=1, label="reference (ARKit)")
    ax[2].plot(Yp[:, 0], Yp[:, 2], "--", lw=1, label="test, aligned")
    ax[2].set_aspect("equal"); ax[2].legend(fontsize=8); ax[2].set_title("camera path, top view (x-z, m)")
    fig.suptitle(f"{a.title}   {len(ids)} common cameras, scale test->ref {s:.4f}")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=110)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
