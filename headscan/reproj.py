"""Reprojection residuals per camera from AliceVision sfm JSON files (structure + observations).

    python reproj.py OUT_PREFIX LABEL=run1/sfm.json LABEL2=run2/sfm.json ...

Projection = AliceVision pinhole + radialk3 (or radial3 of older releases, same formula): x_c = R (X - C),
x = x_c[:2] / x_c[2], x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6), u = fx x_d + cx, v = fy y_d + cy, with
fy = focalLength * max(w, h) / sensorWidth, fx = fy / pixelRatio, (cx, cy) = image centre + principalPoint.
Writes OUT_PREFIX.png (median residual per camera vs frame) and OUT_PREFIX.json.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def residuals(path):
    d = json.load(open(path))
    K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"])
        fx = fy / float(i.get("pixelRatio", 1.0))
        pp = [float(v) for v in i["principalPoint"]]
        k = [float(v) for v in i.get("distortionParams", [])] + [0.0, 0.0, 0.0]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], k[:3])
    P = {}
    for p in d.get("poses", []):
        t = p["pose"]["transform"]
        P[p["poseId"]] = (np.array([float(x) for x in t["rotation"]]).reshape(3, 3, order="F"),
                          np.array([float(x) for x in t["center"]]))
    V = {v["viewId"]: v for v in d["views"]}
    per = {}
    for l in d.get("structure", []):
        X = np.array([float(x) for x in l["X"]])
        for o in l.get("observations", []):
            v = V.get(o["observationId"])
            if v is None or v["poseId"] not in P:
                continue
            R, C = P[v["poseId"]]
            fx, fy, cx, cy, k = K[v["intrinsicId"]]
            xc = R @ (X - C)
            x = xc[:2] / xc[2]
            r2 = x @ x
            xd = x * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
            uv = np.array([fx * xd[0] + cx, fy * xd[1] + cy])
            obs = np.array([float(c) for c in o["x"]])
            per.setdefault(int(v["frameId"]), []).append(np.linalg.norm(uv - obs))
    return per


def main():
    out = sys.argv[1]
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    summ = {}
    for lab, path in runs:
        per = residuals(path)
        fr = sorted(per)
        med = [np.median(per[f]) for f in fr]
        allr = np.concatenate([per[f] for f in fr])
        summ[lab] = {"cameras": len(fr), "observations": int(len(allr)), "rms_px": float(np.sqrt((allr ** 2).mean())),
                     "median_px": float(np.median(allr)), "p95_px": float(np.percentile(allr, 95))}
        ax.plot(fr, med, ".-", ms=3, lw=0.8, label=f"{lab}: median {np.median(allr):.2f} px, rms {summ[lab]['rms_px']:.2f}")
    ax.set_xlabel("frame"); ax.set_ylabel("median reprojection residual per camera, px"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out + ".png", dpi=110)
    json.dump(summ, open(out + ".json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
