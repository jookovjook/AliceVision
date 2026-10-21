"""Model-based fit of the head frame in IMAGE space: CAD + screws fitted to the raw 2-D tag corners of every view.

    python image_fit.py IN_DIR CAD_FIT_JSON OUT_DIR [--ruler 3.5,4,...]

Cameras (poses + pinhole/radialk3 intrinsics) are fixed from the device-frame SfM (IN_DIR/sfm_device.sfm); 2-D tag
corners come from IN_DIR/feat_tags (<viewId>.tag36h11.feat/.desc, index = tagId*4 + corner). Unknowns:
  - rigid pose (6) of part A and of the 3 part-B segments (right arch, apex block, left arch), init from cad_fit S_mix
  - per screw k: depth s_k along its hole axis and wing angle theta_k about it (2), init from the triangulated corners
  - one global scale of the camera centres (the scene scale vs the CAD)
  - one tag inset delta (mm): every printed tag corner is moved delta towards its tag centre (detector bias)
  - per screw (--free-axis): tilt (2, rad) and lateral offset (2, mm) of the screw axis vs the CAD hole axis
    (thread clearance, printed holes), with a prior of --tilt-sd deg and --offset-sd mm
Residuals: reprojection of every observed corner (px), soft-L1 loss (1 px). Tag 20 (mis-glued) is left out.
Readouts per screw: coarse gap = s + 74; fine = phase from theta (hole-thread reference e1_k, 2.5 mm per turn), turn
from the coarse value; calibrated with 2 constants (coarse bias, phase offset) scored leave-one-out against the ruler.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step8_report import E1  # noqa: E402

P = 2.5
BAD_TAGS = {20}


def seg_of(p):
    return "R" if p[0] > 25 else ("L" if p[0] < -25 else "C")


def load(I):
    d = json.load(open(I / "sfm_device.sfm"))
    K = {}
    for i in d["intrinsics"]:
        w, h = float(i["width"]), float(i["height"])
        fy = float(i["focalLength"]) * max(w, h) / float(i["sensorWidth"]); fx = fy / float(i.get("pixelRatio", 1))
        pp = [float(v) for v in i["principalPoint"]]
        K[i["intrinsicId"]] = (fx, fy, w / 2 + pp[0], h / 2 + pp[1], ([float(v) for v in i.get("distortionParams", [])] + [0, 0, 0])[:3])
    Pz = {p["poseId"]: (np.array([float(x) for x in p["pose"]["transform"]["rotation"]]).reshape(3, 3, order="F"),
                        np.array([float(x) for x in p["pose"]["transform"]["center"]]) * 1000.0) for p in d["poses"]}
    cams, obs = [], []
    for v in d["views"]:
        if v.get("poseId") not in Pz:
            continue
        f = I / "feat_tags" / f"{v['viewId']}.tag36h11.feat"
        if not f.exists():
            continue
        pts = np.loadtxt(f, ndmin=2)
        if len(pts) == 0:
            continue
        raw = (I / "feat_tags" / f"{v['viewId']}.tag36h11.desc").read_bytes()
        n = int(np.frombuffer(raw[:8], dtype="<u8")[0]); idx = np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256).argmax(1)
        ci = len(cams); cams.append((*Pz[v["poseId"]], K[v["intrinsicId"]]))
        for (x, y), i in zip(pts[:, :2], idx):
            obs.append((ci, int(i), x, y))
    return cams, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("cadfit"); ap.add_argument("out"); ap.add_argument("--ruler", default="")
    ap.add_argument("--free-axis", action="store_true"); ap.add_argument("--tilt-sd", type=float, default=2.0); ap.add_argument("--offset-sd", type=float, default=1.0)
    a = ap.parse_args()
    I, O = Path(a.inp), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    L = json.load(open(I / "frame_tags_layout.json")); CF = json.load(open(a.cadfit))
    cams, obs = load(I)
    holes = L["holes"]
    # model points: (kind, ref, local corner) ; kind 'f' frame tag (segment), 's' screw tag (screw k)
    model = {}
    for t, cs in L["frame"].items():
        cs = np.array(cs, float); c0 = cs.mean(0); part = "A" if int(t) <= 6 else None
        g = "A" if part == "A" else seg_of(c0)
        for k, c in enumerate(cs):
            model[int(t) * 4 + k] = ("f", g, c, c0)
    for k, info in L["screws"].items():
        for tid, cs in info["tags"].items():
            if int(tid) in BAD_TAGS:
                continue
            cs = np.array(cs, float); c0 = cs.mean(0)
            for j, c in enumerate(cs):
                model[int(tid) * 4 + j] = ("s", int(k), c, c0)
    obs = [o for o in obs if o[1] in model]
    print(f"{len(cams)} views, {len(obs)} corner observations ({sum(model[o[1]][0] == 'f' for o in obs)} frame, {sum(model[o[1]][0] == 's' for o in obs)} screw)")
    seg_of_hole = {k: ("A" if h["part"] == "A" else seg_of(np.array(h["origin"]))) for k, h in enumerate(holes)}
    # init
    Tinit = {"A": np.array(CF["transforms"]["P_mix_A"]), **{g: np.array(T) for g, T in CF["segments"]["T"].items()}}
    segs = ["A", "R", "C", "L"]
    x0 = []
    for g in segs:
        T = Tinit[g]; R = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
        x0 += list(Rotation.from_matrix(R).as_rotvec()) + list(T[:3, 3])
    # screw init from cad_fit S_mix readouts (coarse s) and a theta from its fine value
    rd = CF["variants"]["S_mix"]["readouts"]
    for k in range(12):
        r = rd[k]; s = (r["coarse"] - 74) if r else -40.0
        th = 2 * np.pi * ((r["phase_mm"] - 1.5) % P) / P if r else 0.0   # psi from the phase: gap = 1.5 + 2.5 psi/2pi
        x0 += [s, th]
    x0 += [0.0, 0.0]   # log scale, inset
    x0 += [0.0] * 48   # per screw: tilt a, tilt b (rad), offset u, v (mm)
    x0 = np.array(x0)
    ci = np.array([o[0] for o in obs]); mi = np.array([o[1] for o in obs]); uv = np.array([[o[2], o[3]] for o in obs])
    Rc = np.array([c[0] for c in cams]); Cc = np.array([c[1] for c in cams]); Kc = [c[2] for c in cams]
    fx = np.array([k[0] for k in Kc]); fy = np.array([k[1] for k in Kc]); cx = np.array([k[2] for k in Kc]); cy = np.array([k[3] for k in Kc])
    kd = np.array([k[4] for k in Kc])
    kind = np.array([model[i][0] for i in mi]); ref = [model[i][1] for i in mi]
    loc = np.array([model[i][2] for i in mi]); cen = np.array([model[i][3] for i in mi])

    def unpack(x):
        T = {}
        for j, g in enumerate(segs):
            T[g] = (Rotation.from_rotvec(x[6 * j:6 * j + 3]).as_matrix(), x[6 * j + 3:6 * j + 6])
        sc = x[24:48].reshape(12, 2)
        return T, sc, np.exp(x[48]), x[49], x[50:98].reshape(12, 4)

    def world_points(x):
        T, sc, scale, inset, ax = unpack(x)
        if not a.free_axis:
            ax = np.zeros_like(ax)
        d = loc - cen; nd = np.linalg.norm(d, axis=1, keepdims=True)
        p_loc = cen + d * (1 - inset / np.maximum(nd, 1e-9))          # inset towards the tag centre
        W = np.zeros_like(p_loc)
        for g in segs:
            m = (kind == "f") & np.array([r == g for r in ref], bool)
            R, t = T[g]; W[m] = p_loc[m] @ R.T + t
        for k in range(12):
            m = (kind == "s") & np.array([r == k for r in ref], bool)
            if not m.any():
                continue
            R, t = T[seg_of_hole[k]]; h = holes[k]
            o = np.array(h["origin"], float); dn = np.array(h["dirn"], float); dn /= np.linalg.norm(dn)
            e1 = np.array(E1[k], float); e1 -= (e1 @ dn) * dn; e1 /= np.linalg.norm(e1); e2 = np.cross(dn, e1)
            s, th = sc[k]; xs = np.cos(th) * e1 + np.sin(th) * e2; ys = np.cross(dn, xs)
            Rs = np.c_[xs, ys, dn]
            Rt = Rotation.from_rotvec(ax[k, 0] * e1 + ax[k, 1] * e2).as_matrix()     # tilt about the axis through o
            Rs = Rt @ Rs; ts = o + Rt @ (s * dn) + ax[k, 2] * e1 + ax[k, 3] * e2
            W[m] = (p_loc[m] @ Rs.T + ts) @ R.T + t
        return W, scale

    def residuals(x):
        W, scale = world_points(x)
        Xc = np.einsum("nij,nj->ni", Rc[ci], W - scale * Cc[ci])
        xn = Xc[:, :2] / Xc[:, 2:3]; r2 = (xn ** 2).sum(1, keepdims=True); k = kd[ci]
        xd = xn * (1 + k[:, :1] * r2 + k[:, 1:2] * r2 ** 2 + k[:, 2:3] * r2 ** 3)
        pr = np.c_[fx[ci] * xd[:, 0] + cx[ci], fy[ci] * xd[:, 1] + cy[ci]]
        res = (pr - uv).ravel()
        if a.free_axis:
            ax = x[50:98].reshape(12, 4)
            prior = np.r_[ax[:, :2].ravel() / np.radians(a.tilt_sd), ax[:, 2:].ravel() / a.offset_sd]
            res = np.r_[res, prior]
        return res

    r0 = residuals(x0)
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200, verbose=0)
    r = sol.fun[:2 * len(obs)].reshape(-1, 2); e = np.linalg.norm(r, axis=1)
    T, sc, scale, inset, axv = unpack(sol.x)
    print(f"reprojection median {np.median(np.linalg.norm(r0.reshape(-1, 2), axis=1)):.2f} px -> {np.median(e):.2f} px "
          f"(frame {np.median(e[kind == 'f']):.2f}, screws {np.median(e[kind == 's']):.2f}); scale {scale:.5f}, inset {inset:.3f} mm")
    # uncertainty of each screw's s, theta from the Gauss-Newton covariance
    J = sol.jac; res_var = np.median(e) ** 2 / 0.4549 / 2   # robust variance per coordinate
    try:
        cov = np.linalg.pinv(J.T @ J) * res_var
    except Exception:
        cov = None
    ruler = [float(v) for v in a.ruler.split(",")] if a.ruler else None
    rows = []
    for k in range(12):
        s, th = sc[k]
        psi = np.arctan2(np.sin(th), np.cos(th)); sf = P * psi / (2 * np.pi)
        sfine = sf + np.round((s - sf) / P) * P
        sd_s = float(np.sqrt(cov[24 + 2 * k, 24 + 2 * k])) if cov is not None else None
        sd_th = float(np.degrees(np.sqrt(cov[25 + 2 * k, 25 + 2 * k]))) if cov is not None else None
        nobs = int(((kind == "s") & np.array([r_ == k for r_ in ref], bool)).sum())
        rows.append({"hole": k, "coarse": float(s + 74), "fine": float(sfine + 74), "phase_mm": float((sf + 74) % P),
                     "sd_coarse_mm": sd_s, "sd_theta_deg": sd_th, "obs": nobs,
                     "tilt_deg": float(np.degrees(np.hypot(*axv[k, :2]))), "offset_mm": float(np.hypot(*axv[k, 2:]))})
    out = {"scale": float(scale), "inset_mm": float(inset), "reproj_px": {"median": float(np.median(e)), "frame": float(np.median(e[kind == "f"])),
           "screws": float(np.median(e[kind == "s"]))}, "readouts": rows,
           "segments": {g: {"R": T[g][0].tolist(), "t": T[g][1].tolist()} for g in segs}}
    if ruler:
        g = np.array(ruler); co = np.array([r_["coarse"] for r_ in rows]); fi = np.array([r_["fine"] for r_ in rows])
        out["coarse_err"] = (co - g).round(2).tolist(); out["fine_err"] = (fi - g).round(2).tolist()

        def predict(tr, te):
            b = np.median(co[tr] - g[tr]); w = ((fi[tr] - g[tr] + P / 2) % P) - P / 2; phi = np.median(w)
            f = fi[te] - phi; n = np.round(((co[te] - b) - f) / P); return f + n * P, b, phi
        loo = np.array([predict(np.delete(np.arange(12), i), np.array([i]))[0][0] for i in range(12)])
        _, b, phi = predict(np.arange(12), np.arange(12))
        out["calibration"] = {"b_mm": float(b), "phi_mm": float(phi), "loo_err": (loo - g).round(2).tolist(),
                              "loo_mae": float(np.mean(np.abs(loo - g))), "loo_max": float(np.max(np.abs(loo - g)))}
        print(f"coarse err {out['coarse_err']}  (MAE {np.mean(np.abs(co - g)):.2f}, bias {np.median(co - g):+.2f})")
        print(f"fine   err {out['fine_err']}  (MAE {np.mean(np.abs(fi - g)):.2f}, max {np.max(np.abs(fi - g)):.2f})")
        print(f"calibrated LOO {out['calibration']['loo_err']}  (MAE {out['calibration']['loo_mae']:.2f}, max {out['calibration']['loo_max']:.2f}; b {b:.2f}, phi {phi:.2f})")
    print("tilt deg / offset mm:", [(round(r_["tilt_deg"], 1), round(r_["offset_mm"], 1)) for r_ in rows])
    print("per-screw sd (coarse mm / angle deg):", [(round(r_["sd_coarse_mm"], 2), round(r_["sd_theta_deg"], 1)) for r_ in rows])
    json.dump(out, open(O / "image_fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
