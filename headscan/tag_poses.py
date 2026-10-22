"""Tag-only camera poses in the device frame: per-view PnP on the frame-tag corners, then one bundle adjustment over
those corners only (no SIFT, no triangulation). Replaces SfM + tag landmarks + sfmAlignment from_markers.

    python tag_poses.py CAPTURE_DIR VIEWS.sfm FEAT_TAGS_DIR LAYOUT.json OUT.sfm [--segments] [--intrinsics]
        [--arkit-prior] [--min-tags 2]

VIEWS.sfm: any sfmData whose 'views' list the keyframes (viewId = frame index, e.g. the cameraInit or sfm_device of
the SfM path); only the views are used. FEAT_TAGS_DIR: <viewId>.tag36h11.feat/.desc (AliceVision tag36h11
describer, index = tagId*4 + layout corner). Only the rigid frame tags 0-13 constrain the poses.
Unknowns (device frame, mm, scale fixed by the CAD layout):
  - per view: rotation (3) + camera centre (3); init from PnP (>= --min-tags tags), else ARKit through a similarity
    fitted to the PnP views
  - one tag inset delta (mm, detector bias: every corner moves delta towards its tag centre)
  - --intrinsics: shared focal, principal point, radialk3 (init ARKit median + inverse Apple LUT, like stray_to_sfm.py)
  - --segments: rigid pose of each frame segment (part A, right arch, apex block, left arch, as in image_fit.py),
    prior 1 mm / 0.5 deg towards the CAD (that prior is the gauge: the device frame = least-squares mean of the CAD)
  - --arkit-prior: relative pose of consecutive views from ARKit (0.3 deg, 3 mm, ARKit scale free); keeps views
    without tags and bridges gaps
Residuals: corner reprojection (px), soft-L1 (1 px) on everything (the ARKit prior is robust to its jumps).
Writes OUT.sfm (views + one intrinsic + poses, metres, same layout as sfm_device.sfm) and OUT.json (statistics).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stray_to_sfm import H, SW, W, fit_lut_radialk3, read_csv  # noqa: E402

FRAME_TAGS = range(14)
SEGS = ["A", "R", "C", "L"]


def seg_of(tid, c0):
    return "A" if tid <= 6 else ("R" if c0[0] > 25 else ("L" if c0[0] < -25 else "C"))


def read_tags(feat_dir, vid):
    f = Path(feat_dir) / f"{vid}.tag36h11.feat"
    if not f.exists():
        return np.zeros((0, 2)), np.zeros(0, int)
    pts = np.loadtxt(f, ndmin=2)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    raw = (Path(feat_dir) / f"{vid}.tag36h11.desc").read_bytes()
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    return pts[:, :2], np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256).argmax(1)


def distort(xn, k):
    r2 = (xn ** 2).sum(-1, keepdims=True)
    return xn * (1 + k[..., 0:1] * r2 + k[..., 1:2] * r2 ** 2 + k[..., 2:3] * r2 ** 3)


def undistort(uv, f, c, k, iters=20):
    xd = (uv - c) / f; xu = xd.copy()
    for _ in range(iters):
        r2 = (xu ** 2).sum(1, keepdims=True)
        xu = xd / (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return xu


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (X - mx).var(0).sum()
    return s, R, my - s * R @ mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("views"); ap.add_argument("feat"); ap.add_argument("layout"); ap.add_argument("out")
    ap.add_argument("--segments", action="store_true"); ap.add_argument("--intrinsics", action="store_true")
    ap.add_argument("--arkit-prior", action="store_true"); ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--perframe-k", action="store_true", help="per-view ARKit fx, fy, cx, cy (shared corrections only)")
    ap.add_argument("--seg-sd", default="1.0,0.5", help="segment prior: mm, deg")
    ap.add_argument("--arkit-sd", default="3.0,0.3", help="ARKit relative-pose prior: mm, deg")
    a = ap.parse_args()
    t0 = time.time()
    cap = Path(a.capture)
    L = json.load(open(a.layout))
    # model: marker index -> (segment, corner, tag centre), device mm
    model = {}
    for t in FRAME_TAGS:
        cs = np.array(L["frame"][str(t)], float); c0 = cs.mean(0)
        for k, c in enumerate(cs):
            model[t * 4 + k] = (seg_of(t, c0), c, c0)
    views = sorted(json.load(open(a.views))["views"], key=lambda v: int(v["frameId"]))
    odo = {int(r["frame"]): r for r in read_csv(cap / "odometry.csv")}
    sel = [odo[int(v["frameId"])] for v in views]
    fx0, fy0, cx0, cy0 = (float(np.median([float(r[k]) for r in sel])) for k in ("fx", "fy", "cx", "cy"))
    k0 = fit_lut_radialk3(cap, fy0, cx0, cy0, "inverse_distortion")[0]
    # per-view ARKit intrinsics (--perframe-k: the principal point moves by up to ~30 px within a capture) or the median
    Kv = np.array([[float(r[q]) for q in ("fx", "fy", "cx", "cy")] if a.perframe_k else [fx0, fy0, cx0, cy0] for r in sel])
    # observations
    obs = []   # (view index, marker index, u, v)
    ntag = np.zeros(len(views), int)
    for i, v in enumerate(views):
        uv, idx = read_tags(a.feat, v["viewId"])
        keep = [j for j, m in enumerate(idx) if m in model]
        ntag[i] = len({int(idx[j]) // 4 for j in keep})
        obs += [(i, int(idx[j]), *uv[j]) for j in keep]
    vi = np.array([o[0] for o in obs]); mi = np.array([o[1] for o in obs]); uv = np.array([[o[2], o[3]] for o in obs])
    seg = np.array([SEGS.index(model[m][0]) for m in mi]); loc = np.array([model[m][1] for m in mi]); cen = np.array([model[m][2] for m in mi])
    dloc = loc - cen; nd = np.linalg.norm(dloc, axis=1, keepdims=True)
    print(f"{len(views)} views, {(ntag > 0).sum()} with frame tags, {(ntag >= a.min_tags).sum()} with >= {a.min_tags}; "
          f"{len(obs)} corner observations; intrinsics f {fy0:.1f}, k {np.round(k0, 4).tolist()}")

    # 1. PnP per view (undistorted normalised points, CAD corners)
    K1 = np.eye(3)
    pnp = {}
    for i in range(len(views)):
        m = vi == i
        if ntag[i] < a.min_tags:
            continue
        xu = undistort(uv[m], Kv[i, :2], Kv[i, 2:], k0)
        ok, rv, tv, inl = cv2.solvePnPRansac(loc[m], xu, K1, None, reprojectionError=4.0 / fy0, iterationsCount=200,
                                             flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < 6:
            continue
        rv, tv = cv2.solvePnPRefineLM(loc[m][inl[:, 0]], xu[inl[:, 0]], K1, None, rv, tv)
        R = cv2.Rodrigues(rv)[0]
        pr = (loc[m] @ R.T + tv.ravel()); pr = pr[:, :2] / pr[:, 2:3]
        e = np.median(np.linalg.norm(pr - xu, axis=1)) * fy0
        if e < 3.0:
            pnp[i] = (R, -R.T @ tv.ravel(), e)
    print(f"PnP: {len(pnp)} views, median reprojection {np.median([p[2] for p in pnp.values()]):.2f} px")

    # 2. ARKit poses (camera-to-world, OpenCV camera) -> device frame by a similarity fitted on the PnP views
    RA = np.array([Rotation.from_quat([float(r[c]) for c in ("qx", "qy", "qz", "qw")]).as_matrix() for r in sel])   # R_wc
    CA = np.array([[float(r[c]) for c in ("x", "y", "z")] for r in sel])
    ids = sorted(pnp)
    s, Rs, ts = umeyama(CA[ids], np.array([pnp[i][1] for i in ids]))
    for _ in range(2):   # trim > 3x median and refit (ARKit drift)
        res = np.linalg.norm(s * CA[ids] @ Rs.T + ts - np.array([pnp[i][1] for i in ids]), axis=1)
        good = [i for i, r_ in zip(ids, res) if r_ < 3 * np.median(res) + 1]
        s, Rs, ts = umeyama(CA[good], np.array([pnp[i][1] for i in good]))
    # rotation offset: average of R_cw(PnP) vs (Rs R_wc(ARKit))^T
    dR = Rotation.from_matrix(np.array([pnp[i][0] @ Rs @ RA[i] for i in good])).mean().as_matrix()
    print(f"ARKit -> device: scale {s:.2f} mm/m, centre residual median {np.median(res):.1f} mm, "
          f"rotation offset {np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())):.2f} deg")
    x_views = np.zeros((len(views), 6))
    src = []
    for i in range(len(views)):
        if i in pnp:
            R, C = pnp[i][:2]; src.append("pnp")
        else:
            R = dR @ (Rs @ RA[i]).T; C = s * Rs @ CA[i] + ts; src.append("arkit")
        x_views[i] = np.r_[Rotation.from_matrix(R).as_rotvec(), C]
    if not a.arkit_prior:   # views without tags cannot be posed
        posed = np.array([i in pnp for i in range(len(views))])
    else:
        posed = np.ones(len(views), bool)

    # 3. bundle adjustment
    nv = len(views); I_IN = 6 * nv; I_SEG = I_IN + 6; I_DEL = I_SEG + 24; I_LS = I_DEL + 1; NP = I_LS + 1
    seg_sd = [float(v) for v in a.seg_sd.split(",")]; ark_sd = [float(v) for v in a.arkit_sd.split(",")]
    x0 = np.r_[x_views.ravel(), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, np.zeros(24), 0.0, np.log(s)]
    k_base = np.array(k0)
    pairs = [(i, i + 1) for i in range(nv - 1)] if a.arkit_prior else []
    # ARKit relative motion in camera i: R_ij = R_ci_w R_cj_w^T ; t_ij = R_ci_w (C_j - C_i)
    RcwA = np.transpose(RA, (0, 2, 1))
    pi = np.array([p[0] for p in pairs], int); pj = np.array([p[1] for p in pairs], int)
    relR = RcwA[pi] @ RA[pj]; relt = np.einsum("nij,nj->ni", RcwA[pi], CA[pj] - CA[pi])
    obs_m = posed[vi]
    nres_obs = 2 * obs_m.sum()

    def unpack(x):
        xv = x[:I_IN].reshape(nv, 6)
        ds, pr, dcx, dcy, dk1, dk2 = x[I_IN:I_SEG]   # shared corrections of the ARKit intrinsics
        return (xv, (Kv[:, 0] * (1 + ds) / pr, Kv[:, 1] * (1 + ds), Kv[:, 2] + dcx, Kv[:, 3] + dcy, k_base * np.array([1 + dk1, 1 + dk2, 1.0])),
                x[I_SEG:I_DEL].reshape(4, 6), x[I_DEL], x[I_LS])

    def residuals(x):
        xv, (fx, fy, cx, cy, k), sg, dl, ls = unpack(x)
        p = cen + dloc * (1 - dl / np.maximum(nd, 1e-9))
        Rg = Rotation.from_rotvec(sg[:, :3]).as_matrix()
        Wp = np.einsum("nij,nj->ni", Rg[seg], p) + sg[seg, 3:]
        m = obs_m; v_ = vi[m]
        Rc = Rotation.from_rotvec(xv[v_, :3]).as_matrix()
        Xc = np.einsum("nij,nj->ni", Rc, Wp[m] - xv[v_, 3:])
        xd = distort(Xc[:, :2] / Xc[:, 2:3], np.broadcast_to(k, (len(Xc), 3)))
        r = (np.c_[fx[v_] * xd[:, 0] + cx[v_], fy[v_] * xd[:, 1] + cy[v_]] - uv[m]).ravel()
        out = [r]
        if a.segments:
            out.append(np.r_[np.degrees(sg[:, :3]).ravel() / seg_sd[1], sg[:, 3:].ravel() / seg_sd[0]])
        if pairs:
            Rv = Rotation.from_rotvec(xv[:, :3]).as_matrix()
            dRr = Rotation.from_matrix(Rv[pi] @ np.transpose(Rv[pj], (0, 2, 1)) @ np.transpose(relR, (0, 2, 1))).as_rotvec()
            dt = np.einsum("nij,nj->ni", Rv[pi], xv[pj, 3:] - xv[pi, 3:]) - np.exp(ls) * relt
            out.append(np.c_[np.degrees(dRr) / ark_sd[1], dt / ark_sd[0]].ravel())
        return np.concatenate(out)

    # sparsity
    nres = nres_obs + (24 if a.segments else 0) + 6 * len(pairs)
    S = lil_matrix((nres, NP), dtype=np.int8)
    rows = np.arange(nres_obs // 2)
    vm, sm = vi[obs_m], seg[obs_m]
    for r_ in (0, 1):
        for c in range(6):
            S[2 * rows + r_, 6 * vm + c] = 1
            S[2 * rows + r_, I_SEG + 6 * sm + c] = 1
        for c in range(6):
            S[2 * rows + r_, I_IN + c] = 1
        S[2 * rows + r_, I_DEL] = 1
    o = nres_obs
    if a.segments:
        for c in range(24):
            S[o + c, I_SEG + c] = 1
        o += 24
    for q, (i, j) in enumerate(pairs):
        for r_ in range(6):
            for c in range(6):
                S[o + 6 * q + r_, 6 * i + c] = 1; S[o + 6 * q + r_, 6 * j + c] = 1
            S[o + 6 * q + r_, I_LS] = 1
    # fixed blocks: columns removed through bounds-free masking (active set)
    free = np.ones(NP, bool)
    free[:I_IN][~np.repeat(posed, 6)] = False
    if not a.intrinsics:
        free[I_IN:I_SEG] = False
    if not a.segments:
        free[I_SEG:I_DEL] = False
    if not pairs:
        free[I_LS] = False

    def res_free(xf):
        x = x0.copy(); x[free] = xf; return residuals(x)

    r0 = residuals(x0)[:nres_obs].reshape(-1, 2)
    sol = least_squares(res_free, x0[free], jac_sparsity=S.tocsr()[:, free], loss="soft_l1", f_scale=1.0,
                        x_scale="jac", method="trf", tr_solver="lsmr", max_nfev=100, verbose=0)
    x = x0.copy(); x[free] = sol.x
    rr = residuals(x)[:nres_obs].reshape(-1, 2); e = np.linalg.norm(rr, axis=1); e0 = np.linalg.norm(r0, axis=1)
    xv, (fx, fy, cx, cy, k), sg, dl, ls = unpack(x)
    # gauge: move everything so the reconstructed frame-tag corners fit the CAD rigidly (the from_markers definition of
    # the device frame, without its free scale: the tags set the scale here)
    gauge = (0.0, 0.0)
    if a.segments:
        ms = sorted(set(mi)); cad = np.array([model[m][1] for m in ms]); gi = np.array([SEGS.index(model[m][0]) for m in ms])
        Rg = Rotation.from_rotvec(sg[gi, :3]).as_matrix(); rec = np.einsum("nij,nj->ni", Rg, cad) + sg[gi, 3:]
        mx, my = rec.mean(0), cad.mean(0); U, _, Vt = np.linalg.svd((cad - my).T @ (rec - mx))
        D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rf = U @ D @ Vt; tf = my - Rf @ mx   # rec -> cad
        for j in range(4):
            R_ = Rf @ Rotation.from_rotvec(sg[j, :3]).as_matrix(); sg[j, :3] = Rotation.from_matrix(R_).as_rotvec(); sg[j, 3:] = Rf @ sg[j, 3:] + tf
        Rv = Rotation.from_rotvec(xv[:, :3]).as_matrix() @ Rf.T
        xv[:, :3] = Rotation.from_matrix(Rv).as_rotvec(); xv[:, 3:] = xv[:, 3:] @ Rf.T + tf
        gauge = (float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rf).as_rotvec()))), float(np.linalg.norm(tf)))
        print(f"gauge: frame corners -> CAD by {gauge[0]:.2f} deg, {gauge[1]:.2f} mm")
    dt = time.time() - t0
    per_view = np.full(nv, np.nan)
    for i in range(nv):
        m_ = vi[obs_m] == i
        if m_.any():
            per_view[i] = np.median(e[m_])
    print(f"BA: reprojection median {np.median(e0):.2f} -> {np.median(e):.2f} px (p90 {np.percentile(e, 90):.2f}); "
          f"inset {dl:.3f} mm; f {np.median(fy):.1f} (ARKit {fy0:.1f}), pp {np.median(cx):.1f},{np.median(cy):.1f}; ARKit scale {np.exp(ls):.2f} mm/m; {dt:.1f} s")
    if a.segments:
        print("segments (deg, mm):", {g: (round(float(np.degrees(np.linalg.norm(sg[j, :3]))), 3), round(float(np.linalg.norm(sg[j, 3:])), 2)) for j, g in enumerate(SEGS)})
    # 4. write sfmData (metres) + stats
    def intrinsic(iid, fx, fy, cx, cy):
        return {"intrinsicId": str(iid), "width": str(W), "height": str(H), "sensorWidth": str(SW), "sensorHeight": str(SW * H / W),
                "serialNumber": f"tag_poses_{iid}", "type": "pinhole", "initializationMode": "calibrated",
                "initialFocalLength": str(fy0 * SW / max(W, H)), "focalLength": repr(float(fy * SW / max(W, H))),
                "pixelRatio": repr(float(fy / fx)), "pixelRatioLocked": "true", "scaleLocked": "true", "offsetLocked": "true",
                "principalPoint": [repr(float(cx - W / 2)), repr(float(cy - H / 2))], "distortionType": "radialk3",
                "distortionParams": [repr(float(v)) for v in k], "distortionLocked": "true", "distortionInitializationMode": "calibrated",
                "undistortionType": "none", "undistortionOffset": ["0", "0"], "undistortionParams": [], "locked": "true"}
    intr = [intrinsic(1, *(np.median(q) for q in (fx, fy, cx, cy)))] if not a.perframe_k else []
    vout, pout = [], []
    for i, v in enumerate(views):
        if not posed[i]:
            continue
        v = dict(v); v["intrinsicId"] = v["viewId"] if a.perframe_k else "1"; v["poseId"] = v["viewId"]; vout.append(v)
        if a.perframe_k:
            intr.append(intrinsic(v["viewId"], fx[i], fy[i], cx[i], cy[i]))
        R = Rotation.from_rotvec(xv[i, :3]).as_matrix()
        pout.append({"poseId": v["viewId"], "pose": {"transform": {"rotation": [repr(float(q)) for q in R.flatten(order="F")],
                     "center": [repr(float(q) / 1000.0) for q in xv[i, 3:]]}, "locked": "1"}})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": ["1", "2", "14"], "views": vout, "intrinsics": intr, "poses": pout}, open(a.out, "w"), indent=1)
    stats = {"views": nv, "posed": int(posed.sum()), "pnp": len(pnp), "with_tags": int((ntag > 0).sum()), "obs": int(obs_m.sum()),
             "reproj_px": {"init_median": float(np.median(e0)), "median": float(np.median(e)), "p90": float(np.percentile(e, 90))},
             "inset_mm": float(dl), "focal_px": float(np.median(fy)), "focal_arkit_px": fy0, "pp": [float(np.median(cx)), float(np.median(cy))], "k": [float(q) for q in k],
             "arkit_mm_per_m": float(np.exp(ls)), "segments": {g: {"rotvec_deg": np.degrees(sg[j, :3]).tolist(), "t_mm": sg[j, 3:].tolist()} for j, g in enumerate(SEGS)},
             "gauge_deg_mm": gauge, "seconds": dt, "options": vars(a),
             "per_view": [{"viewId": views[i]["viewId"], "tags": int(ntag[i]), "init": src[i], "reproj_px": None if np.isnan(per_view[i]) else float(per_view[i])} for i in range(nv)]}
    json.dump(stats, open(Path(a.out).with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
