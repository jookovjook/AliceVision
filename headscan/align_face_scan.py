"""Align a TrueDepth face scan (StandardCyborg .ply, camera frame, metres) to the reconstructed head (device frame).

    python align_face_scan.py SCAN.ply REF.ply OUT_DIR

Hair, clothes and the frame differ between the two scans, so only the FACE is used:
  1. both point sets are rendered from the front (z-buffered point splats), MediaPipe FaceMesh runs on several
     yaw/pitch offsets, landmarks are lifted to 3-D from the render's point-index buffer, median over renders;
  2. rigid Umeyama on the matched landmarks (scale reported, not applied: both are metric);
  3. trimmed point-to-plane ICP of the scan's face (within 90 mm of the nose tip, below the forehead landmark +15 mm,
     in front of the ears, not below the chin) onto the reference surface.
REF.ply: device frame (origin ear-axis midpoint, x right, y forward, z up), metres, vertex colours.
Writes OUT_DIR/align.json (transform scan -> device, residuals, landmark errors), scan_device.ply, figures.
Needs mediapipe 0.10.21 (legacy solutions API).
"""
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from scipy.spatial import cKDTree

LM = {"between_eyes": 168, "nose_bridge": 6, "glabella": 9, "forehead": 10, "nose_tip": 1, "subnasale": 2, "chin": 152,
      "eye_R_outer": 33, "eye_R_inner": 133, "eye_L_inner": 362, "eye_L_outer": 263, "mouth_R": 61, "mouth_L": 291,
      "face_R": 234, "face_L": 454}
W, H, F = 900, 1100, 1900.0
ARC = "/Users/jack/aspect/tfus-simulation/.claude/worktrees/vastai-cuda-setup-69d92b/results/arc_redesign/tags"
FRAME_STL = [f"{ARC}/frame_L116_A_plates.stl", f"{ARC}/frame_L116_B_plates.stl"]


def load_points(path):
    m = trimesh.load(path, process=False)
    P = np.asarray(m.vertices, float)
    C = np.asarray(m.visual.vertex_colors)[:, :3] if hasattr(m, "visual") and getattr(m.visual, "vertex_colors", None) is not None else np.full((len(P), 3), 180)
    N = np.asarray(m.vertex_normals) if isinstance(m, trimesh.Trimesh) and len(m.faces) else None
    return P, C.astype(np.uint8), N, m


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, -u, f])  # rows: camera x (right), y (down), z (forward)


def render(P, C, eye, target, up, rad=2):
    R = look_at(eye, target, up); X = (P - eye) @ R.T
    ok = X[:, 2] > 0.05; idx = np.nonzero(ok)[0]; X = X[ok]
    u = (F * X[:, 0] / X[:, 2] + W / 2).round().astype(int); v = (F * X[:, 1] / X[:, 2] + H / 2).round().astype(int)
    zb = np.full((H, W), np.inf); ib = np.full((H, W), -1)
    order = np.argsort(-X[:, 2])  # far first, near overwrites
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu, vv = u[order] + du, v[order] + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv, z, ii = uu[m], vv[m], X[order][m, 2], idx[order][m]
            closer = z < zb[vv, uu]
            # sequential overwrite in far->near order keeps the nearest
            zb[vv[closer], uu[closer]] = z[closer]; ib[vv[closer], uu[closer]] = ii[closer]
    img = np.full((H, W, 3), 255, np.uint8); hit = ib >= 0; img[hit] = C[ib[hit]]
    return img, ib


def landmarks(P, C, centre, fwd, up, tag, out):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=False, max_num_faces=1)
    right = np.cross(fwd, up); res = {k: [] for k in LM}; nok = 0
    for yaw in (-20, -10, 0, 10, 20):
        for pitch in (-10, 0, 10):
            a, b = np.radians(yaw), np.radians(pitch)
            d = np.cos(b) * (np.cos(a) * fwd + np.sin(a) * right) + np.sin(b) * up
            img, ib = render(P, C, centre + 0.5 * d, centre, up)
            r = fm.process(np.ascontiguousarray(img))
            if not r.multi_face_landmarks:
                continue
            nok += 1
            if yaw == 0 and pitch == 0:
                vis = img[..., ::-1].copy()
            for k, i in LM.items():
                p = r.multi_face_landmarks[0].landmark[i]; x, y = int(p.x * W), int(p.y * H)
                win = ib[max(0, y - 2):y + 3, max(0, x - 2):x + 3].ravel(); win = win[win >= 0]
                if len(win):
                    res[k].append(np.median(P[win], 0))
                if yaw == 0 and pitch == 0:
                    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
    if nok and "vis" in locals():
        cv2.imwrite(str(out / f"landmarks_{tag}.jpg"), vis)
    print(f"{tag}: FaceMesh found the face in {nok}/15 renders")
    return {k: (np.median(v, 0), float(np.median(np.linalg.norm(np.array(v) - np.median(v, 0), axis=1)))) for k, v in res.items() if v}


def face_frame(L):
    """Face centre, forward and up from 3-D landmarks (subject's right = eye_R_*)."""
    eR, eL = (L["eye_R_outer"][0] + L["eye_R_inner"][0]) / 2, (L["eye_L_outer"][0] + L["eye_L_inner"][0]) / 2
    across = eL - eR; across /= np.linalg.norm(across)
    up = (eR + eL) / 2 - L["chin"][0]; up -= (up @ across) * across; up /= np.linalg.norm(up)
    fwd = np.cross(up, across)
    if fwd @ (L["nose_tip"][0] - (eR + eL) / 2) < 0:
        fwd = -fwd
    return L["nose_tip"][0] - 0.07 * fwd + 0.02 * up, fwd, up


def umeyama(X, Y, scale=False):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (X - mx).var(0).sum() if scale else 1.0
    return s * R, my - s * R @ mx, s


def main():
    scan, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    Ps, Cs, _, ms = load_points(scan)
    Pr, Cr, Nr, mr = load_points(ref)
    # scan: camera at the origin looking at the cloud; image up = -y (camera y down)
    cs = np.median(Ps, 0); fs = -cs / np.linalg.norm(cs)
    best = None
    for up in (np.array([0, -1.0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])):
        upo = up - (up @ fs) * fs; upo /= np.linalg.norm(upo)
        L = landmarks(Ps, Cs, cs, fs, upo, "scan", out)
        if len(L) > (len(best[0]) if best else 0):
            best = (L, upo)
    Ls = best[0]
    for _ in range(2):   # re-render square to the face found so far (the scan's first camera may see it from the side)
        c, fwd, up = face_frame(Ls)
        Ls = landmarks(Ps, Cs, c, fwd, up, "scan", out) or Ls
    # reference: device frame, face forward = +y, up = +z, centred on the face
    cr = np.array([0.0, 0.08, -0.07])
    Lr = landmarks(Pr, Cr, cr, np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), "ref", out)
    # the frame's nose bar covers between_eyes / nose_bridge / glabella / forehead on the reference; contour points slide
    keys = [k for k in LM if k in Ls and k in Lr and k not in ("face_R", "face_L", "between_eyes", "nose_bridge", "glabella", "forehead")]
    X = np.array([Ls[k][0] for k in keys]); Y = np.array([Lr[k][0] for k in keys])
    R, t, _ = umeyama(X, Y)
    _, _, s = umeyama(X, Y, scale=True)
    lm_err = {k: float(np.linalg.norm(X[i] @ R.T + t - Y[i]) * 1000) for i, k in enumerate(keys)}
    print(f"landmarks ({len(keys)}): rigid RMS {np.sqrt(np.mean(np.square(list(lm_err.values())))):.1f} mm; similarity scale {s:.4f}")
    # ICP on the face
    Lnose = Lr["nose_tip"][0]
    eye_z = np.mean([Lr[k][0][2] for k in ("eye_R_outer", "eye_R_inner", "eye_L_inner", "eye_L_outer")])
    if Nr is None:
        Nr = trimesh.Trimesh(Pr).vertex_normals
    # drop reference points on or near the frame (nominal CAD in the device frame, mm -> m; 10 mm margin for the bend)
    keep_ref = np.ones(len(Pr), bool)
    for f in FRAME_STL:
        cad = trimesh.load(f); sp = trimesh.sample.sample_surface_even(cad, 60000, seed=0)[0] / 1000.0
        dd, _ = cKDTree(sp).query(Pr, distance_upper_bound=0.011); keep_ref &= ~np.isfinite(dd)
    print(f"reference: {np.sum(~keep_ref)} of {len(Pr)} points dropped near the frame")
    Pr, Nr = Pr[keep_ref], Nr[keep_ref]
    tree = cKDTree(Pr)
    sub = Ps[np.random.default_rng(0).choice(len(Ps), min(60000, len(Ps)), replace=False)]

    def face_mask(Q):
        return (np.linalg.norm(Q - Lnose, axis=1) < 0.09) & (Q[:, 2] < eye_z + 0.035) & (Q[:, 1] > 0.035) & \
               (Q[:, 2] > Lr["chin"][0][2] - 0.01)
    for it in range(40):
        Q = sub @ R.T + t; m = face_mask(Q)
        d, j = tree.query(Q[m]); keep = d < np.quantile(d, 0.85) if it > 2 else d < 0.01
        A, Nn, Bp = Q[m][keep], Nr[j[keep]], Pr[j[keep]]
        r = ((Bp - A) * Nn).sum(1); J = np.c_[np.cross(A, Nn), Nn]
        x = np.linalg.lstsq(J, r, rcond=None)[0]
        dR = trimesh.transformations.rotation_matrix(np.linalg.norm(x[:3]), x[:3] / (np.linalg.norm(x[:3]) + 1e-12))[:3, :3]
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x) < 1e-7:
            break
    Q = Ps @ R.T + t; m = face_mask(Q); d, j = tree.query(Q[m]); dn = np.abs(((Q[m] - Pr[j]) * Nr[j]).sum(1))
    lm_icp = {k: float(np.linalg.norm(Ls[k][0] @ R.T + t - Lr[k][0]) * 1000) for k in LM if k in Ls and k in Lr}
    res = {"T_scan_to_device": np.r_[np.c_[R, t], [[0, 0, 0, 1]]].tolist(), "landmarks_used": keys,
           "landmark_err_mm_after_icp": lm_icp, "landmark_similarity_scale": s,
           "face_icp": {"points": int(m.sum()), "median_mm": float(np.median(dn) * 1000), "p90_mm": float(np.percentile(dn, 90) * 1000),
                        "rms_trim85_mm": float(np.sqrt(np.mean(np.sort(dn)[:int(0.85 * len(dn))] ** 2)) * 1000)}}
    # ear-to-ear check: scan points near the reference ear balls (where the knobs press)
    print(json.dumps(res["face_icp"]), "landmarks after ICP (mm):", {k: round(v, 1) for k, v in lm_icp.items()})
    json.dump(res, open(out / "align.json", "w"), indent=1)
    allD, _ = tree.query(Q)
    trimesh.PointCloud(Q, colors=np.c_[Cs, np.full(len(Cs), 255)]).export(out / "scan_device.ply")
    np.save(out / "scan_dist_to_ref_mm.npy", (allD * 1000).astype(np.float32))


if __name__ == "__main__":
    main()
