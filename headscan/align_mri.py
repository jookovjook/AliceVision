"""DIRECT fit (kept for reference: it slides, see mri_via_phone.py for the method in use). Align Jack's MRI skin (tfus-simulation results/me/viewer.npz, charm skin + skull, head frame mm) to a reconstructed
head (device frame, metres) by the face only.

    python align_mri.py VIEWER.npz REF.ply OUT_DIR

1. start: rigid Umeyama on MediaPipe landmarks: the MRI's own (L_mri, stored with the skin) vs the reconstruction's
   (rendered + FaceMesh, as align_face_scan.py); landmarks the frame covers on the reconstruction are left out;
2. trimmed point-to-plane ICP of the MRI skin's reliable face (within 70 mm of the nose tip: nose, eyes, brows, lower
   forehead, upper lip; not the cheeks and ears the coil pads pressed, not above the brows + 35 mm where hair is)
   onto the reconstruction without the frame; then the same with a free scale (reported, not applied: both are metric).
Writes OUT_DIR/align_mri.json (T MRI head frame mm -> device m, residuals, scale) and skin_device.ply / bone_device.ply.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_face_scan import FRAME_STL, LM, landmarks, load_points, umeyama  # noqa: E402

KEYS = ["nose_tip", "subnasale", "eye_R_outer", "eye_R_inner", "eye_L_inner", "eye_L_outer", "mouth_R", "mouth_L"]


def icp(src, tree, Pr, Nr, R, t, scale=False, iters=60):
    """Trimmed point-to-plane ICP (linearised about the origin, as align_face_scan.py); optional log-scale about the
    source centroid. Returns R, t, s and the point-to-plane distances of the final pose."""
    s = 1.0
    for it in range(iters):
        c = (src @ R.T + t).mean(0)
        X = s * (src @ R.T + t - c) + c
        d, j = tree.query(X)
        gate = 0.02 if it < 10 else (0.01 if it < 20 else np.quantile(d, 0.85))   # wide first, then trimmed
        keep = d < gate
        if keep.sum() < 100:
            break
        A, N, B = X[keep], Nr[j[keep]], Pr[j[keep]]
        r = ((B - A) * N).sum(1)
        J = np.c_[np.cross(A, N), N]
        if scale:
            J = np.c_[J, ((A - c) * N).sum(1)]
        x = np.linalg.lstsq(J, r, rcond=None)[0]
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        R, t = dR @ R, dR @ t + x[3:6]
        if scale:
            s *= np.exp(x[6])
        if np.linalg.norm(x) < 1e-8:
            break
    c = (src @ R.T + t).mean(0); X = s * (src @ R.T + t - c) + c; d, j = tree.query(X)
    return R, t, s, np.abs(((X - Pr[j]) * Nr[j]).sum(1))


def main():
    vnpz, ref, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    V = np.load(vnpz)
    Sv, Sf = V["charm_v"].astype(float) / 1000.0, V["charm_f"]; Bv, Bf = V["bone_v"].astype(float) / 1000.0, V["bone_f"]
    Lm = {k: V["L_mri"][i] / 1000.0 for k, i in LM.items() if np.all(np.isfinite(V["L_mri"][i]))}
    Pr, Cr, Nr, _ = load_points(ref)
    Lr = landmarks(Pr, Cr, np.array([0.0, 0.08, -0.07]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), "ref", out)
    # MRI reliable face (in its own frame, from its landmarks)
    nose = Lm["nose_tip"]; eye_z = np.mean([Lm[k][2] for k in KEYS[2:6]])
    mface = (np.linalg.norm(Sv - nose, axis=1) < 0.07) & (Sv[:, 2] < eye_z + 0.035) & (Sv[:, 1] > nose[1] - 0.06)
    src = Sv[mface]
    keys = [k for k in KEYS if k in Lm and k in Lr]
    R, t, _ = umeyama(np.array([Lm[k] for k in keys]), np.array([Lr[k][0] for k in keys]))
    lm0 = {k: float(np.linalg.norm(R @ Lm[k] + t - Lr[k][0]) * 1e3) for k in keys}
    print(f"start from {len(keys)} landmarks: RMS {np.sqrt(np.mean(np.square(list(lm0.values())))):.1f} mm")
    # reference without the frame (nominal CAD + 11 mm)
    if Nr is None:
        Nr = trimesh.Trimesh(Pr).vertex_normals
    keep = np.ones(len(Pr), bool)
    for f in FRAME_STL:
        sp = trimesh.sample.sample_surface_even(trimesh.load(f), 60000, seed=0)[0] / 1000.0
        dd, _ = cKDTree(sp).query(Pr, distance_upper_bound=0.011); keep &= ~np.isfinite(dd)
    Pr, Nr = Pr[keep], Nr[keep]; tree = cKDTree(Pr)
    # several starts: the landmark fit, and nose tip on nose tip with the head frame's axes (both are x right, y forward,
    # z up) tilted by up to 10 deg; the lowest median face distance wins
    starts = [(R, t)]
    for ax in ([0, 0, 0], [10, 0, 0], [-10, 0, 0], [0, 0, 10], [0, 0, -10], [0, 10, 0], [0, -10, 0]):
        Rs = Rotation.from_euler("xyz", ax, degrees=True).as_matrix(); starts.append((Rs, Lr["nose_tip"][0] - Rs @ nose))
    best = None
    for Rs, ts in starts:
        R1, t1, _, dn = icp(src, tree, Pr, Nr, Rs, ts)
        if best is None or np.median(dn) < np.median(best[2]):
            best = (R1, t1, dn)
    R1, t1, dn = best
    _, _, s, dns = icp(src, tree, Pr, Nr, R1, t1, scale=True)
    lm1 = {k: float(np.linalg.norm(R1 @ Lm[k] + t1 - Lr[k][0]) * 1e3) for k in keys}
    T = np.eye(4); T[:3, :3] = R1; T[:3, 3] = t1
    res = {"T_mri_m_to_device_m": T.tolist(), "landmarks_used": keys, "landmark_err_start_mm": lm0, "landmark_err_mm": lm1,
           "face_points": int(mface.sum()),
           "face_surface_mm": {"median": float(np.median(dn) * 1e3), "p90": float(np.percentile(dn, 90) * 1e3),
                               "rms_trim85": float(np.sqrt(np.mean(np.sort(dn)[:int(0.85 * len(dn))] ** 2)) * 1e3)},
           "free_scale": float(s), "free_scale_median_mm": float(np.median(dns) * 1e3)}
    json.dump(res, open(out / "align_mri.json", "w"), indent=1)
    trimesh.Trimesh(Sv @ R1.T + t1, Sf, process=False).export(out / "skin_device.ply")
    trimesh.Trimesh(Bv @ R1.T + t1, Bf, process=False).export(out / "bone_device.ply")
    print(json.dumps({k: res[k] for k in ("face_points", "face_surface_mm", "free_scale")}), {k: round(v, 1) for k, v in lm1.items()})


if __name__ == "__main__":
    main()
