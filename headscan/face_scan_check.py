"""Agreement between an aligned face scan and a reconstructed head: surface distance, scale, and alignment uncertainty.

    python face_scan_check.py SCAN_DEVICE.ply REF.ply OUT.json [--nose X Y Z] [--boot 30]

SCAN_DEVICE.ply: the scan already aligned to REF (align_face_scan.py). On the face (within 85 mm of the nose tip,
in front of the ears) it reports:
  - surface distance scan -> REF (point-to-plane): median, p90, trimmed RMS (the alignment metric)
  - scale: a similarity ICP (rotation, translation, log-scale about the face centroid); scale != 1 means one of the two
    is stretched (the TrueDepth scan is metric)
  - uncertainty: bootstrap of the rigid ICP (resampled face points, start perturbed by 2 mm / 1 deg); spread of the face
    position (mm at the face points) = how well the face pins the alignment
"""
import argparse
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ARC = "/Users/jack/aspect/tfus-simulation/.claude/worktrees/vastai-cuda-setup-69d92b/results/arc_redesign/tags"
FRAME_STL = [f"{ARC}/frame_L116_A_plates.stl", f"{ARC}/frame_L116_B_plates.stl"]


def icp(Q, tree, Pr, Nr, scale=False, iters=40, trim=0.85):
    """Point-to-plane ICP of Q onto (Pr, Nr); returns (R, t, s) acting about the centroid of Q."""
    c = Q.mean(0); R, t, s = np.eye(3), np.zeros(3), 1.0
    for _ in range(iters):
        X = s * (Q - c) @ R.T + c + t
        d, j = tree.query(X); keep = d < np.quantile(d, trim)
        A, N, B = X[keep], Nr[j[keep]], Pr[j[keep]]
        r = ((B - A) * N).sum(1)
        J = np.c_[np.cross(A - c, N), N] if not scale else np.c_[np.cross(A - c, N), N, ((A - c) * N).sum(1)]
        x = np.linalg.lstsq(J, r, rcond=None)[0]
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        R = dR @ R; t = dR @ t + x[3:6]
        if scale:
            s *= np.exp(x[6])
        if np.linalg.norm(x) < 1e-8:
            break
    return R, t, s, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan"); ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--nose", type=float, nargs=3, default=None, help="nose tip, device metres (default: most forward scan point)")
    ap.add_argument("--boot", type=int, default=30)
    a = ap.parse_args()
    S = np.asarray(trimesh.load(a.scan, process=False).vertices)
    ref = trimesh.load(a.ref, process=False)
    Pr = np.asarray(ref.vertices); Nr = np.asarray(ref.vertex_normals) if isinstance(ref, trimesh.Trimesh) and len(ref.faces) else None
    nose = np.array(a.nose) if a.nose else S[np.argmax(S[:, 1])]
    # the alignment's face region (align_face_scan.py): no hair above the brows (eyes ~ 57 mm above the nose tip), not
    # below the chin, in front of the ears; reference without the frame (nominal CAD + 11 mm)
    face = S[(np.linalg.norm(S - nose, axis=1) < 0.09) & (S[:, 1] > 0.035) & (S[:, 2] < nose[2] + 0.09) & (S[:, 2] > nose[2] - 0.075)]
    near = np.linalg.norm(Pr - nose, axis=1) < 0.12
    for f in FRAME_STL:
        sp = trimesh.sample.sample_surface_even(trimesh.load(f), 60000, seed=0)[0] / 1000.0
        dd, _ = cKDTree(sp).query(Pr, distance_upper_bound=0.011); near &= ~np.isfinite(dd)
    Pr, Nr = Pr[near], Nr[near]
    tree = cKDTree(Pr)
    d, j = tree.query(face); dn = np.abs(((face - Pr[j]) * Nr[j]).sum(1)); ok = d < 0.01
    out = {"face_points": int(len(face)), "surface_mm": {"median": float(np.median(dn[ok]) * 1e3), "p90": float(np.percentile(dn[ok], 90) * 1e3),
           "rms_trim85": float(np.sqrt(np.mean(np.sort(dn[ok])[:int(0.85 * ok.sum())] ** 2)) * 1e3), "beyond_10mm_frac": float(1 - ok.mean())}}
    sub = face[np.random.default_rng(0).choice(len(face), min(30000, len(face)), replace=False)]
    R, t, s, c = icp(sub, tree, Pr, Nr, scale=True)
    out["similarity"] = {"scale": float(s), "rot_deg": float(np.degrees(Rotation.from_matrix(R).magnitude())), "t_mm": (t * 1e3).round(2).tolist()}
    rng = np.random.default_rng(1); disp = []
    for b in range(a.boot):
        q = face[rng.choice(len(face), min(30000, len(face)), replace=True)]
        dR0 = Rotation.from_rotvec(rng.normal(0, np.radians(1.0) / np.sqrt(3), 3)).as_matrix(); dt0 = rng.normal(0, 0.002 / np.sqrt(3), 3)
        cq = q.mean(0); q0 = (q - cq) @ dR0.T + cq + dt0
        R, t, _, c = icp(q0, tree, Pr, Nr)
        # where the perturbed + re-fitted face ends up vs where it started (identity): displacement at the face points
        X = ((sub - cq) @ dR0.T + cq + dt0 - c) @ R.T + c + t
        disp.append(np.linalg.norm(X - sub, axis=1))
    disp = np.array(disp)
    out["bootstrap"] = {"runs": a.boot, "face_displacement_mm": {"median": float(np.median(disp) * 1e3), "p95": float(np.percentile(disp, 95) * 1e3)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
