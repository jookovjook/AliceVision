"""Place Jack's MRI (skin + skull, tfus-simulation results/me/viewer.npz, head frame mm) in the device frame through the
phone scans: the Oct-14 TrueDepth scan stored with the MRI (phone_P, already aligned to the MRI at 0.87 mm) is fitted to
today's TrueDepth scan, which align_face_scan.py placed on the reconstructed head by the face.

    python mri_via_phone.py VIEWER.npz SCAN.ply SCAN_ALIGN.json REF.ply OUT_DIR

Direct MRI-skin-to-head ICP slides (supine face, closed eyes, coil-pressed cheeks); the phone-to-phone fit does not
(0.27-0.44 mm). Check: head face points -> MRI skin distance (frame removed). Writes OUT_DIR/mri_via_phone.json and
mri_skin_device.ply / mri_skull_device.ply (metres, device frame).
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_mri import icp  # noqa: E402
from align_face_scan import FRAME_STL  # noqa: E402

NOSE_DEV = np.array([-0.005, 0.117, -0.057])   # nose tip in the device frame (face landmarks of the head capture)


def read_scan(path):
    """StandardCyborg ascii PLY -> points, normals (metres, scan frame)."""
    body = open(path).read().split("end_header\n", 1)[1]
    D = np.loadtxt(body.splitlines(), usecols=range(6))
    return D[:, :3], D[:, 3:6]


def main():
    vnpz, scan, salign, ref, out = sys.argv[1:6]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    V = np.load(vnpz)
    L = V["L_mri"] / 1000.0; nose = L[1]; eye_z = np.mean([L[i][2] for i in (33, 133, 362, 263)])
    Q = V["phone_P"].astype(float) / 1000.0
    src = Q[(np.linalg.norm(Q - nose, axis=1) < 0.07) & (Q[:, 2] < eye_z + 0.035) & (Q[:, 1] > nose[1] - 0.05)]
    src = src[np.random.default_rng(0).choice(len(src), min(20000, len(src)), replace=False)]
    T = np.array(json.load(open(salign))["T_scan_to_device"])
    P0, N0 = read_scan(scan); P = P0 @ T[:3, :3].T + T[:3, 3]; N = N0 @ T[:3, :3].T
    m = np.linalg.norm(P - NOSE_DEV, axis=1) < 0.1; tree = cKDTree(P[m])
    best = None
    for ax in ([0, 0, 0], [25, 0, 0], [-25, 0, 0], [0, 0, 15], [0, 0, -15], [15, 10, 0], [-15, -10, 0]):
        Rs = Rotation.from_euler("xyz", ax, degrees=True).as_matrix()
        R, t, _, dn = icp(src, tree, P[m], N[m], Rs, NOSE_DEV - Rs @ nose)
        if best is None or np.median(dn) < np.median(best[2]):
            best = (R, t, dn)
    R, t, dn = best
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    skin = trimesh.Trimesh(V["charm_v"].astype(float) / 1000.0 @ R.T + t, V["charm_f"], process=False)
    skull = trimesh.Trimesh(V["bone_v"].astype(float) / 1000.0 @ R.T + t, V["bone_f"], process=False)
    # check: head face (frame removed) -> MRI skin
    H = np.asarray(trimesh.load(ref, process=False).vertices)
    sp = np.vstack([trimesh.sample.sample_surface_even(trimesh.load(f), 60000, seed=0)[0] / 1000.0 for f in FRAME_STL])
    dfr, _ = cKDTree(sp).query(H, distance_upper_bound=0.011)
    hm = (np.linalg.norm(H - NOSE_DEV, axis=1) < 0.07) & (H[:, 1] > 0.035) & (H[:, 2] < 0.035) & (H[:, 2] > NOSE_DEV[2] - 0.075) & ~np.isfinite(dfr)
    X = H[hm][np.random.default_rng(0).choice(hm.sum(), min(40000, hm.sum()), replace=False)]
    cp, dist, tri = trimesh.proximity.closest_point(skin, X)
    sgn = np.sign(((X - cp) * skin.face_normals[tri]).sum(1))
    res = {"T_mri_head_m_to_device_m": M.tolist(),
           "phone_to_phone_mm": {"median": float(np.median(dn) * 1e3), "p90": float(np.percentile(dn, 90) * 1e3)},
           "head_face_to_mri_skin_mm": {"median": float(np.median(dist) * 1e3), "p90": float(np.percentile(dist, 90) * 1e3),
                                        "signed_median": float(np.median(sgn * dist) * 1e3)}}
    json.dump(res, open(out / "mri_via_phone.json", "w"), indent=1)
    skin.export(out / "mri_skin_device.ply"); skull.export(out / "mri_skull_device.ply")
    print(json.dumps({k: v for k, v in res.items() if k != "T_mri_head_m_to_device_m"}))


if __name__ == "__main__":
    main()
