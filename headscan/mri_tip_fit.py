"""Re-fit the MRI skin placement with the screw tips as contact constraints, and map the soft-tissue thickness.

    python mri_tip_fit.py SKIN.ply BONE.ply HEAD.ply TIPS.npy OUT_DIR [--alpha 0.4] [--max-press 6] [--w-tip 3]

SKIN/BONE: MRI skin and skull already in the device frame (mri_via_phone.py), metres. HEAD: the reconstructed head
(frame removed by the CAD as elsewhere). TIPS: 12 screw tips (device, metres) from the calibrated readouts.
Unknown: one rigid correction of the MRI (rotation about the head centre + translation). Cost (mm):
  - face: head face points -> MRI skin, point-to-plane, Huber 1 mm (keeps today's face fit)
  - tips: each tip must touch the skin: outside the skin costs its distance; inside is free down to
    L_k = min(alpha x soft tissue under it, max-press) (tissue gives way), deeper costs the excess; weight w-tip
Soft tissue = skin -> skull distance at each skin vertex (heatmap, also written as vertex colours).
Writes OUT_DIR/tip_fit.json, skin_tipfit.ply, bone_tipfit.ply, skin_thickness.ply.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ARC = "/Users/jack/aspect/tfus-simulation/.claude/worktrees/vastai-cuda-setup-69d92b/results/arc_redesign/tags"
FRAME_STL = [f"{ARC}/frame_L116_A_plates.stl", f"{ARC}/frame_L116_B_plates.stl"]
NAMES = [f"sag{i + 1}" for i in range(8)] + [f"cor{i + 1}" for i in range(4)]
NOSE_DEV = np.array([-0.005, 0.117, -0.057])


def thickness_colors(t_mm, vmax=15.0):
    """Sequential map: thin tissue (bone right under the skin) light, thick (fat / fluid / muscle) dark blue."""
    x = np.clip(t_mm / vmax, 0, 1)[:, None]
    lo, hi = np.array([222, 235, 250.]), np.array([16, 66, 129.])
    return (lo + (hi - lo) * x).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skin"); ap.add_argument("bone"); ap.add_argument("head"); ap.add_argument("tips"); ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=0.4); ap.add_argument("--max-press", type=float, default=6.0)
    ap.add_argument("--w-tip", type=float, default=3.0)
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    skin = trimesh.load(a.skin, process=False); bone = trimesh.load(a.bone, process=False)
    tips = np.load(a.tips)
    # soft tissue thickness at every skin vertex (closest skull point)
    _, thk, _ = trimesh.proximity.closest_point(bone, skin.vertices)
    thk_mm = thk * 1e3
    tm = skin.copy(); tm.visual.vertex_colors = np.c_[thickness_colors(thk_mm), np.full(len(thk_mm), 255, np.uint8)]
    tm.export(O / "skin_thickness.ply")
    # dense skin samples with normals for fast signed point-to-plane distances
    sp, fi = trimesh.sample.sample_surface_even(skin, 400000, seed=0); sn = skin.face_normals[fi]
    st = cKDTree(sp)
    sp_thk = thk_mm[skin.faces[fi][:, 0]]
    # head face points without the frame
    H = np.asarray(trimesh.load(a.head, process=False).vertices)
    fr = np.vstack([trimesh.sample.sample_surface_even(trimesh.load(f), 60000, seed=0)[0] / 1000.0 for f in FRAME_STL])
    dfr, _ = cKDTree(fr).query(H, distance_upper_bound=0.011)
    hm = (np.linalg.norm(H - NOSE_DEV, axis=1) < 0.07) & (H[:, 1] > 0.035) & (H[:, 2] < 0.035) & (H[:, 2] > NOSE_DEV[2] - 0.075) & ~np.isfinite(dfr)
    face = H[hm][np.random.default_rng(0).choice(hm.sum(), min(8000, hm.sum()), replace=False)]
    c = np.r_[0.0, 0.0, 0.03]   # rotate about the head centre (device frame, ~ear axis, a bit up)

    def sdist(X, x):
        """signed distance of device points X to the MRI skin moved by x (inverse-move the points instead)."""
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        Y = (X - c - x[3:]) @ R + c            # into the original skin frame
        d, j = st.query(Y)
        return ((Y - sp[j]) * sn[j]).sum(1) * 1e3, j      # + outside the skin

    def limits(j):
        return np.minimum(a.alpha * sp_thk[j], a.max_press)

    def res(x):
        sf, _ = sdist(face, x)
        st_, j = sdist(tips, x)
        L = limits(j)
        rt = np.where(st_ > 0, st_, np.where(st_ < -L, st_ + L, 0.0))
        return np.r_[sf / np.sqrt(len(face) / 100.0), a.w_tip * rt]   # face term ~ 100 effective points

    def report(x, tag):
        sf, _ = sdist(face, x); st_, j = sdist(tips, x)
        return {"face_median_mm": float(np.median(np.abs(sf))), "face_p90_mm": float(np.percentile(np.abs(sf), 90)),
                "tips": {n: {"signed_mm": round(float(s), 2), "allowed_press_mm": round(float(l), 2), "tissue_mm": round(float(t), 1)}
                         for n, s, l, t in zip(NAMES, st_, limits(j), sp_thk[j])}}
    x0 = np.zeros(6)
    before = report(x0, "before")
    sol = least_squares(res, x0, loss="huber", f_scale=1.0, x_scale=[0.01, 0.01, 0.01, 0.001, 0.001, 0.001], diff_step=1e-4)
    after = report(sol.x, "after")
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    move = lambda V: (V - c) @ R.T + c + sol.x[3:]
    trimesh.Trimesh(move(skin.vertices), skin.faces, process=False).export(O / "skin_tipfit.ply")
    trimesh.Trimesh(move(bone.vertices), bone.faces, process=False).export(O / "bone_tipfit.ply")
    tm2 = tm.copy(); tm2.vertices = move(tm.vertices); tm2.export(O / "skin_thickness_tipfit.ply")
    out = {"params": vars(a), "correction": {"rot_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
           "rot_euler_xyz_deg": np.degrees(Rotation.from_rotvec(sol.x[:3]).as_euler("xyz")).round(2).tolist(), "t_mm": (sol.x[3:] * 1e3).round(2).tolist()},
           "before": before, "after": after}
    json.dump(out, open(O / "tip_fit.json", "w"), indent=1)
    print(f"correction {out['correction']}")
    print(f"face median {before['face_median_mm']:.2f} -> {after['face_median_mm']:.2f} mm (p90 {before['face_p90_mm']:.2f} -> {after['face_p90_mm']:.2f})")
    for n in NAMES:
        b_, f_ = before["tips"][n], after["tips"][n]
        print(f"  {n:5} tip {b_['signed_mm']:+6.2f} -> {f_['signed_mm']:+6.2f} mm  (allowed press {f_['allowed_press_mm']:.1f}, tissue {f_['tissue_mm']:.1f})")


if __name__ == "__main__":
    main()
