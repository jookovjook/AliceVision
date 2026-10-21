"""Convert every reconstruction to compact GLB files in the DEVICE frame for the web viewer.

    python export_viewer_assets.py SRC_DIR APPLE_COMPARE_DIR OUT_DIR

Each mesh is put in the device frame (metres; origin = ear-axis midpoint, x right, y forward, z up), cropped to a
0.30 m sphere around the head (0.02, 0.04, 0.04), decimated, given vertex colours (textures are sampled at the
vertices) and written as GLB with the three.js axis convention (y up): (x, y, z)_device -> (x, z, -y)_gl.
Frames: AliceVision OBJ = (x, -y, -z) of its SfM world; step-2 and Apple meshes are mapped to the device frame by a
similarity fitted on shared camera centres (refined-ARKit world for steps 1-2, raw ARKit for Apple).
Also writes assets.json (name, file, faces, notes) and markers.json (tag corners, ear balls, landmarks, screws).
"""
import json
import sys
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

HEAD = np.array([0.02, 0.04, 0.04])
GL = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # device -> three.js (y up)


def cams(path):
    d = json.load(open(path))
    P = {p["poseId"]: np.array([float(x) for x in p["pose"]["transform"]["center"]]) for p in d["poses"]}
    return {v["viewId"]: P[v["poseId"]] for v in d["views"] if v.get("poseId") in P}


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (X - mx).var(0).sum()
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = my - s * R @ mx
    return T


def fit(src, dst):
    a, b = cams(src), cams(dst)
    ids = sorted(set(a) & set(b))
    T = umeyama(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]))
    res = np.linalg.norm((np.c_[np.array([a[i] for i in ids]), np.ones(len(ids))] @ T.T)[:, :3] - np.array([b[i] for i in ids]), axis=1)
    return T, float(np.median(res) * 1000)


def load_coloured(path, T=None, flip=False):
    m = trimesh.load(path, process=False)
    geoms = list(m.geometry.values()) if isinstance(m, trimesh.Scene) else [m]
    out = []
    for g in geoms:
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        if g.visual.kind == "texture":
            g.visual = g.visual.to_color()
        col = g.visual.vertex_colors[:, :3] if g.visual.kind == "vertex" else np.full((len(g.vertices), 3), 190, np.uint8)
        V = np.asarray(g.vertices, float)
        if flip:
            V = V * [1, -1, -1]
        if T is not None:
            V = V @ T[:3, :3].T + T[:3, 3]
        out.append((V, np.asarray(g.faces), np.asarray(col, np.uint8)))
    V = np.concatenate([o[0] for o in out]); C = np.concatenate([o[2] for o in out])
    off = np.cumsum([0] + [len(o[0]) for o in out[:-1]])
    F = np.concatenate([o[1] + k for o, k in zip(out, off)])
    return V, F, C


def write(name, V, F, C, out, target, radius=0.30):
    keepf = (np.linalg.norm(V[F].mean(1) - HEAD, axis=1) < radius)
    F = F[keepf]
    used = np.unique(F); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    V, C, F = V[used], C[used], remap[F]
    n0 = len(F)
    if len(F) > target:
        V2, F2 = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=1 - target / len(F))
        from scipy.spatial import cKDTree
        _, idx = cKDTree(V).query(V2); C = C[idx]; V, F = V2.astype(float), F2
    m = trimesh.Trimesh(V @ GL.T, F, vertex_colors=np.c_[C, np.full(len(C), 255, np.uint8)], process=False)
    m.export(out / f"{name}.glb")
    sz = (out / f"{name}.glb").stat().st_size
    print(f"{name:22s} {n0:8d} -> {len(F):7d} faces, {sz / 1e6:5.1f} MB")
    return {"file": f"{name}.glb", "faces_full": int(n0), "faces": int(len(F))}


def main():
    src, apple, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    T_B, rB = fit(src / "box_work/step2/B_refine/04_sfm/cameras.sfm", src / "box_work/step4/cameras_device.sfm")
    T_A, rA = fit(src / "arkit_poses.sfm", src / "box_work/step4/cameras_device.sfm")
    print(f"refined-ARKit -> device: camera residual {rB:.1f} mm; raw ARKit -> device: {rA:.1f} mm")
    S = src
    jobs = [
        ("step1_rgb", apple / "av_step1_head.obj", T_B, False, 150000),
        ("step2_arkit", apple / "av_step2_head.obj", T_B, False, 150000),
        ("step5_lidar_tsdf", S / "step5/lidar_tsdf.ply", None, False, 150000),
        ("step5_mvs_graphcut", S / "box_work/step5/mvs_full/05_dense/meshF/mesh.obj", None, True, 200000),
        ("step5_mvs_tsdf", S / "step5/tsdf_mvs_fixed.ply", None, False, 250000),
        ("step5_mvs_lidar_tsdf", S / "step5/tsdf_mvs_lidar_fixed.ply", None, False, 200000),
        ("apple_rgbd", apple / "apple_rgbd_head.obj", T_A, False, 60000),
        ("apple_rgb", apple / "apple_rgb_head.obj", T_A, False, 60000),
        ("gs_stochastic_solids", S / "ss_tetra.ply", None, False, 350000),
        ("gs_dnsplatter_lidar", S / "gs/dnsplatter/out/dn/tsdf_gs.ply", None, False, 250000),
        ("gs_dnsplatter_rgb", S / "gs/dnsplatter/out/rgb/tsdf_gs.ply", None, False, 250000),
        ("gs_ags_mesh", S / "gs/dnsplatter/out/ags/tsdf_gs.ply", None, False, 250000),
        ("seg_labels", S / "step8_on_stochastic_solids/seg/head_labels.ply", None, False, 300000),
        ("seg_head_closed", S / "step8_on_stochastic_solids/seg/head_closed_flags.ply", None, False, 300000),
        ("seg_device", S / "step8_on_stochastic_solids/seg/device.ply", None, False, 150000),
        ("cad_frame_screws", S / "step8/cad_frame_screws_read.ply", None, False, 120000),
    ]
    assets = {}
    for name, path, T, flip, target in jobs:
        if not Path(path).exists():
            print("missing", path); continue
        V, F, C = load_coloured(path, T, flip)
        if name == "cad_frame_screws":
            C[:] = [150, 160, 175]
        assets[name] = write(name, V, F, C, out, target, radius=0.40 if name == "cad_frame_screws" else 0.30)
    json.dump({"assets": assets, "fits_mm": {"refined_arkit": rB, "raw_arkit": rA}}, open(out / "assets.json", "w"), indent=1)

    # markers: tag corners (reconstructed vs CAD), ear balls, face landmarks, screw tips + axes
    rd = json.load(open(S / "step8_on_stochastic_solids/readout.json"))
    rep = json.load(open(S / "step8_on_stochastic_solids/step8_readouts.json"))
    face = json.load(open(S / "step8_on_stochastic_solids/facelm.json"))["landmarks"]
    g = lambda p: (np.array(p, float) / 1000.0 @ GL.T).round(5).tolist()
    screws = []
    for r, row in zip(rd["screws"], rep["readouts"]):
        o = np.array(r["origin_mm"]); d = np.array(r["dirn"]); d /= np.linalg.norm(d)
        inner = o - 4 * d
        gap = row.get("gap_fine")
        screws.append({"hole": r["hole"], "readout": row["readout"], "gap_mm": gap, "sim_mm": row["sim_gap"],
                       "inner": g(inner), "tip": g(inner + (gap or 0) * d), "axis_out": g(inner - 100 * d)})
    a = rep["alignment"]
    markers = {"screws": screws, "ear": {"R": g(a["ear_ball_ring_R_mm"]), "L": g(a["ear_ball_ring_L_mm"]), "E_mm": a["ear_halfwidth_mm"]},
               "landmarks": {k: g(v["mm"]) for k, v in face.items() if v.get("spread_mm", 99) < 6},
               "device_nasion_point": g([0, 116, 0]), "alignment": a}
    json.dump(markers, open(out / "markers.json", "w"), indent=1)
    print("markers:", len(screws), "screws,", len(markers["landmarks"]), "landmarks")


if __name__ == "__main__":
    main()
