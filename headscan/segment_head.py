"""Steps 6-7: cut the head out of the fused mesh and split it into skin/hair vs the device, in the device frame.

    python segment_head.py MESH.ply ARC_DIR OUT_DIR [--chin 0.05,-0.125] [--nape -0.08,-0.05] [--frame-mm 4]

Device frame (from the frame-tag fit): origin = ear-axis midpoint, x right, y forward, z up, metres.
Step 6: keep the side of the plane through the under-chin and nape points (given as (y, z), plane parallel to x)
        that contains the head, within 0.30 m of the origin; drop components < 2 % of the largest.
Step 7: vertex labels
  frame  - within --frame-mm of the CAD frame (tags/frame_L116_A_plates.stl + B_plates.stl, mm, device frame;
           sampled at ~0.7 mm and queried with a KD-tree);
  screw  - within 6 mm of one of the 12 hole axes (frame_tags_layout.json 'holes': origin, dirn into the head) for
           axial positions -95..+60 mm, or within 13 mm outside the frame (t < 0: shaft + wing, up to ~90 mm out);
  ear    - near the ear axis (x axis): radius 7 mm for 0.060 < |x| < 0.085 (ear-screw ball), 13 mm beyond (barrel,
           knob, wing);
  head   - everything else (skin + hair; the screw tips' contact patches are labelled screw for now, step 8 refines).
Step 7b: close_head() fills what the device covered (screened Poisson), filled parts flagged.
Writes OUT_DIR/head_all.ply (step 6), head_labels.ply (colours by label), head_skin.ply, device.ply, head_closed.ply,
head_closed_flags.ply, segment.json
(counts + CAD-vs-reconstruction distances, the check that the CAD really sits on the reconstructed frame).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

COL = {"head": [214, 170, 140], "frame": [40, 90, 200], "screw": [230, 120, 20], "ear": [40, 170, 60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("arc"); ap.add_argument("out")
    ap.add_argument("--chin", default="0.05,-0.125"); ap.add_argument("--nape", default="-0.08,-0.05")
    ap.add_argument("--frame-mm", type=float, default=4.0); ap.add_argument("--poisson-depth", type=int, default=10)
    a = ap.parse_args()
    out, arc = Path(a.out), Path(a.arc); out.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(a.mesh, process=False)
    # ---- step 6: plane cut
    cy, cz = map(float, a.chin.split(",")); ny, nz = map(float, a.nape.split(","))
    t = np.array([0.0, cy - ny, cz - nz]); t /= np.linalg.norm(t)
    n = np.cross([1.0, 0, 0], t)                     # normal in the y-z plane
    p0 = np.array([0.0, ny, nz])
    if n @ (np.array([0, 0, 0.06]) - p0) < 0:
        n = -n
    V = m.vertices
    keepv = ((V - p0) @ n > 0) & (np.linalg.norm(V, axis=1) < 0.30)
    keepf = keepv[m.faces].all(1)
    h = m.copy(); h.update_faces(keepf); h.remove_unreferenced_vertices()
    parts = h.split(only_watertight=False)
    big = max(len(p.faces) for p in parts)
    h = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.02 * big])
    h.export(out / "head_all.ply")
    # ---- step 7: labels
    L = json.load(open(arc / "tags" / "frame_tags_layout.json"))
    cad = trimesh.util.concatenate([trimesh.load(arc / "tags" / f) for f in ("frame_L116_A_plates.stl", "frame_L116_B_plates.stl")])
    cad.apply_scale(0.001)
    S, _ = trimesh.sample.sample_surface_even(cad, 600000)
    dcad, _ = cKDTree(S).query(h.vertices)
    V = h.vertices
    lab = np.full(len(V), "head", dtype=object)
    lab[dcad < a.frame_mm / 1000] = "frame"
    for hole in L["holes"]:
        o = np.array(hole["origin"]) / 1000; d = np.array(hole["dirn"]) / np.linalg.norm(hole["dirn"])
        tt = (V - o) @ d
        r = np.linalg.norm((V - o) - tt[:, None] * d, axis=1)
        # 96 mm screws (tip z = 70, wing z 0..-26 in the screw frame); gaps 10-54 mm -> wing reaches ~90 mm outside
        lab[((r < 0.006) & (tt > -0.095) & (tt < 0.060)) | ((r < 0.013) & (tt > -0.095) & (tt < 0.0) & (lab != "frame"))] = "screw"
    rx = np.hypot(V[:, 1], V[:, 2]); ax_ = np.abs(V[:, 0])
    lab[((ax_ > 0.060) & (ax_ < 0.085) & (rx < 0.007)) | ((ax_ >= 0.085) & (rx < 0.013))] = "ear"
    cols = np.array([COL[l] for l in lab], np.uint8)
    hl = h.copy(); hl.visual = trimesh.visual.ColorVisuals(hl, vertex_colors=np.c_[cols, np.full(len(cols), 255, np.uint8)])
    hl.export(out / "head_labels.ply")
    for name, sel in (("head_skin", lab == "head"), ("device", lab != "head")):
        s = h.copy(); s.update_faces(sel[s.faces].all(1)); s.remove_unreferenced_vertices()
        if name == "head_skin":  # drop small skin islands left between device parts
            ps = s.split(only_watertight=False); bs = max(len(p.faces) for p in ps)
            s = trimesh.util.concatenate([p for p in ps if len(p.faces) >= 0.01 * bs])
        s.export(out / f"{name}.ply")
    # CAD vs reconstruction: how far is the visible CAD surface from the reconstructed mesh
    dS, _ = cKDTree(h.vertices).query(S)
    vis = dS < 0.010  # CAD points that have reconstructed geometry within 1 cm (i.e. not hidden/not reconstructed)
    res = {"faces_step6": int(len(h.faces)), "plane": {"normal": n.tolist(), "point": p0.tolist()},
           "labels": {k: int((lab == k).sum()) for k in COL},
           "frame_vertices_to_cad_mm": {"median": float(np.median(dcad[lab == "frame"]) * 1000)},
           "cad_to_mesh_mm": {"fraction_within_1cm": float(vis.mean()), "median": float(np.median(dS[vis]) * 1000),
                              "p90": float(np.percentile(dS[vis], 90) * 1000)}}
    json.dump(res, open(out / "segment.json", "w"), indent=1)
    cad.export(out / "cad_frame_device_m.ply")
    res["closed_head"] = close_head(out, n, p0, a.poisson_depth)
    json.dump(res, open(out / "segment.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def close_head(out, n, p0, depth):
    """Step 7b: screened Poisson over the skin/hair vertices (normals oriented away from the head centre) to fill
    what the device covered; vertices farther than 2 mm from any measured skin/hair vertex are flagged 'filled'
    (magenta in head_closed_flags.ply) so they are never mistaken for measurements."""
    import open3d as o3d
    s = trimesh.load(out / "head_skin.ply", process=False)
    P = s.vertices; N = s.vertex_normals.copy()
    flip = (N * (P - np.array([0, 0, 0.03]))).sum(1) < 0
    N[flip] *= -1
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals = o3d.utility.Vector3dVector(N)
    pc.colors = o3d.utility.Vector3dVector(s.visual.vertex_colors[:, :3] / 255.0)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.05)
    dens = np.asarray(dens)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    # trim by distance to measured skin/hair (not by density: the patches that fill unseen areas are low-density)
    dd, _ = cKDTree(P).query(V)
    keepv = (dd < 0.025) & ((V - p0) @ n > 0.003) & (np.linalg.norm(V - [0, 0, 0.03], axis=1) < 0.16)
    c = trimesh.Trimesh(V, F, vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8), process=False)
    c.update_faces(keepv[F].all(1)); c.remove_unreferenced_vertices()
    c = max(c.split(only_watertight=False), key=lambda q: len(q.faces))
    d, _ = cKDTree(P).query(c.vertices)
    filled = d > 0.002
    c.export(out / "head_closed.ply")
    f = c.copy(); vc = f.visual.vertex_colors.copy(); vc[filled] = [230, 0, 200, 255]
    f.visual = trimesh.visual.ColorVisuals(f, vertex_colors=vc); f.export(out / "head_closed_flags.ply")
    return {"faces": int(len(c.faces)), "filled_vertex_fraction": float(filled.mean()), "poisson_depth": depth}


if __name__ == "__main__":
    main()
