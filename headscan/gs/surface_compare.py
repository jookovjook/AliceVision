"""Surface distances between a reference mesh and test meshes / point clouds inside the head cube (device frame, m).

    python surface_compare.py REF.ply OUT.json LABEL=TEST.ply [LABEL=TEST.ply ...]
                              [--center 0.02 0.04 0.04 --half 0.22 --n 1000000 --heat-dir DIR --heat-max 0.005]
                              [--visible SPARSE_DIR --cam-step 2 --clean-dir DIR]

Meshes are sampled area-uniformly (n points), point clouds are used as they are (vertices with opacity >= 0.5 when
the PLY has an opacity field). Distances are exact point-to-triangle (Open3D RaycastingScene):
  test->ref  how far the test surface lies from the reference (accuracy-like; includes what only the test has)
  ref->test  how far the reference lies from the test surface (completeness-like; meshes only)
Per region in the device frame (origin = ear-axis midpoint, x right, y forward, z up): all, face (y > 0.06,
|x| < 0.07, z > -0.06), ears (|x| > 0.055, |y| < 0.04, |z| < 0.04), crown (z > 0.09).
--heat-dir writes LABEL_heat.ply: the test mesh with vertex colours = distance to the reference (0 .. --heat-max,
viridis; grey where farther than 2 x heat-max), and ref_vs_LABEL_heat.ply: the reference coloured by its distance to
the test mesh (where the reference has surface the test lacks), for headscan/viz/render.py compare --textured.
--visible keeps, in every mesh, only the triangles that some training camera (COLMAP model in SPARSE_DIR, every
--cam-step-th) sees unoccluded by that mesh and inside its image, and turns each kept triangle to the side the cameras
see it from (cosine-weighted vote). This drops what no camera
can see, e.g. the second layer a TSDF leaves at the back of its truncation band when unobserved voxels are not masked,
and fixes inverted winding (render.py's pyrender culls back faces). --clean-dir writes these meshes (LABEL_visible.ply,
ref_visible.ply) for the figures.
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

REGIONS = {
    "all": lambda p: np.ones(len(p), bool),
    "face": lambda p: (p[:, 1] > 0.06) & (np.abs(p[:, 0]) < 0.07) & (p[:, 2] > -0.06),
    "ears": lambda p: (np.abs(p[:, 0]) > 0.055) & (np.abs(p[:, 1]) < 0.04) & (np.abs(p[:, 2]) < 0.04),
    "crown": lambda p: p[:, 2] > 0.09,
}


def in_cube(p, lo, hi):
    return np.all((p >= lo) & (p <= hi), 1)


def load(path, lo, hi):
    """(legacy mesh or None, points in cube)."""
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles):
        v = np.asarray(m.vertices); f = np.asarray(m.triangles)
        m.remove_triangles_by_mask(~in_cube(v, lo, hi)[f].all(1)); m.remove_unreferenced_vertices()
        return m, None
    from plyfile import PlyData
    el = PlyData.read(path)["vertex"]
    p = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    keep = in_cube(p, lo, hi)
    if "opacity" in el.data.dtype.names:
        keep &= np.asarray(el["opacity"]) >= 0.5
    return None, p[keep]


def cameras(sparse, step):
    """[(R world->camera, t, fx, fy, cx, cy, w, h)] from a COLMAP text model, every step-th image."""
    c = [l.split() for l in open(os.path.join(sparse, "cameras.txt")) if l.strip() and not l.startswith("#")][0]
    w, h, fx, fy, cx, cy = int(c[2]), int(c[3]), *map(float, c[4:8])
    out = []
    for l in open(os.path.join(sparse, "images.txt")):
        e = l.split()
        if len(e) < 10 or l.startswith("#"):
            continue
        qw, x, y, z = map(float, e[1:5])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - qw * z), 2 * (x * z + qw * y)],
                      [2 * (x * y + qw * z), 1 - 2 * (x * x + z * z), 2 * (y * z - qw * x)],
                      [2 * (x * z - qw * y), 2 * (y * z + qw * x), 1 - 2 * (x * x + y * y)]])
        out.append((R, np.array(e[5:8], float), fx, fy, cx, cy, w, h))
    return out[::step]


def visible_layer(mesh, cams):
    """Triangles that some camera sees unoccluded inside its image (ray camera->centroid first hits the triangle);
    each kept triangle is turned to the side the cameras see it from (cosine-weighted vote)."""
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    ctr = v[f].mean(1)
    mesh.compute_triangle_normals(); n = np.asarray(mesh.triangle_normals)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    seen = np.zeros(len(f), bool); vote = np.zeros(len(f))
    for R, t, fx, fy, cx, cy, w, h in cams:
        C = -R.T @ t
        Xc = ctr @ R.T + t
        z = Xc[:, 2]; u = fx * Xc[:, 0] / z + cx; vv = fy * Xc[:, 1] / z + cy
        idx = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h))
        d = ctr[idx] - C; dist = np.linalg.norm(d, axis=1); d /= dist[:, None]
        hit = sc.cast_rays(o3d.core.Tensor(np.hstack([np.repeat(C[None], len(idx), 0), d]).astype(np.float32)))
        ok = (hit["primitive_ids"].numpy() == idx) | (hit["t_hit"].numpy() >= dist - 1e-5)
        idx = idx[ok]
        seen[idx] = True
        vote[idx] -= (n[idx] * d[ok]).sum(1)  # + when the normal points at the camera
    flip = seen & (vote < 0)
    f2 = f.copy(); f2[flip] = f2[flip][:, ::-1]
    out = o3d.geometry.TriangleMesh(mesh); out.triangles = o3d.utility.Vector3iVector(f2)
    out.remove_triangles_by_mask(~seen); out.remove_unreferenced_vertices(); out.compute_vertex_normals()
    return out, float(seen.mean()), float(flip[seen].mean())


def sample(mesh, n):
    return np.asarray(mesh.sample_points_uniformly(n).points)


def dist_to(mesh, p):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return sc.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()


def heat(mesh, other, vmax, out):
    import matplotlib
    dv = dist_to(other, np.asarray(mesh.vertices))
    c = matplotlib.colormaps["viridis"](np.clip(dv / vmax, 0, 1))[:, :3]
    c[dv > 2 * vmax] = 0.75
    h = o3d.geometry.TriangleMesh(mesh); h.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(out, h)


def stats(d, p):
    out = {}
    for name, f in REGIONS.items():
        dd = d[f(p)] * 1000
        if len(dd) < 100:
            continue
        out[name] = {"n": int(len(dd)), "median_mm": round(float(np.median(dd)), 3), "p90_mm": round(float(np.percentile(dd, 90)), 3),
                     "mean_mm": round(float(dd.mean()), 3), "within_1mm": round(float((dd < 1).mean()), 3),
                     "within_2mm": round(float((dd < 2).mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("out"); ap.add_argument("tests", nargs="+")
    ap.add_argument("--center", type=float, nargs=3, default=[0.02, 0.04, 0.04]); ap.add_argument("--half", type=float, default=0.22)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--heat-dir", default=None); ap.add_argument("--heat-max", type=float, default=0.005)
    ap.add_argument("--visible", default=None); ap.add_argument("--cam-step", type=int, default=2)
    ap.add_argument("--clean-dir", default=None)
    a = ap.parse_args()
    lo, hi = np.array(a.center) - a.half, np.array(a.center) + a.half
    cams = cameras(a.visible, a.cam_step) if a.visible else None

    def clean(mesh, lab):
        if cams is None:
            return mesh, {}
        m, kept, flipped = visible_layer(mesh, cams)
        print(f"{lab}: {kept:.0%} of triangles seen by a camera, {flipped:.0%} of those were facing away", flush=True)
        if a.clean_dir:
            os.makedirs(a.clean_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(a.clean_dir, f"{lab}_visible.ply"), m)
        return m, {"visible_share": round(kept, 3), "flipped_share": round(flipped, 3)}

    ref, _ = load(a.ref, lo, hi)
    ref, ref_vis = clean(ref, "ref")
    ref_pts = sample(ref, a.n)
    res = {"ref": a.ref, "cube": {"center": a.center, "half": a.half}, "ref_triangles": len(ref.triangles),
           "visible_filter": {"sparse": a.visible, "cam_step": a.cam_step} if a.visible else None, "ref_visibility": ref_vis, "tests": {}}
    for t in a.tests:
        lab, path = t.split("=", 1)
        mesh, pts = load(path, lo, hi)
        r = {"path": path}
        if mesh is not None:
            mesh, r["visibility"] = clean(mesh, lab)
            r["triangles"] = len(mesh.triangles)
            pts = sample(mesh, a.n)
            d_rt = dist_to(mesh, ref_pts)
            r["ref_to_test"] = stats(d_rt, ref_pts)
            if a.heat_dir:
                os.makedirs(a.heat_dir, exist_ok=True)
                heat(mesh, ref, a.heat_max, os.path.join(a.heat_dir, f"{lab}_heat.ply"))
                heat(ref, mesh, a.heat_max, os.path.join(a.heat_dir, f"ref_vs_{lab}_heat.ply"))
        else:
            r["points"] = int(len(pts))
        d_tr = dist_to(ref, pts)
        r["test_to_ref"] = stats(d_tr, pts)
        res["tests"][lab] = r
        a_ = r["test_to_ref"]["all"]; b_ = r.get("ref_to_test", {}).get("all", {})
        print(f"{lab:>14}: test->ref median {a_['median_mm']:.2f} mm p90 {a_['p90_mm']:.2f} mm"
              + (f" | ref->test median {b_['median_mm']:.2f} mm p90 {b_['p90_mm']:.2f} mm" if b_ else ""), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
