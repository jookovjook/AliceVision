"""Independent geometry check on the rigid frame: meshes vs the CAD frame placed in the device frame (step 6/7,
fitted to the AprilTag corners at 1.8 mm RMS), plus the layering of a mesh relative to another one.

    python cad_check.py CAD.ply OUT.json LABEL=MESH.ply [...] [--band 0.008 --n 1000000 --layer-ref LABEL --layer-to LABEL]

Arc: mesh samples within --band of the CAD surface are taken as frame; signed distance to the (watertight) CAD,
positive = outside the solid. Reported: median signed offset (bias), median / p90 |d|, and CAD coverage = share of
CAD surface samples with a mesh point within 2 mm (both on the CAD faces the cameras can see and the hidden ones).
Layering (--layer-ref A --layer-to B): signed offset of A's samples along B's normals at the closest B point; a shell
of A outside B shows up as a positive mode.
"""
import argparse
import json

import numpy as np
import open3d as o3d


def scene_of(mesh):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cad"); ap.add_argument("out"); ap.add_argument("meshes", nargs="+")
    ap.add_argument("--band", type=float, default=0.008); ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--layer-ref", default=None); ap.add_argument("--layer-to", default=None)
    a = ap.parse_args()
    cad = o3d.io.read_triangle_mesh(a.cad)
    cad_sc = scene_of(cad)
    cad_pts = np.asarray(cad.sample_points_uniformly(200000).points).astype(np.float32)
    meshes = {lab: o3d.io.read_triangle_mesh(p) for lab, p in (m.split("=", 1) for m in a.meshes)}
    res = {"cad": a.cad, "cad_watertight": bool(cad.is_watertight()), "band_m": a.band, "meshes": {}}
    for lab, m in meshes.items():
        p = np.asarray(m.sample_points_uniformly(a.n).points).astype(np.float32)
        sd = cad_sc.compute_signed_distance(o3d.core.Tensor(p)).numpy()
        arc = np.abs(sd) < a.band
        d = sd[arc] * 1000
        cov = scene_of(m).compute_distance(o3d.core.Tensor(cad_pts)).numpy()
        r = {"arc_samples": int(arc.sum()), "signed_median_mm": round(float(np.median(d)), 3),
             "abs_median_mm": round(float(np.median(np.abs(d))), 3), "abs_p90_mm": round(float(np.percentile(np.abs(d), 90)), 3),
             "cad_within_2mm": round(float((cov < 0.002).mean()), 3)}
        res["meshes"][lab] = r
        print(f"{lab:>8}: arc |d| median {r['abs_median_mm']:.2f} mm p90 {r['abs_p90_mm']:.2f} mm, signed median "
              f"{r['signed_median_mm']:+.2f} mm, CAD surface covered (2 mm) {r['cad_within_2mm']:.0%}", flush=True)
    if a.layer_ref and a.layer_to:
        A, B = meshes[a.layer_ref], meshes[a.layer_to]
        B.compute_triangle_normals()
        pa = np.asarray(A.sample_points_uniformly(a.n).points).astype(np.float32)
        q = scene_of(B).compute_closest_points(o3d.core.Tensor(pa))
        n = np.asarray(B.triangle_normals)[q["primitive_ids"].numpy()]
        qp = q["points"].numpy()
        if np.median(((qp - np.median(qp, 0)) * n).sum(1)) < 0:  # orient B's normals away from its centre
            n = -n
        off = ((pa - qp) * n).sum(1) * 1000
        h, e = np.histogram(off, bins=np.arange(-12, 13, 1.0))
        res["layering"] = {"of": a.layer_ref, "to": a.layer_to, "median_mm": round(float(np.median(off)), 3),
                           "share_outside_3mm": round(float((off > 3).mean()), 3), "share_inside_3mm": round(float((off < -3).mean()), 3),
                           "hist_edges_mm": e.tolist(), "hist": h.tolist()}
        print(f"layering of {a.layer_ref} along {a.layer_to} normals: median {np.median(off):+.2f} mm, "
              f">3 mm outside {(off > 3).mean():.0%}, >3 mm inside {(off < -3).mean():.0%}")
        print("hist (1 mm bins from -12):", h.tolist())
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
