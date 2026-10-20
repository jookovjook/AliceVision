"""Layer / winding check of head meshes: rays from a point inside the head out to the sphere.

    python mesh_layers.py LABEL=MESH.ply [...] [--from 0.02 0.04 0.06 --rays 20000 --min-z -0.2]

Per mesh: hits per ray (a single-layer surface of a star-shaped head gives 1 almost everywhere; a TSDF with an
unmasked truncation band gives 2, ~trunc apart), the gap between the outer two hits, and the share of rays whose
outermost hit has its (winding) normal pointing outward. Rays pointing down (z < --min-z) are skipped (neck, torso).
"""
import argparse

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+"); ap.add_argument("--from", dest="src", type=float, nargs=3, default=[0.02, 0.04, 0.06])
    ap.add_argument("--rays", type=int, default=20000); ap.add_argument("--min-z", type=float, default=-0.2)
    a = ap.parse_args()
    ctr = np.array(a.src, np.float32)
    d = np.random.default_rng(0).normal(size=(a.rays, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > a.min_z].astype(np.float32)
    for lab, p in (m.split("=", 1) for m in a.meshes):
        m = o3d.io.read_triangle_mesh(p); m.compute_triangle_normals(); N = np.asarray(m.triangle_normals)
        sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        li = sc.list_intersections(o3d.core.Tensor(np.hstack([np.repeat(ctr[None], len(d), 0), d])))
        split, t, pid = li["ray_splits"].numpy(), li["t_hit"].numpy(), li["primitive_ids"].numpy()
        cnt = np.diff(split); out, gap = [], []
        for i in np.flatnonzero(cnt):
            s, e = split[i], split[i + 1]
            o = np.argsort(t[s:e]); tt, pp = t[s:e][o], pid[s:e][o]
            out.append(N[pp[-1]] @ d[i] > 0)
            if e - s >= 2:
                gap.append((tt[-1] - tt[-2]) * 1000)
        cnt = cnt[cnt > 0]
        print(f"{lab:>12}: {len(m.triangles)} faces; rays hitting {len(cnt)}, hits per ray median {np.median(cnt):.0f}, "
              f">=2 hits {np.mean(cnt >= 2):.0%}; outermost normal outward {np.mean(out):.0%}"
              + (f"; outer two hits gap median {np.median(gap):.1f} mm" if gap else ""), flush=True)


if __name__ == "__main__":
    main()
