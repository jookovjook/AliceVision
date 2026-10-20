"""Distance from CAD screws, placed at their tag-read depth and wing angle on the hole axes, to a reconstructed mesh.

    python check_screws.py READOUT.json ARC_DIR MESH.ply OUT.json

Per screw: median distance (mm) from points sampled on the placed CAD screw to the nearest mesh vertex, split into
the wing (screw z < 0), the shaft outside the frame, and the part inside the frame / at the head. Small wing and
shaft values confirm the depth and angle read from the tags; the inside part touches hair, so it is looser.
"""
import json
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main():
    rd, arc, mesh, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
    tree = cKDTree(trimesh.load(mesh, process=False).vertices * 1000)
    S, _ = trimesh.sample.sample_surface_even(trimesh.load(f"{arc}/tags/screw_wing.stl"), 4000)
    zl = S[:, 2]
    res = []
    for r in rd["screws"]:
        if "s_mm" not in r:
            continue
        good = [v for v in r.get("per_tag", {}).values() if v["tilt_deg"] < 10] or [r]
        s = np.mean([v["s_mm"] for v in good]); th = np.radians(np.mean([v["theta_deg"] for v in good]))
        o = np.array(r["origin_mm"]); dn = np.array(r["dirn"]); dn /= np.linalg.norm(dn)
        e1 = np.array([1.0, 0, 0]) if abs(dn[0]) < 0.5 else np.array([0, 1.0, 0]); e1 -= (e1 @ dn) * dn; e1 /= np.linalg.norm(e1)
        e2 = np.cross(dn, e1); x = np.cos(th) * e1 + np.sin(th) * e2; y = np.cross(dn, x)
        d, _ = tree.query(S @ np.c_[x, y, dn].T + o + s * dn)
        med = lambda m: float(np.median(d[m])) if m.any() else None
        res.append({"hole": r["hole"], "s_mm": float(s), "wing": med(zl < 0), "shaft_outside": med((zl >= 0) & (zl < -s - 5)),
                    "inside": med(zl >= -s)})
        print(f"hole {r['hole']:2d}: wing {res[-1]['wing']:5.1f} mm, shaft outside {res[-1]['shaft_outside'] or float('nan'):5.1f} mm, inside {res[-1]['inside'] or float('nan'):5.1f} mm")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
