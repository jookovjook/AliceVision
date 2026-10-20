import json, sys, numpy as np, trimesh
rd = json.load(open(sys.argv[1])); arc = sys.argv[2]; out = sys.argv[3]
L = json.load(open(f"{arc}/tags/frame_tags_layout.json"))
scr = trimesh.load(f"{arc}/tags/screw_wing.stl")
parts = [trimesh.load(f"{arc}/tags/frame_L116_{p}_plates.stl") for p in "AB"]
for r in rd["screws"]:
    if "s_mm" not in r: continue
    per = r.get("per_tag", {})
    good = [v for v in per.values() if v["tilt_deg"] < 10] or [r]
    s = np.mean([v["s_mm"] for v in good]); th = np.radians(np.mean([v["theta_deg"] for v in good]))
    o = np.array(r["origin_mm"]); dn = np.array(r["dirn"]); dn /= np.linalg.norm(dn)
    e1 = np.array([1.0, 0, 0]) if abs(dn[0]) < 0.5 else np.array([0, 1.0, 0]); e1 -= (e1 @ dn) * dn; e1 /= np.linalg.norm(e1); e2 = np.cross(dn, e1)
    x = np.cos(th) * e1 + np.sin(th) * e2; y = np.cross(dn, x)
    T = np.eye(4); T[:3, :3] = np.c_[x, y, dn]; T[:3, 3] = o + s * dn
    m = scr.copy(); m.apply_transform(T); parts.append(m)
cad = trimesh.util.concatenate(parts); cad.apply_scale(0.001); cad.export(out)
print("placed", len(parts) - 2, "screws ->", out)
