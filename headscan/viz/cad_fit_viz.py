"""Pictures and viewer assets for cad_fit.py: CAD parts placed by a fit, coloured by the distance to the mesh.

    python cad_fit_viz.py IN_DIR FIT_DIR OUT_SITE_DIR

Writes, into OUT_SITE_DIR: cadfit_<variant>.glb.txt (base64 GLB, device frame -> three.js axes; colour = distance from
the CAD surface to the nearest reconstructed vertex: 0 mm blue -> 2.5 mm white -> 5 mm red, grey = nothing within 8 mm,
i.e. hidden or not reconstructed), fig/cadfit_tags.jpg, fig/cadfit_ruler.jpg and cad_fit_summary.json.
"""
import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

GL = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
NAMES = [f"sag{i + 1}" for i in range(8)] + [f"cor{i + 1}" for i in range(4)]


def colour(d):
    t = np.clip(d / 5.0, 0, 1)[:, None]
    lo, mid, hi = np.array([40, 90, 200]), np.array([245, 245, 245]), np.array([200, 50, 40])
    c = np.where(t < 0.5, lo + (mid - lo) * (t / 0.5), mid + (hi - mid) * ((t - 0.5) / 0.5))
    c[d > 8] = [150, 150, 150]
    return c.astype(np.uint8)


def main():
    I, F, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    (S / "fig").mkdir(parents=True, exist_ok=True)
    J = json.load(open(F / "cad_fit.json"))
    tree = cKDTree(np.asarray(trimesh.load(I / "ss_tetra.ply", process=False).vertices) * 1000)
    parts = {}
    for p in "AB":
        m = trimesh.load(I / f"frame_L116_{p}_plates.stl")
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=2.0)
        parts[p] = trimesh.Trimesh(v, f, process=False)
    Tr = J["transforms"]
    SEG = J["segments"]["T"]
    variants = {"global_tags": {p: Tr["G_sim"] for p in "AB"}, "per_part": {p: Tr[f"P_mixs_{p}"] for p in "AB"},
                "segments": {"A": Tr["P_mix_A"], "B": None}}
    stats = {}
    for name, Ts in variants.items():
        ms = []; stats[name] = {}
        for p, m in parts.items():
            if Ts[p] is None:   # part B in rigid segments by x (right > 25 mm, left < -25 mm, apex block between)
                x = m.vertices[:, 0]; V = np.zeros_like(m.vertices)
                for g, sel in (("R", x > 25), ("L", x < -25), ("C", np.abs(x) <= 25)):
                    T = np.array(SEG[g]); V[sel] = m.vertices[sel] @ T[:3, :3].T + T[:3, 3]
            else:
                T = np.array(Ts[p]); V = m.vertices @ T[:3, :3].T + T[:3, 3]
            d, _ = tree.query(V)
            vis = d < 8
            stats[name][p] = {"median_mm": float(np.median(d[vis])), "p90_mm": float(np.percentile(d[vis], 90)), "visible": float(vis.mean())}
            mm = trimesh.Trimesh(V / 1000.0 @ GL.T, m.faces, vertex_colors=np.c_[colour(d), np.full(len(d), 255, np.uint8)], process=False)
            ms.append(mm)
        glb = trimesh.util.concatenate(ms).export(file_type="glb")
        (S / f"cadfit_{name}.glb.txt").write_text(base64.b64encode(glb).decode())
        print(name, stats[name])

    sm = J["summary"]; V_ = J["variants"]
    tagsG = V_["G_sim"]["tag_res_mm"]; tagsP = V_["P_mixs"]["tag_res_mm"]; tagsS = V_["S_mix"]["tag_res_mm"]
    ks = sorted(tagsG, key=int)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(ks))
    ax.bar(x - 0.27, [tagsG[k] for k in ks], 0.27, label="one rigid fit (global)", color="#9aa2ad")
    ax.bar(x, [tagsP[k] for k in ks], 0.27, label="parts A and B fitted separately", color="#1f4fa3")
    ax.bar(x + 0.27, [tagsS[k] for k in ks], 0.27, label="part B in 3 rigid segments", color="#c76b1c")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n{'A' if int(k) <= 6 else 'B'}" for k in ks]); ax.set_xlabel("frame tag, part")
    ax.set_ylabel("tag corner residual, mm"); ax.legend(fontsize=8); ax.set_title(f"Frame tags vs CAD: part A rigid, part B bends ({J['segments']['bend_deg']['R_vs_L']:.1f} deg across the arch)")
    plt.tight_layout(); plt.savefig(S / "fig/cadfit_tags.jpg", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for n, lab, c in (("G_sim", "global fit (before)", "#9aa2ad"), ("S_mix", "segment fit, uncalibrated", "#1f4fa3")):
        e = sm[n]["fine_err"]; ax.plot(range(12), e, "o-", color=c, label=f"{lab}: mean {sm[n]['fine_mae']:.2f}, max {sm[n]['fine_max']:.2f} mm")
    cal = J["calibration"]["S_mix"]
    ax.plot(range(12), cal["loo_err"], "o-", color="#c76b1c", label=f"segment fit, calibrated (leave-one-out): mean {cal['loo_mae']:.2f}, max {cal['loo_max']:.2f} mm")
    ax.axhline(0, color="k", lw=0.8); ax.axhspan(-0.5, 0.5, color="#2e7d4f", alpha=0.08)
    ax.set_xticks(range(12)); ax.set_xticklabels(NAMES); ax.set_ylabel("tags (fine) - ruler, mm"); ax.legend(fontsize=8)
    ax.set_title("Screw gaps vs your ruler")
    plt.tight_layout(); plt.savefig(S / "fig/cadfit_ruler.jpg", dpi=110); plt.close()
    json.dump({"summary": sm, "calibration": J["calibration"], "segments": {k: v for k, v in J["segments"].items() if k != "T"}, "joint_B_rel_A": J["joint_B_rel_A"], "heatmap": stats, "ruler": J["ruler"],
               "tag_res": {"G_sim": tagsG, "P_mixs": tagsP, "S_mix": tagsS},
               "fine": {n: [r["fine"] if r else None for r in V_[n]["readouts"]] for n in ("G_sim", "P_mixs", "L_mix")}},
              open(S / "cad_fit_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
