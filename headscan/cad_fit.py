"""Piecewise fit of the frame CAD to the reconstruction, and screw readouts in each fitted frame.

    python cad_fit.py IN_DIR OUT_DIR [--ruler 3.5,4,...]

IN_DIR: sfm_device.sfm + feat_tags/ (reconstructed tag corners), frame_L116_{A,B}_plates.stl + frame_tags_layout.json
(CAD, device frame, mm), ss_tetra.ply (reconstructed mesh, device frame, metres).
A fit is a transform T: CAD (nominal device frame) -> scene. Variants:
  G_sim   tags only, one similarity (the step-4 method: free scale)
  G_rig   tags only, one rigid transform (scale fixed by the CAD)
  G_icp   shape only: rigid ICP of the CAD surface onto the mesh (from G_rig)
  G_mix   tags + shape, one rigid transform
  G_mixs  tags + shape, one similarity (scale check)
  P_mix   tags + shape, part A and part B separately (rigid); P_mixs the same with free scale per part
  L_mix   per screw: tags + shape of the screw's own part within 70 mm of its hole (rigid)
  S_mix   part A rigid + part B split into rigid segments (right arch + barrel x > 25 mm, left arch + barrel
          x < -25 mm, apex block |x| <= 25 mm), each by its own tags + shape, started from the part-B fit
Calibration (--ruler): gap = phase + 2.5 n with two constants from the ruler, b (coarse bias, picks the turn n)
and phi (thread-phase offset); judged leave-one-out (each screw predicted with b, phi from the other 11).
Shape term: point-to-plane on CAD surface samples whose nearest mesh point is within 5 mm, best 80 % kept.
Screw readouts per variant: screw pose from its wing-tag corners (Kabsch, stickers off the axis by > 10 deg dropped),
hole axis taken from the variant's transform; gap_coarse = s + 74 mm; fine = wing angle psi from the hole-thread
reference e1_k (2.5 mm per turn, nearest turn to the coarse value). Compared with the ruler when given.
Writes OUT_DIR/cad_fit.json, figures and CAD GLBs coloured by the distance to the mesh.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step8_report import E1  # noqa: E402  (hole-thread reference axes from the CAD)

P = 2.5


def marker_corners(sfm, feat_dir):
    d = json.load(open(sfm)); cache, best = {}, {}
    for l in d["structure"]:
        if l["descType"] != "tag36h11":
            continue
        ids = Counter()
        for o in l["observations"]:
            vid = int(o["observationId"])
            if vid not in cache:
                raw = (Path(feat_dir) / f"{vid}.tag36h11.desc").read_bytes()
                n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
                cache[vid] = np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256).argmax(1)
            ids[int(cache[vid][int(o["featureId"])])] += 1
        if len(ids) == 1:
            i = next(iter(ids))
            if i not in best or len(l["observations"]) > best[i][0]:
                best[i] = (len(l["observations"]), np.array([float(x) for x in l["X"]]) * 1000.0)
    return {i: v[1] for i, v in best.items()}


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def apply(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def solve(T0, tagP, tagQ, icpS, tree, tgtN, tgtX, scale=False, w_icp=1.0, iters=40, max_d=5.0, keep=0.8):
    """Gauss-Newton on [rot(3), trans(3), log-scale] minimising tag point-to-point + ICP point-to-plane (mm)."""
    T = T0.copy()
    for _ in range(iters):
        rows, rhs = [], []
        if tagP is not None and len(tagP):
            Y = apply(T, tagP); r = (tagQ - Y)
            for i in range(len(Y)):
                J = np.zeros((3, 7)); J[:, :3] = -skew(Y[i] - T[:3, 3]); J[:, 3:6] = np.eye(3); J[:, 6] = Y[i] - T[:3, 3]
                rows.append(J); rhs.append(r[i])
        if icpS is not None and w_icp > 0:
            Y = apply(T, icpS); d, j = tree.query(Y)
            ok = d < max_d
            if ok.sum() > 50:
                thr = np.quantile(d[ok], keep); ok &= d <= thr
                Yo, X, N = Y[ok], tgtX[j[ok]], tgtN[j[ok]]
                res = ((X - Yo) * N).sum(1)
                Jn = np.zeros((len(Yo), 7))
                Jn[:, :3] = np.cross(Yo - T[:3, 3], N); Jn[:, 3:6] = N; Jn[:, 6] = ((Yo - T[:3, 3]) * N).sum(1)
                wgt = w_icp * np.sqrt(max(1, len(tagP) if tagP is not None else 1) / len(Yo))
                rows.append(Jn * wgt); rhs.append(res * wgt)
        A = np.vstack([r if r.ndim == 2 else r[None] for r in rows]); b = np.concatenate([np.atleast_1d(x).ravel() for x in rhs])
        if not scale:
            A = A[:, :6]
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        dR = rodrigues(x[:3]); ds = np.exp(x[6]) if scale else 1.0
        c = T[:3, 3].copy()
        T[:3, :3] = ds * dR @ T[:3, :3]; T[:3, 3] = c + x[3:6]
        if np.linalg.norm(x[:6]) < 1e-6:
            break
    return T


def umeyama(X, Y, scale=True):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (X - mx).var(0).sum() if scale else 1.0
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = my - s * R @ mx
    return T


def kabsch(P_, Q):
    T = umeyama(P_, Q, scale=False); return T[:3, :3], T[:3, 3]


def tscale(T):
    return float(np.cbrt(np.linalg.det(T[:3, :3])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--ruler", default="")
    a = ap.parse_args()
    I, O = Path(a.inp), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    L = json.load(open(I / "frame_tags_layout.json"))
    X = marker_corners(I / "sfm_device.sfm", I / "feat_tags")
    parts = {p: trimesh.load(I / f"frame_L116_{p}_plates.stl") for p in "AB"}
    mesh = trimesh.load(I / "ss_tetra.ply", process=False)
    tgtX = np.asarray(mesh.vertices) * 1000.0; tgtN = np.asarray(mesh.vertex_normals)
    tree = cKDTree(tgtX)
    # CAD samples per part (outer + inner surfaces; ICP trimming keeps what the mesh actually shows)
    samp = {p: trimesh.sample.sample_surface_even(m, 12000, seed=1)[0] for p, m in parts.items()}
    # frame-tag corners and their part
    part_of_tag = {}
    for t, cs in L["frame"].items():
        c = np.array(cs).mean(0)
        part_of_tag[int(t)] = min("AB", key=lambda p: trimesh.proximity.signed_distance(parts[p], [c])[0] ** 2)
    tags = [(int(t) * 4 + k, np.array(c, float), part_of_tag[int(t)]) for t, cs in L["frame"].items() for k, c in enumerate(cs) if int(t) * 4 + k in X]
    ids = np.array([i for i, _, _ in tags]); Pc = np.array([c for _, c, _ in tags]); Qm = np.array([X[i] for i in ids]); prt = np.array([p for _, _, p in tags])
    print(f"{len(X)} reconstructed corners; frame-tag corners used: {len(ids)} (A {np.sum(prt == 'A')}, B {np.sum(prt == 'B')})")

    allS = np.vstack([samp["A"], samp["B"]])
    V = {}
    V["G_sim"] = umeyama(Pc, Qm, True)
    V["G_rig"] = umeyama(Pc, Qm, False)
    V["G_icp"] = solve(V["G_rig"], None, None, allS, tree, tgtN, tgtX)
    V["G_mix"] = solve(V["G_rig"], Pc, Qm, allS, tree, tgtN, tgtX)
    V["G_mixs"] = solve(V["G_rig"], Pc, Qm, allS, tree, tgtN, tgtX, scale=True)
    for p in "AB":
        m = prt == p
        V[f"P_mix_{p}"] = solve(V["G_mix"], Pc[m], Qm[m], samp[p], tree, tgtN, tgtX)
        V[f"P_mixs_{p}"] = solve(V["G_mix"], Pc[m], Qm[m], samp[p], tree, tgtN, tgtX, scale=True)

    def per_tag_res(Tof):
        out = {}
        for i, c, p in tags:
            out.setdefault(i // 4, []).append(float(np.linalg.norm(apply(Tof(p), c[None])[0] - X[i])))
        return {t: float(np.mean(v)) for t, v in out.items()}

    def icp_dist(Tof):
        res = {}
        for p in "AB":
            d, _ = tree.query(apply(Tof(p), samp[p])); d = d[d < 5]
            res[p] = {"median_mm": float(np.median(d)), "p90_mm": float(np.percentile(d, 90)), "n": int(len(d))}
        return res

    fits = {"G_sim": lambda p: V["G_sim"], "G_rig": lambda p: V["G_rig"], "G_icp": lambda p: V["G_icp"],
            "G_mix": lambda p: V["G_mix"], "G_mixs": lambda p: V["G_mixs"], "P_mix": lambda p: V[f"P_mix_{p}"],
            "P_mixs": lambda p: V[f"P_mixs_{p}"]}
    # local per-screw fits
    holes = L["holes"]; local = {}
    for k, h in enumerate(holes):
        p = h["part"]; o = np.array(h["origin"])
        m = (prt == p) & (np.linalg.norm(Pc - o, axis=1) < 90)
        if m.sum() < 8:  # need 2 tags; take the nearest 2 tags of the part
            dd = np.where(prt == p, np.linalg.norm(Pc - o, axis=1), np.inf); m = dd <= np.sort(dd)[min(7, len(dd) - 1)]
        S = samp[p][np.linalg.norm(samp[p] - o, axis=1) < 70]
        local[k] = solve(V[f"P_mix_{p}"], Pc[m], Qm[m], S, tree, tgtN, tgtX)

    # part B in rigid segments
    def seg_of(pts):
        x = np.asarray(pts)[:, 0]
        return np.where(x > 25, "R", np.where(x < -25, "L", "C"))
    segP = seg_of(Pc); segS = seg_of(samp["B"]); SEG = {}
    for g in "RLC":
        m = (prt == "B") & (segP == g)
        SEG[g] = solve(V["P_mix_B"], Pc[m], Qm[m], samp["B"][segS == g], tree, tgtN, tgtX) if m.sum() >= 4 else V["P_mix_B"]
    seg_tags = {g: sorted({int(i) // 4 for i in ids[(prt == "B") & (segP == g)]}) for g in "RLC"}
    def T_seg_of(p, pt):
        return V["P_mix_A"] if p == "A" else SEG[str(seg_of(np.asarray(pt, float).reshape(-1, 3))[0])]
    def rel_angle(Ta, Tb):
        Rr = (np.linalg.inv(Ta) @ Tb)[:3, :3]; return float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
    bend = {f"{g}_vs_C": rel_angle(SEG["C"], SEG[g]) for g in "RL"}
    bend["R_vs_L"] = rel_angle(SEG["R"], SEG["L"])
    print("part-B segments, tags:", seg_tags, " bending (deg):", {k: round(v, 2) for k, v in bend.items()})

    ruler = [float(v) for v in a.ruler.split(",")] if a.ruler else None

    def readouts(Tfor_hole):
        rows = []
        for k, h in enumerate(holes):
            T = Tfor_hole(k)
            R = T[:3, :3] / tscale(T)
            o = apply(T, np.array(h["origin"])[None])[0]; dn = R @ (np.array(h["dirn"]) / np.linalg.norm(h["dirn"]))
            e1 = R @ np.array(E1[k], float); e1 -= (e1 @ dn) * dn; e1 /= np.linalg.norm(e1); e2 = np.cross(dn, e1)
            per = []
            for tid, cs in L["screws"][str(k)]["tags"].items():
                ii = [int(tid) * 4 + c for c in range(4)]
                if all(i in X for i in ii):
                    Rs, ts = kabsch(np.array(cs, float), np.array([X[i] for i in ii]))
                    if np.degrees(np.arccos(np.clip(Rs[:, 2] @ dn, -1, 1))) < 10:
                        per.append((float((ts - o) @ dn), Rs[:, 0]))
            if not per:
                rows.append(None); continue
            s = np.mean([q[0] for q in per]); xs = np.mean([q[1] for q in per], 0); xs -= (xs @ dn) * dn
            psi = np.arctan2(xs @ e2, xs @ e1); sf = P * psi / (2 * np.pi)
            sfine = sf + np.round((s - sf) / P) * P
            rows.append({"coarse": float(s + 74), "fine": float(sfine + 74), "phase_mm": float((sf + 74) % P)})
        return rows

    variants = {}
    for name, Tof in fits.items():
        variants[name] = {"scale": {p: tscale(Tof(p)) for p in "AB"}, "tag_res_mm": per_tag_res(Tof), "icp": icp_dist(Tof),
                          "readouts": readouts(lambda k, Tof=Tof: Tof(holes[k]["part"]))}
    seg_res = {}
    for i, c, p in tags:
        seg_res.setdefault(i // 4, []).append(float(np.linalg.norm(apply(T_seg_of(p, c[None]), c[None])[0] - X[i])))
    variants["S_mix"] = {"scale": {"A": 1.0, "B": 1.0}, "tag_res_mm": {t: float(np.mean(v)) for t, v in seg_res.items()}, "icp": None,
                         "readouts": readouts(lambda k: T_seg_of(holes[k]["part"], np.array(holes[k]["origin"])[None])),
                         "segments": seg_tags, "bend_deg": bend}
    variants["L_mix"] = {"scale": {"A": 1.0, "B": 1.0}, "tag_res_mm": None, "icp": None, "readouts": readouts(lambda k: local[k])}
    # joint between the parts: B relative to A after the per-part fit
    TA, TB = V["P_mix_A"], V["P_mix_B"]; Rel = np.linalg.inv(TA) @ TB
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rel[:3, :3]) - 1) / 2, -1, 1))))
    joint = {"rotation_deg": ang, "translation_mm": Rel[:3, 3].round(2).tolist()}

    summary = {}
    for name, v in variants.items():
        row = {"scale": v["scale"]}
        if v["tag_res_mm"]:
            r = np.array(list(v["tag_res_mm"].values())); row["tag_res_rms_mm"] = float(np.sqrt((r ** 2).mean()))
        if v["icp"]:
            row["icp_median_mm"] = {p: v["icp"][p]["median_mm"] for p in "AB"}
        if ruler:
            for kind in ("coarse", "fine"):
                e = np.array([(r[kind] - g) if r else np.nan for r, g in zip(v["readouts"], ruler)])
                row[f"{kind}_err"] = e.round(2).tolist()
                row[f"{kind}_mae"] = float(np.nanmean(np.abs(e))); row[f"{kind}_bias"] = float(np.nanmedian(e))
                row[f"{kind}_max"] = float(np.nanmax(np.abs(e)))
        summary[name] = row
    # calibration: b (coarse bias) and phi (phase offset), leave-one-out
    calib = {}
    if ruler:
        g = np.array(ruler)
        for name, v in variants.items():
            rs = v["readouts"]
            if any(r is None for r in rs):
                continue
            co = np.array([r["coarse"] for r in rs]); ph = np.array([r["fine"] for r in rs])
            def predict(tr, te):
                b = np.median(co[tr] - g[tr])
                w = ((ph[tr] - g[tr] + P / 2) % P) - P / 2; phi = np.median(w)
                f = ph[te] - phi; n = np.round(((co[te] - b) - f) / P)
                return f + n * P, b, phi
            loo = np.array([predict(np.delete(np.arange(12), i), np.array([i]))[0][0] for i in range(12)])
            full, b, phi = predict(np.arange(12), np.arange(12))
            calib[name] = {"b_mm": float(b), "phi_mm": float(phi), "loo_err": (loo - g).round(2).tolist(),
                           "loo_mae": float(np.mean(np.abs(loo - g))), "loo_max": float(np.max(np.abs(loo - g))),
                           "in_sample_mae": float(np.mean(np.abs(full - g))), "calibrated_mm": full.round(2).tolist()}
        print(f"{'variant':8} {'b':>6} {'phi':>6} {'LOO MAE':>8} {'LOO max':>8} {'in-sample':>9}")
        for n_, c in calib.items():
            print(f"{n_:8} {c['b_mm']:6.2f} {c['phi_mm']:6.2f} {c['loo_mae']:8.2f} {c['loo_max']:8.2f} {c['in_sample_mae']:9.2f}   {c['loo_err']}")
    json.dump({"variants": variants, "summary": summary, "calibration": calib, "segments": {"tags": seg_tags, "bend_deg": bend, "T": {g: SEG[g].tolist() for g in SEG}}, "joint_B_rel_A": joint, "ruler": ruler,
               "transforms": {k: v.tolist() for k, v in V.items()}, "local": {str(k): v.tolist() for k, v in local.items()}},
              open(O / "cad_fit.json", "w"), indent=1)
    print(f"joint B relative to A after the per-part fit: rotation {ang:.2f} deg, translation {np.round(Rel[:3, 3], 2).tolist()} mm")
    print(f"{'variant':8} {'scaleA':>7} {'scaleB':>7} {'tagRMS':>6} {'icpA':>5} {'icpB':>5} {'c.bias':>6} {'c.MAE':>6} {'f.MAE':>6} {'f.max':>6}")
    for n, r in summary.items():
        g = lambda k, f="%6.2f": (f % r[k]) if k in r else "     -"
        print(f"{n:8} {r['scale']['A']:7.4f} {r['scale']['B']:7.4f} {g('tag_res_rms_mm')} "
              f"{(r['icp_median_mm']['A'] if 'icp_median_mm' in r else float('nan')):5.2f} {(r['icp_median_mm']['B'] if 'icp_median_mm' in r else float('nan')):5.2f} "
              f"{g('coarse_bias')} {g('coarse_mae')} {g('fine_mae')} {g('fine_max')}")
    if ruler:
        print("coarse errors vs ruler:")
        for n, r in summary.items():
            print(f"  {n:8} {r['coarse_err']}")


if __name__ == "__main__":
    main()
