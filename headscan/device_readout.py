"""Step 8: device readouts from the reconstructed screw/knob tags + where every hole axis meets the head surface.

    python device_readout.py DEVICE_SFM TAG_FEATURES_DIR frame_tags_layout.json HEAD_CLOSED_FLAGS.ply OUT_PREFIX

Tag corners: tag36h11 landmarks of the device-frame sfm (metres -> mm), marker index = tagId*4 + corner (decoded
from the observations' one-hot descriptors; for duplicate ids the longest track wins).
Per screw k (tags 20+2k, 21+2k, corners in the screw frame of frame_tags_layout.json 'screws'):
  free 6-DoF rigid fit screw frame -> device frame (Kabsch, no scale) from all reconstructed corners, then
  - tilt   = angle between the fitted screw +z and the hole 'dirn' (should be ~0: the screw is in the hole),
  - offset = distance of the fitted screw axis from the hole axis at the hole origin (should be ~0),
  - s      = axial position of the screw-frame origin along the hole axis from the hole origin (mm),
  - tip    = s + 70 mm (tip at z = 70 in the screw frame),
  - theta  = wing angle about the hole axis (deg, provisional zero: see 'theta_ref'),
  - rms    = corner residual of the fit (mm).
Per hole: the hole axis is intersected with the head surface (Poisson-closed head of step 7; hits on flagged
'filled' vertices are marked unmeasured): surface distance from the hole origin along dirn (mm).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import trimesh


def marker_corners(sfm, feat_dir):
    d = json.load(open(sfm))
    cache, best = {}, {}
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
        if len(ids) != 1:
            continue
        i = next(iter(ids)); nobs = len(l["observations"])
        if i not in best or nobs > best[i][0]:
            best[i] = (nobs, np.array([float(x) for x in l["X"]]) * 1000.0)
    return {i: v[1] for i, v in best.items()}, {i: v[0] for i, v in best.items()}


def kabsch(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mq - R @ mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sfm"); ap.add_argument("feat"); ap.add_argument("layout"); ap.add_argument("head"); ap.add_argument("out")
    ap.add_argument("--skin", help="measured skin/hair mesh (head_skin.ply): surface under each screw from a ring of 8-20 mm around the axis")
    a = ap.parse_args()
    L = json.load(open(a.layout))
    X, nobs = marker_corners(a.sfm, a.feat)
    head = trimesh.load(a.head, process=False)
    skin = trimesh.load(a.skin, process=False).vertices * 1000.0 if a.skin else None
    magenta = np.all(head.visual.vertex_colors[:, :3] == [230, 0, 200], axis=1)
    rows = []
    for k, hole in enumerate(L["holes"]):
        o = np.array(hole["origin"], float); dn = np.array(hole["dirn"], float); dn /= np.linalg.norm(dn)
        e1 = np.array([1.0, 0, 0]) if abs(dn[0]) < 0.5 else np.array([0, 1.0, 0])  # provisional theta zero
        e1 = e1 - (e1 @ dn) * dn; e1 /= np.linalg.norm(e1); e2 = np.cross(dn, e1)
        r = {"hole": k, "kind": hole["kind"], "part": hole["part"], "origin_mm": o.tolist(), "dirn": dn.tolist()}
        # head surface along the axis
        loc, _, tri = head.ray.intersects_location([o / 1000.0], [dn], multiple_hits=False)
        if len(loc):
            hit = loc[0] * 1000.0
            fv = head.faces[tri[0]]
            r["surface_mm_from_origin"] = float((hit - o) @ dn)
            r["surface_hit_unmeasured"] = bool(magenta[fv].any())
        if skin is not None:
            tt = (skin - o) @ dn; rr = np.linalg.norm((skin - o) - tt[:, None] * dn, axis=1)
            ring = (rr > 8) & (rr < 20) & (tt > -30) & (tt < 90)
            if ring.sum() > 30:
                # local plane through the ring points; where the axis crosses it
                Pr = skin[ring]; c0 = Pr.mean(0); nrm = np.linalg.svd(Pr - c0)[2][2]
                r["ring_points"] = int(ring.sum())
                r["skin_ring_spread_mm"] = float(np.std((Pr - c0) @ nrm))
                r["skin_ring_cos"] = float(abs(dn @ nrm))
                # valid only if the local surface faces the axis (not a wall parallel to it) and is not too rough
                if abs(dn @ nrm) > 0.5 and r["skin_ring_spread_mm"] < 6.0:
                    r["skin_ring_mm_from_origin"] = float((c0 - o) @ nrm / (dn @ nrm))
        # screw pose from its tags
        tags = L["screws"][str(k)]["tags"]
        P, Q, used = [], [], []
        for tid, corners in tags.items():
            for c, pl in enumerate(corners):
                idx = int(tid) * 4 + c
                if idx in X:
                    P.append(pl); Q.append(X[idx]); used.append(idx)
        r["tags_seen"] = sorted({i // 4 for i in used}); r["corners"] = len(used)
        if len(used) >= 4:
            P, Q = np.array(P, float), np.array(Q, float)
            R, t = kabsch(P, Q)
            res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
            z = R[:, 2]; xs = R[:, 0]
            org = t                                   # screw-frame origin in the device frame
            s = float((org - o) @ dn)
            off = float(np.linalg.norm((org - o) - s * dn))
            r.update({"fit_rms_mm": float(np.sqrt((res ** 2).mean())), "tilt_deg": float(np.degrees(np.arccos(np.clip(z @ dn, -1, 1)))),
                      "axis_offset_mm": off, "s_mm": s, "tip_mm_from_origin": s + 70.0,
                      "theta_deg": float(np.degrees(np.arctan2(xs @ e2, xs @ e1))) % 360,
                      "theta_ref": "screw +x vs (device x if the axis is mostly in y-z, else device y) projected on the hole plane",
                      "min_track_obs": int(min(nobs[i] for i in used))})
            if "skin_ring_mm_from_origin" in r:
                r["tip_minus_skin_mm"] = r["tip_mm_from_origin"] - r["skin_ring_mm_from_origin"]
            # each sticker alone (consistency of the two faces of the wing)
            per = {}
            for tid, corners in tags.items():
                ii = [int(tid) * 4 + c for c in range(4)]
                if all(i in X for i in ii):
                    Rt, tt2 = kabsch(np.array(corners, float), np.array([X[i] for i in ii]))
                    per[tid] = {"tilt_deg": float(np.degrees(np.arccos(np.clip(Rt[:, 2] @ dn, -1, 1)))),
                                "s_mm": float((tt2 - o) @ dn), "theta_deg": float(np.degrees(np.arctan2(Rt[:, 0] @ e2, Rt[:, 0] @ e1))) % 360}
            r["per_tag"] = per
        rows.append(r)
    knobs = {}
    for kn, info in L.get("knobs", {}).items():
        P, Q = [], []
        for tid, corners in info["tags"].items():
            for c, pl in enumerate(corners):
                if int(tid) * 4 + c in X:
                    P.append(pl); Q.append(X[int(tid) * 4 + c])
        if len(P) >= 4:
            R, t = kabsch(np.array(P, float), np.array(Q, float))
            knobs[kn] = {"corners": len(P), "origin_mm": t.tolist(), "z_axis": R[:, 2].tolist(),
                         "axis_vs_device_x_deg": float(np.degrees(np.arccos(abs(R[:, 2][0]))))}
    out = {"screws": rows, "knobs": knobs, "markers_reconstructed": len(X)}
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(f"{'hole':>4} {'kind':4} {'tags':10} {'rms':>5} {'tilt':>5} {'off':>5} {'s':>7} {'tip':>7} {'skin':>7} {'tip-skin':>8} {'theta':>6}")
    for r in rows:
        f = lambda key, fmt: (fmt % r[key]) if key in r else "-"
        print(f"{r['hole']:4d} {r['kind']:4} {str(r['tags_seen']):10} {f('fit_rms_mm', '%5.2f'):>5} {f('tilt_deg', '%5.1f'):>5} "
              f"{f('axis_offset_mm', '%5.1f'):>5} {f('s_mm', '%7.1f'):>7} {f('tip_mm_from_origin', '%7.1f'):>7} "
              f"{f('skin_ring_mm_from_origin', '%7.1f'):>7}  {f('tip_minus_skin_mm', '%8.1f'):>8} {f('theta_deg', '%6.1f'):>6}  per-tag " +
              " ".join(f"{t}:tilt{v['tilt_deg']:.0f}/s{v['s_mm']:.0f}/th{v['theta_deg']:.0f}" for t, v in r.get("per_tag", {}).items()))
    print("knobs:", json.dumps(knobs))


if __name__ == "__main__":
    main()
