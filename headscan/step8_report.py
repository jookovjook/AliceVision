"""Step 8 report: the 13 device readouts (sag1-8, cor1-4, ear_halfwidth) from the tags + head alignment.

    python step8_report.py READOUT.json FACE_LM.json OUT_PREFIX [--screw-check CHECK.json] [--title T]

Conventions (tfus-simulation CAD, see the step-8 notes in headscan/README or the commit message):
- hole k: origin o_k sits 4 mm inside the frame's INNER face, dirn d_k into the head; inner face point I_k = o_k - 4 d_k.
- screw frame origin (wing root) t = o_k + s d_k, tip at t + 70 d_k  ->  gap_k = s + 74 (inner face to tip, mm).
- fine reading: wing angle psi of the screw +x (outward normal of tag 20+2k's face) about d_k from the hole-thread
  reference e1_k (table below, trimesh align_vectors frame of the CAD cutter) -> s = 2.5 psi/360 (mod 2.5);
  the coarse s from the tag positions picks the turn. +-0.34 mm thread play.
- ear_halfwidth = 0.5 (|x_R| + |x_L|) of the innermost point of the PRINTED ear ball: the sound bore removes the
  nominal apex (ear-screw z 116 = knob z 123.5); the innermost ring is at ear-screw z 114.57 = knob z 122.07.
- simulated values: frame_rigid_geom.seat_head on Jack's MRI (L116, v1 skin), run read-only by the conventions
  reader: sag [19.0, 21.5, 24.4, 21.5, 20.3, 26.6, 31.5, 37.8], cor [17.0, 18.2, 20.1, 19.1], E 76.3, pitch +0.32 deg.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E1 = {0: [0.35659, 0.333148, -0.872844], 1: [0.680605, 0.498646, -0.536778], 2: [0.897958, 0.395175, -0.193672],
      3: [0.994321, 0.105818, -0.011326], 4: [0.891869, -0.403386, -0.204569], 5: [0.657495, -0.495395, -0.5677],
      6: [0.360145, -0.335978, -0.870296], 7: [0.019356, -0.019353, -0.999625], 8: [0.376284, 0, -0.926504],
      9: [0.772984, 0, -0.634426], 10: [0.772984, 0, 0.634426], 11: [0.376284, 0, 0.926504]}
SIM = [19.0, 21.5, 24.4, 21.5, 20.3, 26.6, 31.5, 37.8, 17.0, 18.2, 20.1, 19.1]
SIM_E, P = 76.3, 2.5
RING_Z, APEX_Z = 122.07, 123.5   # knob frame: printed ball's innermost ring, nominal apex
NAMES = [f"sag{i + 1}" for i in range(8)] + [f"cor{i + 1}" for i in range(4)]


def ear_from_knobs(knobs):
    out = {}
    for kn, side in (("0", +1), ("1", -1)):
        k = knobs[kn]; t = np.array(k["origin_mm"]); z = np.array(k["z_axis"])
        ring, apex = t + RING_Z * z, t + APEX_Z * z
        out[kn] = {"side": "right" if side > 0 else "left", "ball_ring_mm": ring.round(2).tolist(), "E_side_mm": float(side * ring[0]),
                   "E_side_nominal_apex_mm": float(side * apex[0]), "axis_vs_device_x_deg": k["axis_vs_device_x_deg"]}
    return {"knobs": out, "ear_halfwidth_mm": 0.5 * (out["0"]["E_side_mm"] + out["1"]["E_side_mm"]),
            "ear_halfwidth_nominal_apex_mm": 0.5 * (out["0"]["E_side_nominal_apex_mm"] + out["1"]["E_side_nominal_apex_mm"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readout"); ap.add_argument("face"); ap.add_argument("out")
    ap.add_argument("--screw-check"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    rd, face, out = json.load(open(a.readout)), json.load(open(a.face)), a.out
    ear = ear_from_knobs(rd["knobs"])
    chk = {c["hole"]: c for c in json.load(open(a.screw_check))} if a.screw_check else {}
    rows = []
    for r in rd["screws"]:
        k = r["hole"]; d = np.array(r["dirn"]); d /= np.linalg.norm(d)
        per = {t: v for t, v in r.get("per_tag", {}).items() if v["tilt_deg"] < 10}   # drop a mis-glued sticker
        row = {"hole": k, "readout": NAMES[k], "sim_gap": SIM[k], "tags_used": sorted(per)}
        if per:
            s_coarse = float(np.mean([v["s_mm"] for v in per.values()]))
            row["gap_coarse"] = s_coarse + 74.0
            # fine: psi of screw +x about d from e1_k; per-tag theta was measured from a different zero, so redo it
            # from the fitted rotation stored as theta_deg relative to (device x or y): convert via the reference axes
            e1 = np.array(E1[k], float); e1 -= (e1 @ d) * d; e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
            ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.5 else np.array([0, 1.0, 0])
            ref -= (ref @ d) * d; ref /= np.linalg.norm(ref); refp = np.cross(d, ref)
            th = np.radians(np.mean([v["theta_deg"] for v in per.values()]))
            xs = np.cos(th) * ref + np.sin(th) * refp
            psi = np.arctan2(xs @ e2, xs @ e1)
            s_fine = P * psi / (2 * np.pi)
            s = s_fine + np.round((s_coarse - s_fine) / P) * P
            row.update({"psi_deg": float(np.degrees(psi)) % 360, "gap_fine": float(s + 74.0),
                        "fine_minus_coarse": float(s - s_coarse)})
        if "skin_ring_mm_from_origin" in r:
            row["hair_surface_gap"] = r["skin_ring_mm_from_origin"] + 4.0
        if k in chk:
            row["cad_screw_vs_mesh_mm"] = {key: chk[k][key] for key in ("wing", "shaft_outside", "inside")}
        rows.append(row)
    E = ear["ear_halfwidth_mm"]
    L = face["landmarks"]
    get = lambda k: np.array(L[k]["mm"]) if k in L else None
    mid = [get(k)[0] for k in ("between_eyes_168", "glabella_9", "nose_tip_1", "subnasale_2", "chin_152") if get(k) is not None]
    sym = [0.5 * (get(a)[0] + get(b)[0]) for a, b in (("eye_R_inner_133", "eye_L_inner_362"), ("eye_R_outer_33", "eye_L_outer_263"), ("mouth_R_61", "mouth_L_291"))]
    R_ = np.array(ear["knobs"]["0"]["ball_ring_mm"]); L_ = np.array(ear["knobs"]["1"]["ball_ring_mm"]); o = 0.5 * (R_ + L_)
    n = get("between_eyes_168")
    x = R_ - L_; x /= np.linalg.norm(x); y = n - o; y -= (y @ x) * x; N_y = float(np.linalg.norm(y)); y /= np.linalg.norm(y); z = np.cross(x, y)
    Rdh = np.stack([x, y, z])                                    # device -> head (rows = head axes in device coords)
    pitch = float(np.degrees(np.arctan2(Rdh[2, 1], Rdh[1, 1])))  # + = nose up, simulation convention
    # yaw / roll of the face midline plane (normal ~ x) relative to the device
    M = np.array([get(k) for k in ("forehead_10", "glabella_9", "between_eyes_168", "nose_bridge_6", "nose_tip_1", "subnasale_2", "chin_152")])
    nrm = np.linalg.svd(M - M.mean(0))[2][2]; nrm *= np.sign(nrm[0])
    align = {"ear_halfwidth_mm": E, "ear_halfwidth_nominal_apex_mm": ear["ear_halfwidth_nominal_apex_mm"],
             "ear_ball_ring_R_mm": R_.tolist(), "ear_ball_ring_L_mm": L_.tolist(),
             "ear_ball_midpoint_x_mm": float(o[0]), "face_midline_x_mm": float(np.mean(mid + sym)),
             "face_midline_yaw_deg": float(np.degrees(np.arctan2(nrm[1], nrm[0]))), "face_midline_roll_deg": float(np.degrees(np.arctan2(nrm[2], nrm[0]))),
             "nasion_proxy_mm": n.tolist(), "N_y_mm": N_y, "pitch_deg": pitch,
             "device_nasion_point_minus_nasion_mm": (np.array([0, 116.0, 0]) - n).tolist()}
    json.dump({"readouts": rows, "alignment": align}, open(out + ".json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [3, 1.3]})
    xs_ = np.arange(12)
    meas = [r.get("gap_fine", np.nan) for r in rows]; coarse = [r.get("gap_coarse", np.nan) for r in rows]
    hair = [r.get("hair_surface_gap", np.nan) for r in rows]
    ax[0].bar(xs_ - 0.27, SIM, 0.27, color="#bbbbbb", label="simulated from your MRI (L116)")
    ax[0].bar(xs_, meas, 0.27, color="tab:blue", label="read from screw tags (tip, fine)")
    ax[0].bar(xs_ + 0.27, hair, 0.27, color="tab:orange", label="hair surface on the axis (geometry)")
    ax[0].plot(xs_, coarse, "k_", ms=14, label="tags, coarse (position only)")
    for i, r in enumerate(rows):
        ax[0].text(i, (meas[i] if np.isfinite(meas[i]) else 0) + 1.5, f"{meas[i]:.1f}", ha="center", fontsize=8, color="tab:blue")
    ax[0].set_xticks(xs_); ax[0].set_xticklabels(NAMES); ax[0].set_ylabel("gap: frame inner face -> skin along the hole axis, mm")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_title(("Screw readouts (model inputs sag1-8, cor1-4)  " + a.title).strip())
    ax[1].axis("off")
    txt = [f"ear_halfwidth (printed ball, innermost): {E:.1f} mm   [sim, LPA/RPA: {SIM_E}]",
           f"ear balls: R x = {R_[0]:+.1f}, L x = {L_[0]:+.1f} mm  (midpoint {o[0]:+.1f})",
           f"face midline x: {align['face_midline_x_mm']:+.1f} mm  (frame sits right of the face)",
           f"face midline yaw {align['face_midline_yaw_deg']:+.1f} deg, roll {align['face_midline_roll_deg']:+.1f} deg",
           f"nasion proxy (FaceMesh 168, under the arc: interpolated)",
           f"   {np.round(n, 1).tolist()} mm",
           f"   N_y = {N_y:.1f} mm from the ball axis [MRI: 114 (LPA/RPA), 108.5 (concha axis)]",
           f"   pitch = {pitch:+.1f} deg (nose up +)   [sim +0.3]",
           "", "screw 0: tag 20 mis-glued (90 deg), tag 21 used",
           "screw 6, 7, 8, 11: one sticker seen",
           "fine reading: wing angle, 2.5 mm per turn, +-0.34 mm play"]
    ax[1].text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8.5)
    plt.tight_layout(); plt.savefig(out + ".png", dpi=110)
    print(f"{'readout':7} {'sim':>5} {'coarse':>7} {'fine':>6} {'f-c':>5} {'psi':>6} {'hair':>6}")
    for r in rows:
        g = lambda k, f="%6.1f": (f % r[k]) if k in r else "     -"
        print(f"{r['readout']:7} {r['sim_gap']:5.1f} {g('gap_coarse', '%7.1f')} {g('gap_fine')} {g('fine_minus_coarse', '%5.1f')} {g('psi_deg')} {g('hair_surface_gap')}")
    print(json.dumps(align, indent=1))


if __name__ == "__main__":
    main()
