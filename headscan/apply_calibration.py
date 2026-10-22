"""Calibrated readouts from an image_fit.py result: per-screw constants from the table calibration.

    python apply_calibration.py IMAGE_FIT.json CONSTANTS.json OUT.json [--ruler 3.5,4,...]

Per screw k: coarse = s + 74 + b_k (tag position, per-screw bias); fine = 74 + pitch*psi/360 + phi_k (wing angle,
per-screw thread phase), turn chosen nearest the coarse value. 'gap_mm' = fine (best on the head: 0.32 mm mean vs ruler).
Knobs: K = K_coarse (within ~0.2 mm of the caliper without correction).
"""
import argparse
import json

import numpy as np

NAMES = [f"sag{i + 1}" for i in range(8)] + [f"cor{i + 1}" for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("constants"); ap.add_argument("out"); ap.add_argument("--ruler", default="")
    a = ap.parse_args()
    F, C = json.load(open(a.fit)), json.load(open(a.constants))
    sc = C["screws"]; P = sc["pitch"]
    rows = []
    for r in F["readouts"]:
        k = str(r["hole"])
        if not r.get("obs"):
            rows.append({"hole": r["hole"], "name": NAMES[r["hole"]], "gap_mm": None}); continue
        coarse = r["coarse"] + sc["b_k"].get(k, sc["b"])
        f = 74 + P * r["psi_deg"] / 360 + sc["phi_k"].get(k, sc["phi"])
        fine = f + np.round((coarse - f) / P) * P
        rows.append({"hole": r["hole"], "name": NAMES[r["hole"]], "gap_mm": round(float(fine), 2), "gap_coarse_mm": round(float(coarse), 2),
                     "gap_raw_mm": round(r["coarse"], 2), "play_mm": round(float(fine - coarse), 2)})
    knobs = [{"knob": kb["knob"], "side": kb["side"], "K_mm": round(kb["K_coarse"], 2)} for kb in F.get("knobs", []) if kb.get("obs")]
    out = {"screws": rows, "knobs": knobs, "constants": a.constants}
    if a.ruler:
        g = np.array([float(v) for v in a.ruler.split(",")]); fi = np.array([r_["gap_mm"] for r_ in rows], float)
        out["vs_ruler_mm"] = {"err": (fi - g).round(2).tolist(), "mae": float(np.nanmean(np.abs(fi - g))), "max": float(np.nanmax(np.abs(fi - g)))}
    json.dump(out, open(a.out, "w"), indent=1)
    print("calibrated gaps:", [(r_["name"], r_["gap_mm"]) for r_ in rows])
    print("knobs K:", [(kb["side"], kb["K_mm"]) for kb in knobs])
    if a.ruler:
        print(f"vs ruler: MAE {out['vs_ruler_mm']['mae']:.2f} max {out['vs_ruler_mm']['max']:.2f} mm")


if __name__ == "__main__":
    main()
