"""Calibration of the image-space readouts (image_fit.py) against caliper truth, leave-one-capture-out.

    python calibrate.py TRUTH.json OUT_DIR --fit CAPTURE=image_fit.json [...] [--head image_fit.json]

TRUTH.json: headscan/data/calibration_truth_*.json (D per screw, K per knob, screw length, ruler gaps of the head).
Screw k in capture c: coarse gap g_c = s + 74 (CAD) and wing angle psi; truth g = screw_length - D.
Predictors (their constants are fitted on the training captures only):
  coarse        g_c
  coarse+b      g_c + b                     one bias for every screw
  coarse+b_k    g_c + b_k                   per-screw bias (sticker glue position, as-printed wing / hole)
  fine+phi      74 + p psi/360 + phi + n p     one phase offset, turn n from coarse+b
  fine+phi_k    74 + p psi/360 + phi_k + n p   per-screw phase offset (sticker glue angle, thread start), n from coarse+b_k
p = thread advance per turn: 2.5 (CAD), 2.508 (caliper, 10 crests) or fitted (phase coherence on the training captures).
Knob j (0 right, 1 left): K_c = u - 147 and the knob angle psi; right-hand thread: K = phi_j - p psi/360 (mod p).
Writes OUT_DIR/calibration.json and prints the tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PITCHES = {"cad": 2.5, "caliper": 2.508}


def wrap(x, p):
    return (np.asarray(x) + p / 2) % p - p / 2


def circ_mean(r, p):
    a = 2 * np.pi * np.asarray(r) / p
    return float(p * np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi))


def screw_rows(fit, cap, truth):
    rows = []
    for r in fit["readouts"]:
        if not r.get("obs"):
            continue
        g = None if truth is None else truth[r["hole"]]
        rows.append({"cap": cap, "k": r["hole"], "gc": r["coarse"], "psi": r["psi_deg"], "g": g, "sd": r["sd_coarse_mm"],
                     "views": r.get("views"), "tilt": r.get("tilt_deg")})
    return rows


def knob_rows(fit, cap, truth):
    rows = []
    for kb in fit.get("knobs", []):
        if not kb.get("obs"):
            continue
        rows.append({"cap": cap, "k": kb["knob"], "gc": kb["K_coarse"], "psi": kb["psi_deg"],
                     "g": None if truth is None else truth[kb["knob"]], "sd": kb["sd_u_mm"], "views": kb.get("views")})
    return rows


class Screws:
    """Constants fitted on training rows; fine phase in gap space: gap = 74 + p psi/360 + phi (mod p)."""
    sign, base = 1.0, 74.0

    def __init__(self, train, p):
        self.p = p
        e = np.array([r["g"] - r["gc"] for r in train])
        self.b = float(np.median(e))
        self.bk = {k: float(np.mean([r["g"] - r["gc"] for r in train if r["k"] == k])) for k in {r["k"] for r in train}}
        ph = lambda r: r["g"] - self.base - self.sign * p * r["psi"] / 360
        self.phi = circ_mean([ph(r) for r in train], p)
        self.phik = {k: circ_mean([ph(r) for r in train if r["k"] == k], p) for k in self.bk}

    def predict(self, r):
        p = self.p
        c_b = r["gc"] + self.b
        c_bk = r["gc"] + self.bk.get(r["k"], self.b)
        f = self.base + self.sign * p * r["psi"] / 360

        def fine(phi, coarse):
            x = f + phi
            return x + np.round((coarse - x) / p) * p
        return {"coarse": r["gc"], "coarse+b": c_b, "coarse+b_k": c_bk,
                "fine+phi": fine(self.phi, c_b), "fine+phi_k": fine(self.phik.get(r["k"], self.phi), c_bk)}


class Knobs(Screws):
    sign, base = -1.0, 0.0


def fit_pitch(train, model, lo=2.47, hi=2.53, step=0.0005):
    """Pitch that makes the per-item phase offsets most coherent across the training captures."""
    best = None
    for p in np.arange(lo, hi + step / 2, step):
        m = model(train, p)
        res = [wrap(r["g"] - m.base - m.sign * p * r["psi"] / 360 - m.phik[r["k"]], p) for r in train]
        score = float(np.mean(np.abs(res)))
        if best is None or score < best[1]:
            best = (float(p), score)
    return best[0]


def evaluate(rows, model, pitch):
    """Leave one capture out: constants from the other captures, errors on the held-out one."""
    caps = sorted({r["cap"] for r in rows})
    out, errs = {}, {}
    for c in caps:
        tr = [r for r in rows if r["cap"] != c]; te = [r for r in rows if r["cap"] == c]
        p = fit_pitch(tr, model) if pitch == "fit" else PITCHES[pitch]
        m = model(tr, p)
        pr = [m.predict(r) for r in te]
        out[c] = {"pitch": p, "items": [r["k"] for r in te], "truth": [r["g"] for r in te],
                  "pred": {n: [float(x[n]) for x in pr] for n in pr[0]}}
        for n in pr[0]:
            errs.setdefault(n, []).extend([x[n] - r["g"] for x, r in zip(pr, te)])
        out[c]["err"] = {n: (np.array(out[c]["pred"][n]) - np.array(out[c]["truth"])).round(2).tolist() for n in pr[0]}
    summ = {n: {"mae": float(np.mean(np.abs(e))), "max": float(np.max(np.abs(e))), "rms": float(np.sqrt(np.mean(np.square(e))))}
            for n, e in errs.items()}
    return out, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth"); ap.add_argument("out")
    ap.add_argument("--fit", nargs="+", required=True, help="CAPTURE=image_fit.json (table captures with truth)")
    ap.add_argument("--head", default="", help="image_fit.json of the head capture (ruler gaps from TRUTH.json)")
    a = ap.parse_args()
    T = json.load(open(a.truth)); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    Ls = T["screw_length_mm"]
    S, K, meta = [], [], {}
    for spec in a.fit:
        cap, path = spec.split("=", 1); F = json.load(open(path)); t = T["captures"][cap]
        S += screw_rows(F, cap, [Ls - d for d in t["D_mm"]]); K += knob_rows(F, cap, t["K_mm"])
        meta[cap] = {"scale": F["scale"], "inset_mm": F["inset_mm"], "reproj_px": F["reproj_px"], "n_views": F.get("n_views"),
                     "screws_seen": sum(1 for r in F["readouts"] if r.get("obs")), "knobs_seen": sum(1 for kb in F.get("knobs", []) if kb.get("obs"))}
    print("captures:", json.dumps(meta, indent=None))
    res = {"meta": meta, "screws": {}, "knobs": {}}
    print(f"\nSCREWS, leave one capture out ({len(S)} readouts)")
    print(f"{'pitch':8} " + " ".join(f"{n:>17}" for n in ["coarse", "coarse+b", "coarse+b_k", "fine+phi", "fine+phi_k"]) + "   (MAE / max, mm)")
    for pitch in ("cad", "caliper", "fit"):
        loco, summ = evaluate(S, Screws, pitch)
        res["screws"][pitch] = {"loco": loco, "summary": summ}
        print(f"{pitch:8} " + " ".join(f"{summ[n]['mae']:8.2f} /{summ[n]['max']:6.2f}" for n in summ))
    if K:
        print(f"\nKNOBS, leave one capture out ({len(K)} readouts)")
        for pitch in ("cad", "caliper"):
            loco, summ = evaluate(K, Knobs, pitch)
            res["knobs"][pitch] = {"loco": loco, "summary": summ}
            print(f"{pitch:8} " + " ".join(f"{n}: {summ[n]['mae']:.2f}/{summ[n]['max']:.2f}" for n in summ))
    # per-capture detail with the caliper pitch
    for c, v in res["screws"]["caliper"]["loco"].items():
        print(f"\n{c} (constants from the other captures, p = {v['pitch']}):  truth {np.round(v['truth'], 2).tolist()}")
        for n, e in v["err"].items():
            print(f"  {n:11} err {e}")
    if K:
        for c, v in res["knobs"]["caliper"]["loco"].items():
            print(f"{c} knobs truth {v['truth']}: " + "  ".join(f"{n} {e}" for n, e in v["err"].items()))
    # constants from all table captures, applied to the head capture
    allp = {"cad": 2.5, "caliper": 2.508, "fit": fit_pitch(S, Screws)}
    res["pitch_fit_all"] = allp["fit"]
    final = {pn: Screws(S, p) for pn, p in allp.items()}
    res["constants"] = {pn: {"pitch": m.p, "b": m.b, "b_k": m.bk, "phi": m.phi, "phi_k": m.phik} for pn, m in final.items()}
    if K:
        mk = Knobs(K, 2.508)
        res["knob_constants"] = {"pitch": mk.p, "b": mk.b, "b_k": mk.bk, "phi": mk.phi, "phi_k": mk.phik}
    print(f"\npitch fitted on all table captures: {allp['fit']:.4f} mm;  per-screw bias b_k (caliper pitch): "
          f"{ {k: round(v, 2) for k, v in sorted(final['caliper'].bk.items())} }")
    if a.head:
        H = json.load(open(a.head)); ruler = T["head_capture_6ba4b7d84f"]["gap_mm"]
        hr = screw_rows(H, "head", ruler)
        res["head"] = {"ruler": ruler}
        print(f"\nHEAD capture vs ruler {ruler}")
        for pn, m in final.items():
            pr = [m.predict(r) for r in hr]
            res["head"][pn] = {n: [float(x[n]) for x in pr] for n in pr[0]}
            for n in pr[0]:
                e = np.array([x[n] for x in pr]) - np.array(ruler)
                res["head"][pn][n + "_err"] = e.round(2).tolist()
                if pn == "caliper" or n.startswith("fine"):
                    print(f"  p={pn:8} {n:11} MAE {np.mean(np.abs(e)):.2f} max {np.max(np.abs(e)):.2f}  err {e.round(2).tolist()}")
        if K:
            hk = knob_rows(H, "head", None)
            res["head"]["knobs"] = [{"knob": r["k"], **{n: float(v) for n, v in mk.predict(r).items()}} for r in hk]
            print("  knobs (no truth):", [(r["knob"], round(r["coarse+b_k"], 2), round(r["fine+phi_k"], 2)) for r in res["head"]["knobs"]])
    json.dump(res, open(O / "calibration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
