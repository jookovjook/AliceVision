"""Figure of calibrate.py's result: readout error per screw, every table capture (held out) and the head capture.

    python calib_report.py CALIBRATION.json OUT.png [--pitch caliper]

One panel per capture, 12 screws (sag1-8 = holes 0-7, cor1-4 = holes 8-11); bars = error vs the caliper (table) or the
ruler (head): CAD constants (coarse), calibrated coarse (per-screw bias) and calibrated fine (per-screw thread phase).
Table captures use constants from the OTHER table captures; the head uses constants from all three.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = [f"sag{i + 1}" for i in range(8)] + [f"cor{i + 1}" for i in range(4)]
SERIES = [("coarse", "CAD constants", "#a8a7a0"), ("coarse+b_k", "calibrated, tag position", "#2a78d6"),
          ("fine+phi_k", "calibrated, + wing angle", "#eb6834")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def panel(ax, title, items, err, truth_label):
    w = 0.26
    for j, (key, lab, col) in enumerate(SERIES):
        e = np.full(12, np.nan); e[items] = err[key]
        x = np.arange(12) + (j - 1) * (w + 0.02)
        ax.bar(x, e, w, color=col, label=lab, zorder=3)
    ax.axhspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xticks(range(12)); ax.set_xticklabels(NAMES, fontsize=9, color=INK2)
    ax.set_ylabel(f"error vs {truth_label}, mm", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    mae = {k: np.nanmean(np.abs(err[k])) for k, _, _ in SERIES}
    mx = {k: np.nanmax(np.abs(err[k])) for k, _, _ in SERIES}
    ax.set_title(f"{title}    MAE / max:  " + "   ".join(f"{lab} {mae[k]:.2f} / {mx[k]:.2f}" for k, lab, _ in SERIES),
                 fontsize=10, color=INK, loc="left")
    lim = max(2.0, min(6.0, np.nanmax([np.nanmax(np.abs(err[k])) for k, _, _ in SERIES]) * 1.15))
    ax.set_ylim(-lim, lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration"); ap.add_argument("out"); ap.add_argument("--pitch", default="caliper")
    a = ap.parse_args()
    R = json.load(open(a.calibration))
    loco = R["screws"][a.pitch]["loco"]
    rows = list(loco.items()) + ([("head", None)] if "head" in R else [])
    fig, axs = plt.subplots(len(rows), 1, figsize=(13, 2.7 * len(rows)), squeeze=False)
    for ax, (cap, v) in zip(axs[:, 0], rows):
        if v is not None:
            err = {k: np.array(v["err"][k]) for k, _, _ in SERIES}
            what = {"calibration_1": "screws fully in", "calibration_2": "tips at the inner face", "calibration_3": "mixed depths"}.get(cap, "")
            panel(ax, f"{cap} ({what}; held out)", np.array(v["items"]), err, "caliper")
        else:
            H = R["head"]; err = {k: np.array(H[a.pitch][k + "_err"]) for k, _, _ in SERIES}
            panel(ax, "head capture (constants from the 3 table captures)", np.arange(12), err, "ruler")
    axs[0, 0].legend(loc="upper right", fontsize=8.5, frameon=False, ncol=3)
    fig.suptitle(f"Screw readouts vs measured (thread pitch {R['screws'][a.pitch]['loco'][next(iter(loco))]['pitch']:.3f} mm; shaded = ±0.5 mm)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    plt.tight_layout(); plt.savefig(a.out, dpi=110, facecolor="white"); print("wrote", a.out)


if __name__ == "__main__":
    main()
