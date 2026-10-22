"""Test 2 table: fast / frame-box Stochastic Solids runs vs the full SS run of the same capture.

    python fastss_report.py OUT.png RUNS_DIR --ref-min 31.3 [--title T] [--order a,b,...]

RUNS_DIR/<run>/{times.json, compare.json} from headscan/steps/frame_box_ss.sh. Distances: tetra mesh of the run vs the
full SS tetra mesh (cube), median / p90 both ways (run -> full = accuracy-like, full -> run = completeness-like).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("runs"); ap.add_argument("--ref-min", type=float, required=True)
    ap.add_argument("--title", default=""); ap.add_argument("--order", default="")
    ap.add_argument("--cad", default=None, help="cad_check.json (labels = run names, full30k = reference)")
    a = ap.parse_args()
    R = Path(a.runs)
    names = a.order.split(",") if a.order else sorted(p.name for p in R.iterdir() if (p / "compare.json").exists())
    cad = json.load(open(a.cad)) if a.cad else None
    def arc(n):
        if not cad:
            return "-"
        m = cad.get("meshes", cad).get(n)
        return f"{m['abs_median_mm']:.2f}" if m else "-"
    rows = [["full SS (reference)", "30k", "1", "-", f"{a.ref_min:.1f}", "1.0x", "0", "0", arc("full30k")]]
    for n in names:
        if not (R / n / "compare.json").exists():
            continue
        t = json.load(open(R / n / "times.json")); c = json.load(open(R / n / "compare.json"))["tests"]["tetra"]
        tot = (t["dataset_s"] + t["train_s"] + t["extract_s"]) / 60
        tr, rt = c["test_to_ref"]["all"], c["ref_to_test"]["all"]
        rows.append([n, f"{t['iters'] // 1000}k", str(t["res"]), f"{t['peak_gpu_mib'] / 1024:.1f}",
                     f"{tot:.1f} ({t['train_s'] / 60:.1f} train)", f"{a.ref_min / tot:.1f}x",
                     f"{tr['median_mm']:.2f} / {tr['p90_mm']:.2f}", f"{rt['median_mm']:.2f} / {rt['p90_mm']:.2f}", arc(n)])
    cols = ["run", "iters", "-r", "GPU GB", "minutes (total)", "speed-up", "run -> full mm\nmedian / p90", "full -> run mm\nmedian / p90", "arc vs CAD\nmedian mm"]
    fig = plt.figure(figsize=(13, 1.2 + 0.42 * len(rows))); ax = fig.add_subplot(111); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.0)
    for (r, _), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8eef7"); cell.set_text_props(weight="bold")
    ax.set_title(a.title or "Test 2: Stochastic Solids in the frame box vs the full run (tetra meshes, cube)", fontsize=11)
    plt.tight_layout(); plt.savefig(a.out, dpi=120)


if __name__ == "__main__":
    main()
