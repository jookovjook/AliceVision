"""Test 1 figure: tag-only poses vs SfM poses per capture (pose differences, time, image-fit reprojection, screw readouts).

    python tagpose_report.py OUT.png CAP_DIR [CAP_DIR ...] [--variant ark05] [--sfm-min 79,..]

CAP_DIR holds cmp.json (tag_poses_compare.py), tp_<variant>.json (tag_poses.py) and if_ref/out, if_<variant>/out
(image_fit.py with --ruler, so coarse_err is the error against the caliper/ruler gaps).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("caps", nargs="+"); ap.add_argument("--variant", default="ark05")
    ap.add_argument("--sfm-min", default="", help="SfM path wall-clock per capture, minutes (from the box logs)")
    ap.add_argument("--tags-s", default="", help="tag detection seconds per capture")
    a = ap.parse_args()
    sfm_min = [float(v) for v in a.sfm_min.split(",")] if a.sfm_min else [None] * len(a.caps)
    tags_s = [float(v) for v in a.tags_s.split(",")] if a.tags_s else [None] * len(a.caps)
    rows, curves = [], []
    for c, tm, ts in zip(a.caps, sfm_min, tags_s):
        c = Path(c); v = a.variant
        cmp = json.load(open(c / "cmp.json")); tv = cmp["variants"][v]; tp = json.load(open(c / f"tp_{v}.json"))
        fr = json.load(open(c / "if_ref/out/image_fit.json")); ft = json.load(open(c / f"if_{v}/out/image_fit.json"))
        mae = lambda f: float(np.mean(np.abs(f["coarse_err"]))); sdv = lambda f: float(np.std(f["coarse_err"]))
        rows.append([c.name,
                     f"{tm:.0f} min" if tm else "-", f"{(ts or 0) + tp['seconds']:.0f} s",
                     f"{tv['centre_mm']['median']:.1f} / {tv['rotation_deg']['median']:.2f}",
                     f"{tv['centre_after_sim_mm']['median']:.1f} / {tv['rotation_after_sim_deg']['median']:.2f}",
                     f"{cmp['ref_sift_reproj_px']['median']:.2f} | {tv['sift_reproj_px']['median']:.2f}",
                     f"{fr['reproj_px']['frame']:.2f} | {ft['reproj_px']['frame']:.2f}",
                     f"{fr['reproj_px']['screws']:.2f} | {ft['reproj_px']['screws']:.2f}",
                     f"{mae(fr):.2f} | {mae(ft):.2f}", f"{sdv(fr):.2f} | {sdv(ft):.2f}",
                     " ".join(f"{k['K_coarse']:.1f}" for k in fr["knobs"] if k.get("obs")) + " | " +
                     " ".join(f"{k['K_coarse']:.1f}" for k in ft["knobs"] if k.get("obs"))])
        curves.append((c.name, json.load(open(c / f"tp_{v}.json"))["per_view"]))
    cols = ["capture", "SfM path", "tag path", "pose diff\nmm / deg", "after sim.\nmm / deg", "SIFT reproj px\nSfM | tag",
            "frame-tag px\nSfM | tag", "screw-tag px\nSfM | tag", "screw gap MAE mm\nSfM | tag", "gap err SD mm\nSfM | tag", "knobs K mm\nSfM | tag"]
    fig = plt.figure(figsize=(16, 1.4 + 0.45 * len(rows)))
    ax = fig.add_subplot(111); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.2)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8eef7"); cell.set_text_props(weight="bold")
    ax.set_title(f"Test 1: tag-only poses ({a.variant}: PnP + BA on frame-tag corners, segments, ARKit relative prior) vs AliceVision SfM poses", fontsize=11)
    plt.tight_layout(); plt.savefig(a.out, dpi=120)


if __name__ == "__main__":
    main()
