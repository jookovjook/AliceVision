"""Metric scale + device-frame alignment from reconstructed AprilTag 36h11 corners.

    python tag_alignment.py SFM.sfm TAG_FEATURES_DIR frame_tags_layout.json OUT_PREFIX [--frame-tags 0-13]

1. Marker id of every tag36h11 landmark = one-hot index of its observations' descriptors (tagId*4 + corner);
   tracks whose observations disagree are reported and dropped; duplicate ids keep the longest track.
2. Scale check A (layout-free): edge lengths of every reconstructed tag (4 sides) vs the printed 16 mm.
3. Scale + pose B: similarity SfM -> device frame (mm) fitted on the RIGID frame tags (default 0-13), robust
   (iterative 3-sigma trimming), per-corner residuals in mm.
Writes OUT_PREFIX.json (scale, 4x4 SfM->device transform, residuals) and OUT_PREFIX.png.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_ids(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def desc_index(feat_dir, cache, vid):
    if vid not in cache:
        raw = (Path(feat_dir) / f"{vid}.tag36h11.desc").read_bytes()
        n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        cache[vid] = np.frombuffer(raw[8:], dtype=np.uint8).reshape(n, 256).argmax(1)
    return cache[vid]


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (X - mx).var(0).sum()
    return s, R, my - s * R @ mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sfm"); ap.add_argument("feat"); ap.add_argument("layout"); ap.add_argument("out")
    ap.add_argument("--frame-tags", default="0-13")
    a = ap.parse_args()
    d = json.load(open(a.sfm))
    L = json.load(open(a.layout))
    tag_mm = float(L["tag_size_mm"])
    view_of = {v["viewId"]: v for v in d["views"]}
    cache, marks, bad = {}, defaultdict(list), 0
    for l in d.get("structure", []):
        if l["descType"] != "tag36h11":
            continue
        ids = Counter(int(desc_index(a.feat, cache, int(o["observationId"]))[int(o["featureId"])]) for o in l["observations"])
        if len(ids) != 1:
            bad += 1
            continue
        marks[next(iter(ids))].append((len(l["observations"]), np.array([float(x) for x in l["X"]])))
    dup = sum(1 for v in marks.values() if len(v) > 1)
    M = {i: max(v, key=lambda t: t[0]) for i, v in marks.items()}  # index -> (nobs, X)
    print(f"tag landmarks: {sum(len(v) for v in marks.values())} ({len(M)} distinct corner ids, {dup} ids with several tracks, "
          f"{bad} tracks with mixed ids dropped)")

    # A: tag edge lengths (SfM units)
    edges = {}
    for t in sorted({i // 4 for i in M}):
        c = [M.get(t * 4 + k) for k in range(4)]
        if all(c):
            P = np.array([x[1] for x in c])
            edges[t] = [float(np.linalg.norm(P[k] - P[(k + 1) % 4])) for k in range(4)]
    allE = np.array([e for v in edges.values() for e in v])
    scaleA = tag_mm / np.median(allE)  # mm per SfM unit
    print(f"A) {len(edges)} complete tags, median edge {np.median(allE):.6f} SfM units -> {scaleA:.2f} mm per unit")

    # B: similarity to the frame layout on rigid frame tags
    ids = [t * 4 + k for t in parse_ids(a.frame_tags) for k in range(4) if t * 4 + k in M and str(t) in L["frame"]]
    X = np.array([M[i][1] for i in ids])
    Y = np.array([L["frame"][str(i // 4)][i % 4] for i in ids], float)
    keep = np.ones(len(ids), bool)
    for _ in range(5):
        s, R, t = umeyama(X[keep], Y[keep])
        res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
        new = res < max(3 * np.sqrt(np.mean(res[keep] ** 2)), 1.0)
        if (new == keep).all():
            break
        keep = new
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    per_tag = defaultdict(list)
    for i, r in zip(ids, res):
        per_tag[i // 4].append(float(r))
    print(f"B) frame-tag fit on {keep.sum()}/{len(ids)} corners of tags {sorted(per_tag)}: scale {s:.2f} mm per unit, "
          f"rms {np.sqrt(np.mean(res[keep] ** 2)):.2f} mm, median {np.median(res[keep]):.2f} mm, max inlier {res[keep].max():.2f} mm")
    print(f"   scale A vs B: {scaleA / s:.4f} (1 = tag size and layout agree)")
    out = {"scale_mm_per_unit_from_tag_edges": scaleA, "scale_mm_per_unit_from_layout": s,
           "sfm_to_device_mm": T.tolist(), "fit_corners": int(keep.sum()), "fit_corners_total": len(ids),
           "rms_mm": float(np.sqrt(np.mean(res[keep] ** 2))), "median_mm": float(np.median(res[keep])),
           "per_tag_residual_mm": {str(k): v for k, v in per_tag.items()},
           "outlier_corners": [int(i) for i, k in zip(ids, keep) if not k],
           "tag_edge_mm_from_layout_scale": {str(k): [e * s for e in v] for k, v in edges.items()},
           "tags_reconstructed": sorted(int(k) for k in {i // 4 for i in M})}
    json.dump(out, open(a.out + ".json", "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    tg = sorted(per_tag)
    ax[0].bar([str(k) for k in tg], [np.mean(per_tag[k]) for k in tg], color="tab:blue")
    ax[0].set_xlabel("frame tag id"); ax[0].set_ylabel("mean corner residual, mm")
    ax[0].set_title(f"frame tags vs CAD layout: rms {out['rms_mm']:.2f} mm ({keep.sum()}/{len(ids)} corners)")
    et = sorted(edges)
    ax[1].bar([str(k) for k in et], [np.mean(edges[k]) * s for k in et],
              color=["tab:blue" if k <= 13 else "tab:orange" if k <= 43 else "tab:green" for k in et])
    ax[1].axhline(tag_mm, color="k", lw=1); ax[1].set_ylim(tag_mm - 1.5, tag_mm + 1.5)
    ax[1].set_xlabel("tag id"); ax[1].set_ylabel("mean edge, mm (layout scale)")
    ax[1].set_title(f"tag size check: median {np.median(allE) * s:.2f} mm (printed {tag_mm:g})")
    ax[1].tick_params(axis="x", labelsize=6)
    Yp = (s * (R @ X.T)).T + t
    ax[2].scatter(Y[:, 1], Y[:, 2], s=14, label="CAD layout", color="k")
    ax[2].scatter(Yp[:, 1], Yp[:, 2], s=6, label="reconstructed, aligned", color="tab:red")
    ax[2].set_aspect("equal"); ax[2].set_xlabel("y forward, mm"); ax[2].set_ylabel("z up, mm"); ax[2].legend(fontsize=8)
    ax[2].set_title("frame tag corners, side view (device frame)")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=110)
    print("wrote", a.out + ".json", a.out + ".png")


if __name__ == "__main__":
    main()
