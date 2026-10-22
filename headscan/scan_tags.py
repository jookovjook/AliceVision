"""Find the frame's AprilTags in a coloured TrueDepth surfel scan (StandardCyborg .ply) and put the scan in the device frame.

    python scan_tags.py SCAN.ply LAYOUT.json OUT_DIR [--px 0.25] [--views 24]

The scan's points are splatted (z-buffer, one square per surfel of its radius) into virtual pinhole views on a sphere
around the scan centre, at --px mm per pixel at the scan centre. AprilTag 36h11 (pupil-apriltags, corner order fixed
to the layout's TL, TR, BR, BL) runs on each render; each detected corner is lifted to 3-D from the render's depth,
and corners seen in several renders are averaged. The frame tags (0-13) then give a rigid fit layout -> scan (Kabsch),
whose inverse maps the scan into the device frame. Writes OUT_DIR/scan_tags.json, renders with detections, and
scan_device.ply when >= 3 frame tags are found.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            l = f.readline().decode().strip(); hdr.append(l)
            if l == "end_header":
                break
        n = int([h for h in hdr if h.startswith("element vertex")][0].split()[-1])
        props = [h.split()[-1] for h in hdr if h.startswith("property") and "list" not in h]
        if "ascii" in hdr[1]:
            a = np.loadtxt(f, max_rows=n)
        else:
            raise SystemExit("binary PLY not supported")
    col = {p: a[:, i] for i, p in enumerate(props[:a.shape[1]])}
    P = np.c_[col["x"], col["y"], col["z"]]; C = np.c_[col["red"], col["green"], col["blue"]].astype(np.uint8)
    r = col.get("surfel_radius", np.full(n, 5e-4))
    return P, C, r


def look_at(eye, target, up=np.array([0, 1.0, 0])):
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.stack([x, y, z])   # world -> camera rows (OpenCV: x right, y down, z forward)


def render(P, C, r, R, eye, f, W, H):
    Xc = (P - eye) @ R.T
    m = Xc[:, 2] > 0.02
    Xc, Cm, rm = Xc[m], C[m], r[m]
    u = f * Xc[:, 0] / Xc[:, 2] + W / 2; v = f * Xc[:, 1] / Xc[:, 2] + H / 2
    rad = np.maximum(1, np.ceil(f * rm / Xc[:, 2])).astype(int)
    img = np.zeros((H, W, 3), np.uint8); zb = np.full((H, W), np.inf)
    order = np.argsort(-Xc[:, 2])   # far to near: near ones overwrite (3x3 splat for surfels >= 2 px)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            uu, vv = ui[order] + dx, vi[order] + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & ((rad[order] >= 2) | ((dx == 0) & (dy == 0)))
            idx = order[ok]; uu, vv = uu[ok], vv[ok]
            z = Xc[idx, 2]
            closer = z <= zb[vv, uu]
            img[vv[closer], uu[closer]] = Cm[idx[closer]][:, ::-1]
            zb[vv[closer], uu[closer]] = np.minimum(zb[vv[closer], uu[closer]], z[closer])
    hole = ~np.isfinite(zb)   # pin-holes between surfels: take the 3x3 median of the neighbours
    img = np.where(hole[..., None] & (cv2.dilate((~hole).astype(np.uint8), np.ones((3, 3))) > 0)[..., None], cv2.medianBlur(img, 3), img)
    return img, zb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan"); ap.add_argument("layout"); ap.add_argument("out")
    ap.add_argument("--px", type=float, default=0.25, help="mm per pixel at the scan centre")
    ap.add_argument("--views", type=int, default=24); ap.add_argument("--dist", type=float, default=0.35)
    a = ap.parse_args()
    from pupil_apriltags import Detector
    det = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=1, decode_sharpening=0.25)
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    P, C, r = read_ply(a.scan)
    cen = np.median(P, 0); ext = np.percentile(np.linalg.norm(P - cen, axis=1), 98)
    f = a.dist * 1000 / a.px; W = H = int(2.4 * ext * f / a.dist)
    print(f"{len(P)} surfels, median radius {np.median(r) * 1000:.2f} mm, extent {ext * 1000:.0f} mm; renders {W}x{H}, f {f:.0f} px")
    # scan camera looks along +z (TrueDepth frame): views on a cap around -z of the centre (the side the sensor saw)
    rng = np.random.default_rng(0); hits = {}
    dirs = [np.array([0, 0, -1.0])] + [d / np.linalg.norm(d) for d in rng.normal(size=(a.views - 1, 3)) * [0.6, 0.6, 0.3] + [0, 0, -1]]
    sheets = []
    for k, d in enumerate(dirs):
        eye = cen + a.dist * d; R = look_at(eye, cen)
        img, zb = render(P, C, r, R, eye, f, W, H)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ds = [x for x in det.detect(g) if x.hamming <= 1]
        for x in ds:
            cs = np.asarray(x.corners)[[1, 0, 3, 2]]
            for j, (u, v) in enumerate(cs):
                ui, vi = int(round(u)), int(round(v))
                win = zb[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4]
                if not np.isfinite(win).any():
                    continue
                z = np.min(win[np.isfinite(win)])
                Xw = R.T @ (np.array([(u - W / 2) / f, (v - H / 2) / f, 1.0]) * z) + eye
                hits.setdefault(x.tag_id * 4 + j, []).append(Xw)
            cv2.polylines(img, [np.asarray(x.corners).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(img, str(x.tag_id), tuple(np.asarray(x.center).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if k < 6:
            sheets.append(cv2.resize(img, (600, 600)))
        print(f"view {k}: {len(ds)} tags {sorted(x.tag_id for x in ds)}")
    cv2.imwrite(str(O / "renders.jpg"), np.hstack(sheets[:3]) if len(sheets) < 6 else np.vstack([np.hstack(sheets[:3]), np.hstack(sheets[3:6])]))
    L = json.load(open(a.layout))
    lay = {int(t) * 4 + j: np.array(c) for t, cs in L["frame"].items() for j, c in enumerate(cs)}
    corners = {m: np.median(v, 0) for m, v in hits.items()}
    tags = sorted({m // 4 for m in corners})
    res = {"surfels": len(P), "median_radius_mm": float(np.median(r) * 1000), "tags_found": tags, "corner_hits": {str(m): len(v) for m, v in hits.items()}}
    ms = [m for m in corners if m in lay]
    if len({m // 4 for m in ms}) >= 3:
        A = np.array([lay[m] for m in ms]) / 1000.0; B = np.array([corners[m] for m in ms])
        ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
        D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rl = U @ D @ Vt; tl = mb - Rl @ ma   # layout -> scan
        e = np.linalg.norm(A @ Rl.T + tl - B, axis=1) * 1000
        res.update({"frame_tags_used": sorted({m // 4 for m in ms}), "fit_rms_mm": float(np.sqrt((e ** 2).mean())), "fit_max_mm": float(e.max()),
                    "scan_to_device": np.r_[np.c_[Rl.T, -Rl.T @ tl], [[0, 0, 0, 1]]].tolist()})
        Pd = (P - tl) @ Rl
        with open(O / "scan_device.ply", "w") as fp:
            fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(Pd)}\nproperty float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(Pd, C):
                fp.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
        print(f"layout fit on {len(ms)} corners of tags {res['frame_tags_used']}: rms {res['fit_rms_mm']:.2f} mm, max {res['fit_max_mm']:.2f} mm")
    json.dump(res, open(O / "scan_tags.json", "w"), indent=1)
    print("tags found:", tags)


if __name__ == "__main__":
    main()
