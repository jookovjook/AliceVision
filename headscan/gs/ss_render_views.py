"""Splat renders of a few training views (face front, both profiles) next to the input frames.

    python ss_render_views.py --repo REPO -m MODEL_DIR --out OUT.png [--center 0.02 0.04 0.04] [--views NAME ...]

Runs in the repo's env. Without --views it picks the training cameras whose centre direction from --center is closest
to +y (face), +x and -x (profiles) in the device frame. Columns: input frame, splat RGB (no appearance
compensation), rendered normals, median depth; each panel cropped to the head (square around --center's projection).
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--center", type=float, nargs=3, default=[0.02, 0.04, 0.04])
    ap.add_argument("--views", nargs="*", default=[])  # not None: get_combined_args drops None-valued args
    ap.add_argument("--crop-m", type=float, default=0.36, help="panel side at the head, metres")
    pre = ap.parse_known_args()[0]
    sys.path.insert(0, os.path.abspath(pre.repo)); os.chdir(os.path.abspath(pre.repo))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    model = ModelParams(ap, sentinel=True); pipeline = PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(ap)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    torch.set_grad_enabled(False)
    g = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    ctr = np.array(args.center)
    C = np.array([c.camera_center.cpu().numpy() for c in cams])
    u = (C - ctr) / np.linalg.norm(C - ctr, axis=1, keepdims=True)
    if args.views:
        pick = [(n, next(i for i, c in enumerate(cams) if c.image_name == n)) for n in args.views]
    else:
        pick = [(lab, int(np.argmax(u @ np.array(t, float)))) for lab, t in [("face", (0, 1, 0)), ("+x profile", (1, 0, 0)), ("-x profile", (-1, 0, 0))]]
    bg = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")
    fig, axs = plt.subplots(len(pick), 4, figsize=(16, 4.2 * len(pick)), squeeze=False)
    for r, (lab, i) in enumerate(pick):
        cam = cams[i]
        pkg = render(cam, g, pipe, bg, dataset.kernel_size)
        rgb = pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gt = cam.original_image.permute(1, 2, 0).cpu().numpy()
        nrm = pkg["normal"].permute(1, 2, 0).cpu().numpy()
        dep = pkg["median_depth"][0].cpu().numpy()
        # head crop: project the centre, square of crop_m metres at its depth
        Xc = ctr @ cam.R.cpu().numpy() + cam.T.cpu().numpy()
        px, py = cam.Fx * Xc[0] / Xc[2] + cam.Cx, cam.Fy * Xc[1] / Xc[2] + cam.Cy
        hs = int(0.5 * args.crop_m * cam.Fx / Xc[2])
        H, W = dep.shape
        y0, y1, x0, x1 = max(0, int(py) - hs), min(H, int(py) + hs), max(0, int(px) - hs), min(W, int(px) + hs)
        sl = (slice(y0, y1), slice(x0, x1))
        dd = dep[sl]; ok = (dd > 0) & (np.abs(dd - Xc[2]) < 0.2)
        lo_, hi_ = (np.percentile(dd[ok], [2, 98]) if ok.any() else (0, 1))
        panels = [(gt[sl], f"input {cam.image_name}"), (rgb[sl], "splat render"),
                  ((np.clip(-nrm[sl], -1, 1) + 1) / 2, "rendered normal (camera frame)"),
                  (np.where(ok, (dd - lo_) / (hi_ - lo_ + 1e-9), np.nan), f"median depth {lo_:.3f}-{hi_:.3f} m")]
        for c, (img, t) in enumerate(panels):
            axs[r, c].imshow(img, cmap="turbo" if img.ndim == 2 else None, vmin=0, vmax=1)
            axs[r, c].set_title(f"{lab}: {t}" if c == 0 else t, fontsize=10); axs[r, c].set_axis_off()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=80); print("wrote", args.out)
    print("views:", [(lab, cams[i].image_name) for lab, i in pick])


if __name__ == "__main__":
    main()
