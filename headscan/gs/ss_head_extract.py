"""Head-region outputs of a trained Stochastic Solids model (HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids).

    python ss_head_extract.py --repo REPO -m MODEL_DIR --out OUT_DIR [--center 0.02 0.04 0.04 --half 0.22]
                              [--what centres tsdf tetra] [--voxel 0.001 --trunc 0.006] [--iteration -1]

Runs in the repo's env (needs its CUDA extensions); everything stays in the training (COLMAP = device) frame, metres.
  centres  Gaussian centres inside the cube -> OUT/gaussian_centres.ply (RGB from SH DC, opacity, max scale)
  tsdf     the repo's DTU extractor (mesh_extract.py: median depth of the training views -> Open3D TSDF), with our
           voxel/truncation instead of its hard-coded 2 mm, depth pixels outside the cube zeroed -> OUT/tsdf.ply
  tetra    the repo's TnT extractor (mesh_extract_tetrahedra.py: marching tetrahedra on the Gaussian tetra points,
           10 binary-search steps, vertex colours) with the tetra points cut to the cube -> OUT/tetra.ply
Both meshes keep connected components with >= 2 % of the largest one's triangles (as headscan/fuse_tsdf.py), so the
arc survives when it is not connected to the head; the repo keeps only the largest component.
"""
import argparse
import os
import sys
import time

import numpy as np


def keep_components(mesh, frac=0.02):
    import open3d as o3d
    cl, n, _ = mesh.cluster_connected_triangles()
    cl, n = np.asarray(cl), np.asarray(n)
    mesh.remove_triangles_by_mask(n[cl] < max(frac * n.max(), 50))
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def crop_mesh(mesh, lo, hi):
    import open3d as o3d
    v = np.asarray(mesh.vertices)
    inside = np.all((v >= lo) & (v <= hi), axis=1)
    f = np.asarray(mesh.triangles)
    mesh.remove_triangles_by_mask(~inside[f].all(1))
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", type=float, nargs=3, default=[0.02, 0.04, 0.04])
    ap.add_argument("--half", type=float, default=0.22)
    ap.add_argument("--what", nargs="+", default=["centres", "tsdf", "tetra"])
    ap.add_argument("--voxel", type=float, default=0.001)
    ap.add_argument("--trunc", type=float, default=0.006)
    ap.add_argument("--min_opacity", type=float, default=0.0, help="centres: drop Gaussians below this opacity")
    sys.path.insert(0, os.path.abspath(ap.parse_known_args()[0].repo))
    os.chdir(os.path.abspath(ap.parse_known_args()[0].repo))

    import open3d as o3d
    import open3d.core as o3c
    import torch
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    model = ModelParams(ap, sentinel=True)
    pipeline = PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--move_cpu", action="store_true")
    args = get_combined_args(ap)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    os.makedirs(args.out, exist_ok=True)
    ctr = np.array(args.center); lo, hi = ctr - args.half, ctr + args.half

    torch.set_grad_enabled(False)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    print(f"loaded iteration {scene.loaded_iter}: {gaussians.get_xyz.shape[0]} Gaussians, {len(cams)} views", flush=True)

    if "centres" in args.what:
        xyz = gaussians.get_xyz.cpu().numpy()
        op = gaussians.get_opacity_with_3D_filter.squeeze(-1).cpu().numpy()
        sc = gaussians.get_scaling_with_3D_filter.max(-1).values.cpu().numpy()
        rgb = np.clip(0.28209479177387814 * gaussians._features_dc[:, 0].cpu().numpy() + 0.5, 0, 1)
        m = np.all((xyz >= lo) & (xyz <= hi), 1) & (op >= args.min_opacity)
        from plyfile import PlyData, PlyElement
        el = np.empty(m.sum(), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                      ("opacity", "f4"), ("scale", "f4")])
        el["x"], el["y"], el["z"] = xyz[m].T
        el["red"], el["green"], el["blue"] = (rgb[m] * 255 + 0.5).astype(np.uint8).T
        el["opacity"], el["scale"] = op[m], sc[m]
        PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(args.out, "gaussian_centres.ply"))
        print(f"centres: {m.sum()} of {len(xyz)} Gaussians in the cube; opacity median {np.median(op[m]):.2f}, "
              f"scale median {np.median(sc[m]) * 1e3:.2f} mm", flush=True)

    if "tsdf" in args.what:
        t0 = time.time()
        bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")  # as mesh_extract.py
        vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                            attr_channels=((1), (1), (3)), voxel_size=args.voxel,
                                            block_resolution=16, block_count=400000, device=o3d.core.Device("CPU:0"))
        lo_t, hi_t = torch.tensor(lo - args.trunc, device="cuda"), torch.tensor(hi + args.trunc, device="cuda")
        for cam in cams:
            pkg = render(cam, gaussians, pipe, bg, dataset.kernel_size)
            rgb = torch.clamp(pkg["render"], 0, 1).permute(1, 2, 0).contiguous()
            d = pkg["median_depth"][0].clone()
            H, W = d.shape
            x = (torch.arange(W, device="cuda") - cam.Cx) / cam.Fx
            y = (torch.arange(H, device="cuda") - cam.Cy) / cam.Fy
            Xc = torch.stack([d * x[None], d * y[:, None], d], -1)
            Xw = (Xc - cam.T) @ cam.R.T  # camera -> world (cam.R = R_w2c^T)
            d[~(((Xw >= lo_t) & (Xw <= hi_t)).all(-1))] = 0
            K = o3c.Tensor(np.array([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], np.float64))
            E = o3c.Tensor(cam.world_view_transform.T.cpu().numpy().astype(np.float64))
            di = o3d.t.geometry.Image(d.cpu().numpy())
            ci = o3d.t.geometry.Image(rgb.cpu().numpy())
            blocks = vbg.compute_unique_block_coordinates(di, K, E, 1.0, 8.0, trunc_voxel_multiplier=args.trunc / args.voxel)
            vbg.integrate(blocks, di, ci, K, E, 1.0, 8.0, trunc_voxel_multiplier=args.trunc / args.voxel)
        mesh = vbg.extract_triangle_mesh().to_legacy()
        mesh = keep_components(crop_mesh(mesh, lo, hi))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(args.out, "tsdf.ply"), mesh)
        print(f"tsdf: {len(mesh.triangles)} triangles, voxel {args.voxel * 1e3:.2f} mm, trunc {args.trunc * 1e3:.1f} mm, "
              f"{time.time() - t0:.0f} s", flush=True)

    if "tetra" in args.what:
        t0 = time.time()
        import mesh_extract_tetrahedra as met
        orig = gaussians.get_tetra_points
        lo_g, hi_g = torch.tensor(lo, device="cuda", dtype=torch.float32), torch.tensor(hi, device="cuda", dtype=torch.float32)

        def tetra_points_in_cube():
            pts, scale = orig()
            m = ((pts >= lo_g) & (pts <= hi_g)).all(-1)
            print(f"tetra points in the cube: {int(m.sum())} of {len(pts)}", flush=True)
            return pts[m].contiguous(), scale[m].contiguous()

        gaussians.get_tetra_points = tetra_points_in_cube
        met.post_process_mesh = lambda mesh, n: keep_components(mesh)
        tdir = os.path.join(args.out, "tetra"); os.makedirs(tdir, exist_ok=True)
        bg = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")
        met.marching_tetrahedra_with_binary_search(tdir, cams, gaussians, pipe, dataset.kernel_size, bg,
                                                   export_color=True, move_cpu=args.move_cpu, save_cells=False, num_cluster=1)
        mesh = o3d.io.read_triangle_mesh(os.path.join(tdir, "recon_post.ply"))
        mesh = crop_mesh(mesh, lo, hi)
        o3d.io.write_triangle_mesh(os.path.join(args.out, "tetra.ply"), mesh)
        print(f"tetra: {len(mesh.triangles)} triangles, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
