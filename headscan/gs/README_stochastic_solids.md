# Stochastic Solids on 6ba4b7d84f (head + AprilTag arc), RGB only, frozen metric poses

Geometry-Grounded Gaussian Splatting / "Stochastic Solids" (arXiv 2601.17835, SIGGRAPH Asia 2026),
https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids at commit
`23b418c2f0f19b30b251acad6fe13445dae968c7` (2026-09-24). Inria-style licence: research and evaluation only.
Box: vast-gs, RTX 4090 24 GB, CUDA 12.4 devel image. Driver script and helpers: `headscan/gs/` (branch
`claude/gs-stochastic-solids-c7e0f2`), run as `bash run_stochastic_solids.sh setup|train|extract|compare|figures|upload`.

## Data and frame
`/workspace/work/gs/dataset` (= gdrive `.../gs/dataset`, all 750 files md5-identical): 248 keyframes undistorted onto a
centred 1816x1360 PINHOLE (f = 1458.69 px, cx = 908, cy = 680), COLMAP text model, world->camera poses in the device
frame (metres; origin = ear-axis midpoint, x right, y forward, z up), 110k landmarks for the initialisation.
The repo reads this as-is: no COLMAP run, poses never optimised, no re-centring or rescaling (Scene uses the camera
spread only for the densification extent, 0.744). So every output is in the device frame, and the numbers below
confirm it (0.9 mm to the reference). Cameras sit 0.60-0.75 m from the head, above the rasterizer's hard-coded 0.2
near plane, so the scene was not rescaled. The trainer reads `/workspace/work/gs/stochastic/data`, a directory of
symlinks, so the loader's one-off `points3D.ply` lands there and not in the shared export.

## Commands
```
# env: conda py3.12 "gggss", torch 2.6.0+cu124, extensions built for sm_89 (TORCH_CUDA_ARCH_LIST=8.9), CGAL from conda-forge
python train.py -s /workspace/work/gs/stochastic/data -m /workspace/work/gs/stochastic/model \
    -r 1 --use_decoupled_appearance 3 --data_device cpu          # repo's DTU/TnT recipe, 30k iterations, black bg
python headscan/gs/ss_head_extract.py --repo REPO -m model --out head --center 0.02 0.04 0.04 --half 0.22 \
    --what centres tsdf tetra --voxel 0.001 --trunc 0.006
python headscan/gs/ss_render_views.py --repo REPO -m model --out figures/splat_views.png
python headscan/gs/surface_compare.py REF compare_raw.json tsdf=... tetra=... centres=...
python headscan/gs/surface_compare.py REF compare.json tsdf=... tetra=... centres=... --visible dataset/sparse/0 \
    --clean-dir head/visible --heat-dir head/heat
python headscan/gs/cad_check.py cad/cad_frame_device_m.ply cad_check.json ref=... ss_tsdf=... ss_tetra=...
python headscan/viz/render.py compare /workspace/work/ref/sfm_device.sfm OUT.png "LABEL=MESH.ply:noflip" ... --crop 0.4 [--textured]
```
- `-r 1`: full 1816x1360. Frames stay in host RAM (`--data_device cpu`): on the GPU they would take 9.8 GB as float32 RGB+grey.
- Default 30k iterations. Densification runs to 15k, and the multi-view NCC/geometry and depth-normal losses start at 7k.
  Background is black, the repo's default for DTU/TnT. The frames are opaque, so it only affects empty pixels.
- TSDF = the repo's DTU extractor (`mesh_extract.py`: median depth of all training views, Open3D TSDF), but with a 1 mm
  voxel and 6 mm truncation instead of its hard-coded 2 mm. Depth pixels outside the head cube are zeroed.
- Tetra = the repo's TnT extractor (`mesh_extract_tetrahedra.py`: Delaunay of the Gaussian tetra points, marching
  tetrahedra on the transmittance field, 10 binary-search steps, vertex colours), with tetra points cut to the cube.
- Both meshes keep components with >= 2 % of the largest one's triangles (as fuse_tsdf.py does) instead of the repo's
  largest-only rule, so the arc and pins survive.

## Time and memory
| step | wall-clock | peak GPU (nvidia-smi) |
|---|---|---|
| train, 30k iterations at 1816x1360 (incl. ~2 min frame loading) | 29.3 min (1755 s) | 11.9 GB |
| extract: centres + TSDF (37 s) + tetra (48 s) | 113 s | 3.4 GB |

7k iterations took 2 min; once the multi-view losses start at 7k it runs at ~13-15 it/s. The run ends with 750k
Gaussians; 103k are inside the head cube (36k with opacity >= 0.5), median scale 2.2 mm. Train-view PSNR is 29.0 dB (L1 0.018).

## Reference mesh caveat (affects every earlier render of tsdf_mvs.ply)
`/workspace/work/ref/tsdf_mvs.ply` (headscan/fuse_tsdf.py) is double-layered. Every ray out of the head centre hits it at
least twice, with the outer two hits a median 6.9 mm apart. The grid starts at +1, voxels more than `trunc` behind the
surface are never integrated, and marching cubes is not masked, so a second surface appears ~trunc inside: 48 % of the
triangles are never seen by any camera. The true outer layer is also wound inside-out (`faces[:, ::-1]`): 99 % of its
camera-visible triangles face away from the cameras that see them. render.py (pyrender) culls back faces, so the raw
mesh renders as its inner layer, about 6 mm shrunk and blurred (`figures/compare_geometry_raw.png`, top row).
Fix in fuse_tsdf.py: `marching_cubes(..., mask=W >= min_weight)` or -1 behind observed surfaces, then check the winding.
For a fair comparison, `surface_compare.py --visible` keeps in every mesh only the triangles that some training camera
sees unoccluded and inside its image, and turns each one to face the cameras. That removes 48 % of the reference
(its inner layer) and 1-2 % of the splat meshes. The figures use these camera-visible meshes (`head/visible/`).

## Numbers (head cube: half-size 0.22 m around (0.02, 0.04, 0.04), device frame)
Exact point-to-triangle distances, 1M area-uniform samples per mesh, camera-visible layers (`compare.json`).
Format: median / p90 in mm, then the share of samples within 1 mm.

| vs reference TSDF of AliceVision MVS | all | face | ears | crown |
|---|---|---|---|---|
| SS TSDF -> ref | 0.92 / 3.39 (53 %) | 0.78 / 2.68 | 0.80 / 2.15 | 0.74 / 3.02 |
| ref -> SS TSDF | 0.98 / 4.59 (51 %) | 0.78 / 3.56 | 0.86 / 2.80 | 0.65 / 2.37 |
| SS tetra -> ref | 0.98 / 3.80 (51 %) | 0.79 / 2.78 | 0.82 / 2.34 | 0.86 / 4.40 |
| ref -> SS tetra | 1.01 / 4.61 (50 %) | 0.80 / 3.92 | 0.89 / 3.28 | 0.74 / 2.77 |
| Gaussian centres (opacity >= 0.5) -> ref | 1.30 / 4.54 (41 %) | 1.25 / 3.56 | 1.01 / 2.87 | 1.39 / 4.75 |

Raw meshes (`compare_raw.json`): SS TSDF -> ref 0.88 / 3.19, but ref -> SS TSDF 3.21 / 9.74, inflated by the
reference's inner layer.

Rigid frame vs the CAD frame placed in the device frame from the tag layout (step 6/7; `cad_check.json`). The table
uses mesh samples within 8 mm of the CAD. "CAD covered" = share of the CAD surface within 2 mm of the mesh; this
includes faces no camera sees.

| mesh | abs median | abs p90 | signed median (+ = outside CAD) | CAD covered |
|---|---|---|---|---|
| ref TSDF (visible layer) | 1.49 mm | 5.54 mm | +0.54 mm | 53 % |
| SS TSDF | 1.31 mm | 5.07 mm | +0.86 mm | 55 % |
| SS tetra | 1.30 mm | 4.96 mm | +0.52 mm | 57 % |

## Figures (`figures/`)
- `compare_geometry.png` and `compare_textured.png`: reference vs SS TSDF vs SS tetra, same 6 cameras as the earlier
  steps (render.py compare, --crop 0.4), using the camera-visible layers.
- `distance_heat.png`: each SS mesh coloured by distance to the reference, and the reference coloured by distance to
  the SS TSDF (viridis 0-5 mm; grey > 10 mm).
- `compare_geometry_raw.png`: the meshes as stored (the reference row shows its inner layer, see above).
- `compare_geometry_closeup.png`: 8 views from 0.22 m. Shows skin noise vs smoothness and the arc band edges.
- `splat_views.png`: face front (001428), +x profile (001044) and -x profile (000140). Columns: input frame, splat
  render, rendered normal, median depth.
- `gaussian_centres.png`: Gaussian centres with opacity >= 0.5 in the head cube.

## Files
- `head/tsdf.ply`, `head/tetra.ply`: meshes as extracted (device frame, metres, vertex colours).
- `head/visible/*_visible.ply`: camera-visible layers, including the reference's.
- `head/heat/*.ply`: distance-coloured meshes.
- `head/gaussian_centres.ply`: all 103k centres in the cube, with RGB from the SH DC term, `opacity`, and `scale` (largest axis, m).
- `head/tetra/recon*.ply`: the tetra extractor's intermediates.
- `model/point_cloud/iteration_30000/point_cloud.ply`: the full splat (750k Gaussians; the repo's SIBR viewer loads it).
  Also `model/cfg_args`, `model/cameras.json` and `model/multi_view.json` (the 8 NCC neighbours per view).
- `logs/`: build, train, extract and compare logs, plus nvidia-smi memory traces.

Not uploaded: the 15k checkpoint, the 7k snapshot, the trainer's debug panels, and copies of the reference and CAD.
