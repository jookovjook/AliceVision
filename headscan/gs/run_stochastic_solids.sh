#!/bin/bash
# Stochastic Solids (Geometry-Grounded Gaussian Splatting, arXiv 2601.17835, HKUST-SAIL; Inria-style licence:
# research/evaluation only) on the head+arc capture, RGB only, poses frozen, device frame in metres.
#
#   bash run_stochastic_solids.sh setup|train|extract|compare|figures|upload|all
#
# The dataset (headscan GS export) is a COLMAP text model: one centred PINHOLE camera (undistorted keyframes),
# world->camera poses in the device frame (metres), points3D without tracks. The trainer reads it as-is (no COLMAP
# run, no re-centring or rescaling: Scene only uses the camera spread for the densification extent), and its poses
# are not optimised, so every output is already in the device frame. Cameras sit 0.60-0.75 m from the head, above
# the rasterizer's hard-coded 0.2 near plane, so no rescale is needed.
set -euo pipefail
STAGE=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
AV=${AV:-$(cd "$HERE/../.." && pwd)}
REPO=${REPO:-/workspace/src/gsss}
REPO_COMMIT=${REPO_COMMIT:-23b418c2f0f19b30b251acad6fe13445dae968c7}
CONDA=${CONDA:-/workspace/miniforge}
DATASET=${DATASET:-/workspace/work/gs/dataset}
OUT=${OUT:-/workspace/work/gs/stochastic}
REF_MESH=${REF_MESH:-/workspace/work/ref/tsdf_mvs.ply}
REF_SFM=${REF_SFM:-/workspace/work/ref/sfm_device.sfm}
VIZ_VENV=${VIZ_VENV:-/workspace/venv-viz}
RES=${RES:-1}                       # -r 1 = full 1816x1360
DATA_DEVICE=${DATA_DEVICE:-cpu}     # 248 full-res frames as float32 RGB+grey = 9.8 GB; keep them off the GPU
CENTER="0.02 0.04 0.04"; HALF=0.22  # head cube, device frame
DRIVE=${DRIVE:-gdrive:frame_capture/6ba4b7d84f_recon/gs/stochastic_solids}
MODEL=$OUT/model
mkdir -p "$OUT/logs"

conda_env() { source "$CONDA/etc/profile.d/conda.sh"; conda activate gggss; }
# the box reports 255 CPUs but has a ~30-core quota; unbuffered so the logs follow the run
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} MKL_NUM_THREADS=${OMP_NUM_THREADS:-24} PYTHONUNBUFFERED=1

setup() {
  [ -d "$CONDA" ] || { wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh; bash /tmp/mf.sh -b -p "$CONDA"; }
  [ -d "$REPO" ] || git clone --recursive https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids.git "$REPO"
  git -C "$REPO" checkout -q "$REPO_COMMIT"
  source "$CONDA/etc/profile.d/conda.sh"
  conda env list | grep -q '^gggss ' || conda create -y -n gggss python=3.12
  conda activate gggss
  conda install -y -c conda-forge cgal gmp
  # torch built for CUDA 12.4 to match the image's nvcc (extensions compiled for sm_89 only)
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r "$REPO/requirements.txt" tensorboard
  export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=24 CPATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib
  for m in diff-gaussian-rasterization warp-patch-ncc simple-knn tetra_triangulation; do
    pip install "$REPO/submodules/$m" --no-build-isolation
  done
  pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
  # figures: headscan/viz/render.py (pyrender + EGL)
  apt-get install -y -qq libglu1-mesa libegl1 libgl1 >/dev/null
  [ -d "$VIZ_VENV" ] || python3 -m venv "$VIZ_VENV"
  "$VIZ_VENV/bin/pip" install -q numpy scipy trimesh pyrender "PyOpenGL==3.1.7" "networkx>=3" matplotlib \
    opencv-python-headless OpenEXR scikit-image plyfile
}

train() {
  # source dir of symlinks so the loader's one-off points3D.ply conversion does not write into the shared export
  local SRC=$OUT/data
  mkdir -p "$SRC/sparse/0"
  ln -sfn "$DATASET/images" "$SRC/images"
  for f in cameras.txt images.txt points3D.txt; do ln -sf "$DATASET/sparse/0/$f" "$SRC/sparse/0/$f"; done
  conda_env; cd "$REPO"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$OUT/logs/train_gpu_mem.csv" &
  local SMI=$!
  local T0=$(date +%s)
  # DTU/TnT recipe of the repo (scripts/dtu.sh, tnt.sh): PGSR decoupled appearance, default 30k iterations, black bg
  python train.py -s "$SRC" -m "$MODEL" -r "$RES" --use_decoupled_appearance 3 --data_device "$DATA_DEVICE" \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "Training progress.*it/s\]$" > "$OUT/logs/train.log" || true
  kill $SMI || true
  echo "train_seconds $(( $(date +%s) - T0 ))  peak_gpu_mib $(sort -n "$OUT/logs/train_gpu_mem.csv" | tail -1)" | tee -a "$OUT/logs/train.log"
}

extract() {
  conda_env
  local T0=$(date +%s)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$OUT/logs/extract_gpu_mem.csv" &
  local SMI=$!
  PYTORCH_ALLOC_CONF=expandable_segments:True python "$HERE/ss_head_extract.py" --repo "$REPO" -m "$MODEL" --out "$OUT/head" \
    --center $CENTER --half $HALF --what centres tsdf tetra --voxel 0.001 --trunc 0.006 ${MOVE_CPU:+--move_cpu} \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "progress.*it/s\]$" > "$OUT/logs/extract.log"
  kill $SMI || true
  echo "extract_seconds $(( $(date +%s) - T0 ))  peak_gpu_mib $(sort -n "$OUT/logs/extract_gpu_mem.csv" | tail -1)" | tee -a "$OUT/logs/extract.log"
  python "$HERE/ss_render_views.py" --repo "$REPO" -m "$MODEL" --out "$OUT/figures/splat_views.png" --center $CENTER
}

compare() {
  conda_env
  local TESTS=(tsdf="$OUT/head/tsdf.ply" tetra="$OUT/head/tetra.ply" centres="$OUT/head/gaussian_centres.ply")
  # raw meshes as extracted
  python "$HERE/surface_compare.py" "$REF_MESH" "$OUT/compare_raw.json" --center $CENTER --half $HALF "${TESTS[@]}" \
    | tee "$OUT/logs/compare_raw.log"
  # camera-visible layer of every mesh (drops the reference TSDF's second layer at the back of its truncation band)
  python "$HERE/surface_compare.py" "$REF_MESH" "$OUT/compare.json" --center $CENTER --half $HALF "${TESTS[@]}" \
    --visible "$DATASET/sparse/0" --clean-dir "$OUT/head/visible" --heat-dir "$OUT/head/heat" | tee "$OUT/logs/compare.log"
  # independent check on the rigid frame: CAD placed in the device frame from the tag layout (step 6/7)
  [ -f "$OUT/cad/cad_frame_device_m.ply" ] || rclone copy gdrive:frame_capture/6ba4b7d84f_recon/step6_7/cad_frame_device_m.ply "$OUT/cad/"
  python "$HERE/cad_check.py" "$OUT/cad/cad_frame_device_m.ply" "$OUT/cad_check.json" ref="$OUT/head/visible/ref_visible.ply" \
    ss_tsdf="$OUT/head/visible/tsdf_visible.ply" ss_tetra="$OUT/head/visible/tetra_visible.ply" | tee "$OUT/logs/cad_check.log"
}

figures() {
  mkdir -p "$OUT/figures"
  local R="$VIZ_VENV/bin/python $AV/headscan/viz/render.py compare $REF_SFM" V=$OUT/head/visible
  export PYOPENGL_PLATFORM=egl
  # camera-visible layers, faces turned to the cameras (render.py culls back faces; the raw reference TSDF is wound
  # inside-out and would show its inner truncation-band layer)
  $R "$OUT/figures/compare_geometry.png" "TSDF of AliceVision MVS (ref)=$V/ref_visible.ply:noflip" \
     "Stochastic Solids TSDF=$V/tsdf_visible.ply:noflip" "Stochastic Solids tetra=$V/tetra_visible.ply:noflip" --crop 0.4
  $R "$OUT/figures/compare_textured.png" "TSDF of AliceVision MVS (ref)=$V/ref_visible.ply:noflip" \
     "Stochastic Solids TSDF=$V/tsdf_visible.ply:noflip" "Stochastic Solids tetra=$V/tetra_visible.ply:noflip" --crop 0.4 --textured
  $R "$OUT/figures/distance_heat.png" "SS TSDF, 0-5 mm to ref=$OUT/head/heat/tsdf_heat.ply:noflip" \
     "SS tetra, 0-5 mm to ref=$OUT/head/heat/tetra_heat.ply:noflip" "ref, 0-5 mm to SS TSDF=$OUT/head/heat/ref_vs_tsdf_heat.ply:noflip" \
     --crop 0.4 --textured
  # as extracted, for the record (reference row = its inner layer, see README)
  $R "$OUT/figures/compare_geometry_raw.png" "ref TSDF as stored=$REF_MESH:noflip" \
     "Stochastic Solids TSDF=$OUT/head/tsdf.ply:noflip" "Stochastic Solids tetra=$OUT/head/tetra.ply:noflip" --crop 0.4
  HEADSCAN_VIZ=$AV/headscan/viz "$VIZ_VENV/bin/python" "$HERE/points_view.py" "$REF_SFM" "$OUT/figures/gaussian_centres.png" \
     "Gaussian centres (opacity 0.5+)=$OUT/head/gaussian_centres.ply" --crop 0.4
}

upload() {
  cp "$HERE/README_stochastic_solids.md" "$OUT/README.md"
  # no >2 GB files; skip the 15k checkpoint, the 7k snapshot, the trainer's per-200-iteration debug panels and copies
  rclone copy "$OUT" "$DRIVE" --max-size 2G --transfers 8 --exclude "model/debug/**" --exclude "data/**" \
    --exclude "model/chkpnt*" --exclude "model/point_cloud/iteration_7000/**" --exclude "test7k/**" \
    --exclude "ref_check/**" --exclude "cad/**"
}

case $STAGE in
  all) train; extract; compare; figures; upload ;;
  *) $STAGE ;;
esac
