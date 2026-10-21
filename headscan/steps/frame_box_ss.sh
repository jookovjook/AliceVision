#!/usr/bin/env bash
# Test 2: fast Stochastic Solids inside the device-frame box. One run = frame_box_dataset.py (LiDAR in-box mask + LiDAR
# init points) -> SS train with the repo schedule scaled to ITERS -> extract (tetra + TSDF, cube) -> distance to the
# full SS mesh of the same capture.
#   bash headscan/steps/frame_box_ss.sh NAME EXPORT_DIR REF_MESH OUT_ROOT
# Env: ITERS (7000), RES (1 = export size, 2 = half), LO/HI (box, device metres), CENTER/HALF (extraction + compare
# cube), SCALE (export units per metre), BOX_ARGS (extra frame_box_dataset.py flags, e.g. "--views-every 2").
# Schedule for N iterations (repo defaults at 30k): densify until N/2, multi-view regularisation from 7N/30, opacity reset
# every N/10, position lr decays over N.
set -euo pipefail
NAME=$1; EXPORT=$2; REF=$3; ROOT=$4
ITERS=${ITERS:-7000}; RES=${RES:-1}; SCALE=${SCALE:-1}
LO=${LO:-"-0.26 -0.22 -0.20"}; HI=${HI:-"0.26 0.30 0.28"}
CENTER=${CENTER:-"0.02 0.04 0.04"}; HALF=${HALF:-0.22}
HS=/workspace/AliceVision/headscan; PY=/workspace/hsvenv/bin/python
CONDA=${CONDA:-/workspace/miniforge}; REPO=${REPO:-/workspace/src/gsss}
O=$ROOT/$NAME; mkdir -p "$O/logs"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} MKL_NUM_THREADS=${OMP_NUM_THREADS:-24} PYTHONUNBUFFERED=1
T0=$(date +%s)
[ -f "$O/data/box_dataset.json" ] || $PY -I "$HS/gs/frame_box_dataset.py" "$EXPORT" "$O/data" --lo $LO --hi $HI --scale "$SCALE" ${BOX_ARGS:-} > "$O/logs/dataset.log" 2>&1
T1=$(date +%s)
source "$CONDA/etc/profile.d/conda.sh"; conda activate gggss
if [ ! -f "$O/model/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$O/logs/train_gpu_mem.csv" & SMI=$!
  (cd "$REPO" && python train.py -s "$O/data" -m "$O/model" -r "$RES" --use_decoupled_appearance 3 --data_device cpu \
    --iterations "$ITERS" --position_lr_max_steps "$ITERS" --densify_until_iter $((ITERS / 2)) \
    --regularization_from_iter $((ITERS * 7 / 30)) --opacity_reset_interval $((ITERS / 10)) \
    --save_iterations "$ITERS" --test_iterations "$ITERS" \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "Training progress.*it/s\]$" > "$O/logs/train.log") || true
  kill $SMI || true
fi
T2=$(date +%s)
if [ ! -f "$O/head/tetra.ply" ]; then
  PYTORCH_ALLOC_CONF=expandable_segments:True python "$HS/gs/ss_head_extract.py" --repo "$REPO" -m "$O/model" --out "$O/head" \
    --center $CENTER --half $HALF --scale "$SCALE" --what centres tsdf tetra --voxel 0.001 --trunc 0.006 \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "progress.*it/s\]$" > "$O/logs/extract.log"
fi
T3=$(date +%s)
python "$HS/gs/surface_compare.py" "$REF" "$O/compare.json" --center $CENTER --half $HALF --n 300000 \
  tetra="$O/head/tetra.ply" tsdf="$O/head/tsdf.ply" > "$O/logs/compare.log" 2>&1
echo "{\"name\": \"$NAME\", \"iters\": $ITERS, \"res\": $RES, \"dataset_s\": $((T1 - T0)), \"train_s\": $((T2 - T1)), \"extract_s\": $((T3 - T2)), \"peak_gpu_mib\": $(sort -n "$O/logs/train_gpu_mem.csv" 2>/dev/null | tail -1 || echo 0)}" > "$O/times.json"
cat "$O/times.json"; grep -E "tetra|tsdf" "$O/logs/compare.log" | grep median
