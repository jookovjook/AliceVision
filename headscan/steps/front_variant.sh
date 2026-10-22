#!/usr/bin/env bash
# Face-only Stochastic Solids on the head capture: views that look at the face (gs/front_views.py) -> 7k schedule ->
# TSDF in a face cube -> distance to the full 30k mesh in that cube; the all-views 7k mesh of fast_capture.sh is scored in
# the same cube for comparison.
#   bash headscan/steps/front_variant.sh NAME MAX_ANGLE_DEG     (after fast_capture.sh 6ba4b7d84f has run)
set -euo pipefail
NAME=$1; ANG=$2
HS=/workspace/AliceVision/headscan; PY=/workspace/hsvenv/bin/python
CONDA=${CONDA:-/workspace/miniforge}; REPO=${REPO:-/workspace/src/gsss}
F=/workspace/fast/6ba4b7d84f; O=$F/front/$NAME; REF=/workspace/ref/6ba4b7d84f_tsdf.ply
ITERS=${ITERS:-7000}; CENTER="0 0.07 -0.02"; HALF=0.11   # face: ear to ear, forehead to chin
mkdir -p "$O/logs"
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTHONUNBUFFERED=1
T0=$(date +%s)
$PY -I "$HS/gs/front_views.py" "$F/gs/dataset" "$O/data" --max-angle "$ANG" | tee "$O/logs/views.log"
source "$CONDA/etc/profile.d/conda.sh"; conda activate gggss
T1=$(date +%s)
(cd "$REPO" && python train.py -s "$O/data" -m "$O/model" -r 1 --use_decoupled_appearance 3 --data_device cpu \
  --iterations "$ITERS" --position_lr_max_steps "$ITERS" --densify_until_iter $((ITERS / 2)) \
  --regularization_from_iter $((ITERS * 7 / 30)) --opacity_reset_interval $((ITERS / 10)) \
  --save_iterations "$ITERS" --test_iterations "$ITERS" \
  2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "Training progress.*it/s\]$" > "$O/logs/train.log") || true
T2=$(date +%s)
PYTORCH_ALLOC_CONF=expandable_segments:True python "$HS/gs/ss_head_extract.py" --repo "$REPO" -m "$O/model" --out "$O/head" \
  --center $CENTER --half $HALF --what tsdf --voxel 0.001 --trunc 0.006 > "$O/logs/extract.log" 2>&1
T3=$(date +%s)
python "$HS/gs/surface_compare.py" "$REF" "$O/compare.json" --center $CENTER --half $HALF --n 300000 \
  front="$O/head/tsdf.ply" allviews_7k="$F/gs/head/tsdf.ply" > "$O/logs/compare.log" 2>&1
echo "{\"name\": \"$NAME\", \"max_angle\": $ANG, \"views\": $(python -c "import json;print(json.load(open('$O/data/front_views.json'))['views_kept'])"), \"select_s\": $((T1 - T0)), \"train_s\": $((T2 - T1)), \"extract_s\": $((T3 - T2))}" > "$O/times.json"
cat "$O/times.json"; grep median "$O/logs/compare.log" || true
rclone copy "$O" "gdrive:frame_capture/fast_pipeline/6ba4b7d84f/front/$NAME" --exclude "data/**" --exclude "model/**" 2>&1 | grep -v NOTICE || true
