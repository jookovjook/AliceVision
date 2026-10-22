#!/usr/bin/env bash
# Fast pipeline for one capture (no SIFT SfM): keyframes -> tag36h11 corners -> tag-only poses in the device frame
# (tag_poses.py: PnP + BA on the frame tags, 4 segments, ARKit relative prior) -> tag-corner landmarks
# (tag_triangulate.py) -> GS export -> Stochastic Solids with the 30k schedule scaled to ITERS -> TSDF mesh ->
# distance to the full-run mesh (REF) -> CAD fit + image fit -> essentials to Drive.
#   bash headscan/steps/fast_capture.sh CAPTURE [STAGES]      CAPTURE = folder in /workspace/data (Stray Scanner)
#   STAGES (default all): frames,init,tags,poses,joint,export,ss,compare,fit,upload
#   joint = SIFT between neighbouring keyframes + joint bundle adjustment of tag corners and SIFT tracks (env SIFT_MAX
#   tracks, default 8000; SIFT_NEIGHBOURS frames, default 8; intrinsics stay fixed (ARKit) unless JOINT_INTRINSICS=1:
#   freeing them moved the ear knobs by ~0.5 mm on the table captures)
# Env: WIN (8), ITERS (7000), SCALE (1 head / 3 table: rasterizer near plane 0.2), CENTER/HALF (extraction cube, m),
#      REF (full-run TSDF mesh to compare with), RULER (head: ruler gaps for image_fit.py --ruler), DRIVE,
#      CALIB (calibration constants for apply_calibration.py; default data/calibration_constants_oct22.json).
# Final output: readouts.json (calibrated screw gaps, knob K) next to the mesh and the fits.
set -euo pipefail
CAP=$1; STAGES=${2:-frames,init,tags,poses,joint,export,ss,compare,fit,upload}
HS=/workspace/AliceVision/headscan; PY=/workspace/hsvenv/bin/python
CONDA=${CONDA:-/workspace/miniforge}; REPO=${REPO:-/workspace/src/gsss}
D=/workspace/data/$CAP; W=/workspace/fast/$CAP
LAYOUT=${LAYOUT:-/workspace/arc/tags/frame_tags_layout.json}
WIN=${WIN:-8}; ITERS=${ITERS:-7000}; SCALE=${SCALE:-1}
CENTER=${CENTER:-"0.02 0.04 0.04"}; HALF=${HALF:-0.22}
DRIVE=${DRIVE:-${DRIVE_ROOT:-gdrive:frame_capture/fast_pipeline}/$CAP}
export AV_CORES=${AV_CORES:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} MKL_NUM_THREADS=${OMP_NUM_THREADS:-24} PYTHONUNBUFFERED=1
mkdir -p "$W/logs"
exec 9>"$W/.lock"; flock 9
has() { [[ ",$STAGES," == *",$1,"* ]]; }
t() { echo "[$CAP] $1  $(date +%H:%M:%S)"; date +%s > "$W/logs/t_$1"; }

if has frames && [ ! -f "$W/frames/keyframes.csv" ]; then
  t frames; $PY -I "$HS/extract_frames.py" "$D" "$W/frames" --window "$WIN" > "$W/logs/frames.log"
fi
if has init && [ ! -f "$W/sfm/00_init/cameraInit.sfm" ]; then
  t init; (. /opt/av/env.sh && $PY "$HS/pipeline.py" "$W/sfm" --capture "$D" --frames "$W/frames/keyframes" \
    --poses arkit-refine --distortion lut-inv --stages init > "$W/logs/init.log" 2>&1)
fi
if has tags && [ ! -f "$W/feat_tags/.done" ]; then
  t tags; mkdir -p "$W/feat_tags"
  (. /opt/av/env.sh && aliceVision_featureExtraction --input "$W/sfm/00_init/cameraInit.sfm" --output "$W/feat_tags" \
    --describerTypes tag36h11 --forceCpuExtraction 1 --maxThreads "$AV_CORES" --verboseLevel info > "$W/logs/tags.log" 2>&1)
  touch "$W/feat_tags/.done"
fi
if has poses && [ ! -f "$W/device/sfm_device.sfm" ]; then
  t poses; mkdir -p "$W/device"
  $PY -I "$HS/tag_poses.py" "$D/" "$W/sfm/00_init/cameraInit.sfm" "$W/feat_tags" "$LAYOUT" "$W/device/tag_poses.sfm" \
    --segments --arkit-prior --arkit-sd 0.5,0.05 > "$W/logs/poses.log" 2>&1
  $PY -I "$HS/tag_triangulate.py" "$W/device/tag_poses.sfm" "$W/feat_tags" "$W/device/sfm_device.sfm" >> "$W/logs/poses.log" 2>&1
  tail -2 "$W/logs/poses.log"
fi
if has joint && [ ! -f "$W/device/joint.done" ]; then
  # joint pose refinement: SIFT on the keyframes, matched between neighbouring frames, tracks -> one bundle adjustment of
  # the frame-tag corners + these tracks (poses fit the head as well as the frame); replaces the tag-only device sfm
  t joint; S2=$W/sift; mkdir -p "$S2/feat" "$S2/match"
  (. /opt/av/env.sh
   aliceVision_featureExtraction --input "$W/sfm/00_init/cameraInit.sfm" --output "$S2/feat" --describerTypes dspsift \
     --describerPreset normal --forceCpuExtraction 1 --maxThreads "$AV_CORES" --verboseLevel info > "$W/logs/sift_feat.log" 2>&1
   aliceVision_imageMatching --input "$W/sfm/00_init/cameraInit.sfm" --featuresFolders "$S2/feat" --method Sequential \
     --nbNeighbors "${SIFT_NEIGHBOURS:-8}" --output "$S2/pairs.txt" --maxCoresAvailable "$AV_CORES" --verboseLevel info > "$W/logs/sift_pairs.log" 2>&1
   aliceVision_featureMatching --input "$W/sfm/00_init/cameraInit.sfm" --featuresFolders "$S2/feat" --imagePairsList "$S2/pairs.txt" \
     --output "$S2/match" --describerTypes dspsift --guidedMatching 0 --maxCoresAvailable "$AV_CORES" --verboseLevel info > "$W/logs/sift_match.log" 2>&1
   aliceVision_tracksBuilding --input "$W/sfm/00_init/cameraInit.sfm" --featuresFolders "$S2/feat" --matchesFolders "$S2/match" \
     --describerTypes dspsift --output "$S2/tracks.json" --maxCoresAvailable "$AV_CORES" --verboseLevel info > "$W/logs/sift_tracks.log" 2>&1)
  t joint_ba
  $PY -I "$HS/tag_poses.py" "$D/" "$W/sfm/00_init/cameraInit.sfm" "$W/feat_tags" "$LAYOUT" "$W/device/joint_poses.sfm" \
    --segments --arkit-prior --arkit-sd 0.5,0.05 ${JOINT_INTRINSICS:+--intrinsics} --sift "$S2/tracks.json" --sift-max "${SIFT_MAX:-8000}" > "$W/logs/joint.log" 2>&1
  $PY -I "$HS/tag_triangulate.py" "$W/device/joint_poses.sfm" "$W/feat_tags" "$W/device/sfm_device.sfm" >> "$W/logs/joint.log" 2>&1
  grep -E "SIFT|BA:" "$W/logs/joint.log"; touch "$W/device/joint.done"
fi
if has export && [ ! -f "$W/gs/dataset/export_check.json" ]; then
  t export; $PY -I "$HS/gs_export.py" "$W/device/sfm_device.sfm" "$D" "$W/gs/dataset" --scale "$SCALE" > "$W/logs/export.log" 2>&1
fi
if has ss && [ ! -f "$W/gs/head/tsdf.ply" ]; then
  t ss; S=$W/gs; mkdir -p "$S/data/sparse/0"
  ln -sfn "$S/dataset/images" "$S/data/images"
  for f in cameras.txt images.txt points3D.txt; do ln -sf "$S/dataset/sparse/0/$f" "$S/data/sparse/0/$f"; done
  source "$CONDA/etc/profile.d/conda.sh"; conda activate gggss
  # the repo's 30k schedule scaled to ITERS: densify until N/2, multi-view regularisation from 7N/30, opacity reset N/10
  (cd "$REPO" && python train.py -s "$S/data" -m "$S/model" -r 1 --use_decoupled_appearance 3 --data_device cpu \
    --iterations "$ITERS" --position_lr_max_steps "$ITERS" --densify_until_iter $((ITERS / 2)) \
    --regularization_from_iter $((ITERS * 7 / 30)) --opacity_reset_interval $((ITERS / 10)) \
    --save_iterations "$ITERS" --test_iterations "$ITERS" \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "Training progress.*it/s\]$" > "$W/logs/train.log") || true
  t ss_extract
  PYTORCH_ALLOC_CONF=expandable_segments:True python "$HS/gs/ss_head_extract.py" --repo "$REPO" -m "$S/model" --out "$S/head" \
    --center $CENTER --half $HALF --scale "$SCALE" --what centres tsdf --voxel 0.001 --trunc 0.006 \
    2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "progress.*it/s\]$" > "$W/logs/extract.log"
  python "$HS/gs/ss_render_views.py" --repo "$REPO" -m "$S/model" --out "$W/figures/splat_views.png" --center $CENTER --scale "$SCALE" > /dev/null 2>&1 || true
  conda deactivate
fi
if has compare && [ -n "${REF:-}" ] && [ ! -f "$W/compare.json" ]; then
  t compare; /workspace/miniforge/envs/gggss/bin/python "$HS/gs/surface_compare.py" "$REF" "$W/compare.json" --center $CENTER --half $HALF \
    --n 300000 fast_tsdf="$W/gs/head/tsdf.ply" > "$W/logs/compare.log" 2>&1 || true
  grep median "$W/logs/compare.log" | tail -2 || true
fi
if has fit && [ ! -f "$W/fit/img/image_fit.json" ]; then
  t fit; mkdir -p "$W/fit/in"
  ln -sfn "$W/device/sfm_device.sfm" "$W/fit/in/sfm_device.sfm"; ln -sfn "$W/feat_tags" "$W/fit/in/feat_tags"
  ln -sfn "$W/gs/head/tsdf.ply" "$W/fit/in/ss_tetra.ply"
  for f in frame_L116_A_plates.stl frame_L116_B_plates.stl frame_tags_layout.json screw_wing.stl; do ln -sfn "$(dirname "$LAYOUT")/$f" "$W/fit/in/$f"; done
  $PY -I "$HS/cad_fit.py" "$W/fit/in" "$W/fit/cad" ${RULER:+--ruler $RULER} > "$W/logs/cad_fit.log" 2>&1
  $PY -I "$HS/image_fit.py" "$W/fit/in" "$W/fit/cad/cad_fit.json" "$W/fit/img" --free-axis ${RULER:+--ruler $RULER} > "$W/logs/image_fit.log" 2>&1
  grep -E "reprojection|coarse gap|knobs" "$W/logs/image_fit.log"
  # calibrated readouts (per-screw constants from the frame-on-the-table captures)
  $PY -I "$HS/apply_calibration.py" "$W/fit/img/image_fit.json" "${CALIB:-$HS/data/calibration_constants_oct22.json}" \
    "$W/readouts.json" ${RULER:+--ruler $RULER} | tee "$W/logs/readouts.log"
fi
t done
# stage durations
$PY -I - "$W/logs" > "$W/times.json" <<'EOF'
import sys, os, json
L = sys.argv[1]; order = ["frames", "init", "tags", "poses", "joint", "joint_ba", "export", "ss", "ss_extract", "compare", "fit", "done"]
t = {k: int(open(f"{L}/t_{k}").read()) for k in order if os.path.exists(f"{L}/t_{k}")}
ks = [k for k in order if k in t]
print(json.dumps({a: t[b] - t[a] for a, b in zip(ks, ks[1:])}))
EOF
cat "$W/times.json"
if has upload; then
  # essentials only: poses, tag corners, mesh, fits, figures, logs (frames, export images, the Gaussian model stay on the box)
  # Drive is slow with many small files: the tag detections go up as one archive; in the background so the GPU queue moves on
  tar -C "$W" -czf "$W/feat_tags.tar.gz" feat_tags
  nohup rclone copy "$W" "$DRIVE" --transfers 8 --exclude "frames/all/**" --exclude "frames/keyframes/**" --exclude "gs/dataset/**" \
    --exclude "gs/data/**" --exclude "gs/model/**" --exclude "sfm/0[1-9]*/**" --exclude "fit/in/**" --exclude "feat_tags/**" --exclude "sift/**" \
    > "$W/logs/upload.log" 2>&1 &
fi
