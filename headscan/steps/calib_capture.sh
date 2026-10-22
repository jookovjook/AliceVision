#!/usr/bin/env bash
# One frame-on-the-table capture, end to end on the box: keyframes -> AliceVision SfM from ARKit (refined) -> tag36h11
# landmarks -> device frame from the frame-tag layout -> GS export x SCALE -> Stochastic Solids (train, extract,
# scaled back to device metres) -> Drive.
#   bash headscan/steps/calib_capture.sh CAPTURE [STAGES]      CAPTURE = folder in /workspace/data (Stray Scanner)
#   STAGES (default all): frames,sfm,tags,device,export,ss_train,ss_extract,fit,upload
# Every stage is skipped when its output exists. Env: WIN (keyframe window, 8), SCALE (3: nearest geometry sits at
# 0.17-0.25 m, inside the rasterizer's 0.2 near plane), CENTER/HALF (extraction cube, device metres).
set -euo pipefail
CAP=$1; STAGES=${2:-frames,sfm,tags,device,export,ss_train,ss_extract,fit,upload}
HS=/workspace/AliceVision/headscan
PY=/workspace/hsvenv/bin/python
D=/workspace/data/$CAP; W=/workspace/work/$CAP
LAYOUT=${LAYOUT:-/workspace/arc/tags/frame_tags_layout.json}
WIN=${WIN:-8}; SCALE=${SCALE:-3}
# frame + screws at gap 0 (wing ends 100 mm out of the holes) + knobs at 61 mm: x +-0.25, y +-0.22, z -0.04..0.25
CENTER=${CENTER:-"0 0 0.1"}; HALF=${HALF:-0.26}
DRIVE=${DRIVE:-gdrive:frame_capture/calibration_run/$CAP}
# the box shows every host CPU (256) but has a ~61-core quota and a pids limit: cap AliceVision / OpenMP threads
export AV_CORES=${AV_CORES:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
mkdir -p "$W" /workspace/logs
exec 9>"$W/.lock"; flock 9   # one invocation per capture at a time (parallel CPU jobs, orchestrators, fit waiter)
has() { [[ ",$STAGES," == *",$1,"* ]]; }
t() { echo "[$CAP] $1  $(date +%H:%M:%S)"; }

if has frames && [ ! -f "$W/frames/keyframes.csv" ]; then
  t frames; $PY -I "$HS/extract_frames.py" "$D" "$W/frames" --window "$WIN"
fi
if has sfm && [ ! -f "$W/sfm/04_sfm/sfm.json" ]; then
  t sfm; (. /opt/av/env.sh && $PY "$HS/pipeline.py" "$W/sfm" --capture "$D" --frames "$W/frames/keyframes" \
    --poses arkit-refine --distortion lut-inv --stages init,features,matching,tracks,sfm)
fi
if has tags && [ ! -f "$W/tags/sfm/sfm.sfm" ]; then
  t tags; bash "$HS/steps/step3_tags_sfm.sh" "$W/sfm" "$W/tags"
fi
if has device && [ ! -f "$W/device/sfm_device.sfm" ]; then
  t device; mkdir -p "$W/device"
  $PY -I "$HS/layout_to_sfm.py" "$LAYOUT" "$W/device/layout_ref.sfm"
  (. /opt/av/env.sh && aliceVision_sfmAlignment --input "$W/tags/sfm/sfm.sfm" --reference "$W/device/layout_ref.sfm" \
    --method from_markers --featuresFolders "$W/tags/feat_tags" --output "$W/device/sfm_device.sfm" \
    --outputViewsAndPoses "$W/device/cameras_device.sfm" --maxCoresAvailable "$AV_CORES" --verboseLevel info > "$W/device/align.log" 2>&1)
  grep -iE "marker|scale" "$W/device/align.log" | tail -5 || true
fi
if has export && [ ! -f "$W/gs/dataset/export_check.json" ]; then
  t export; $PY -I "$HS/gs_export.py" "$W/device/sfm_device.sfm" "$D" "$W/gs/dataset" --scale "$SCALE"
fi
SSENV=(DATASET="$W/gs/dataset" OUT="$W/gs/ss" SCALE="$SCALE" CENTER="$CENTER" HALF="$HALF" DRIVE="$DRIVE/gs")
if has ss_train && [ ! -f "$W/gs/ss/model/point_cloud/iteration_30000/point_cloud.ply" ]; then
  t ss_train; env "${SSENV[@]}" bash "$HS/gs/run_stochastic_solids.sh" train
fi
if has ss_extract && [ ! -f "$W/gs/ss/head/tetra.ply" ]; then
  t ss_extract; env "${SSENV[@]}" bash "$HS/gs/run_stochastic_solids.sh" extract
fi
if has fit && [ ! -f "$W/fit/image_fit_free/image_fit.json" ]; then
  # CAD fit (tags + ICP on the Stochastic Solids tetra mesh) -> image-space fit of frame, screws and knobs
  t fit; mkdir -p "$W/fit/in"
  ln -sfn "$W/device/sfm_device.sfm" "$W/fit/in/sfm_device.sfm"; ln -sfn "$W/tags/feat_tags" "$W/fit/in/feat_tags"
  ln -sfn "$W/gs/ss/head/tetra.ply" "$W/fit/in/ss_tetra.ply"
  for f in frame_L116_A_plates.stl frame_L116_B_plates.stl frame_tags_layout.json screw_wing.stl; do ln -sfn "$(dirname "$LAYOUT")/$f" "$W/fit/in/$f"; done
  $PY -I "$HS/cad_fit.py" "$W/fit/in" "$W/fit/cad" > "$W/fit/cad_fit.log" 2>&1
  $PY -I "$HS/image_fit.py" "$W/fit/in" "$W/fit/cad/cad_fit.json" "$W/fit/image_fit_free" --free-axis > "$W/fit/image_fit_free.log" 2>&1
  tail -4 "$W/fit/image_fit_free.log"
fi
if has upload; then
  t upload
  # frames and the export images are reproducible from the capture zip + sfm_device.sfm; symlinks are skipped
  rclone copy "$W" "$DRIVE" --transfers 8 --max-size 2G --exclude "frames/all/**" --exclude "frames/keyframes/**" \
    --exclude "gs/dataset/images/**" --exclude "gs/dataset/depth*/**" --exclude "gs/ss/data/**" \
    --exclude "gs/ss/model/debug/**" --exclude "gs/ss/model/chkpnt*" --exclude "gs/ss/model/point_cloud/iteration_7000/**" \
    --exclude "sfm/01_features/**" --exclude "sfm/02_matching/**" --exclude "tags/match_tags/**" 2>&1 | grep -v NOTICE || true
fi
t done
