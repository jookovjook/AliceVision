#!/usr/bin/env bash
# Steps 6-8 on any head mesh in the device frame (metres, vertex colours): segmentation, device readouts,
# CAD-screw check against the mesh, face landmarks, report.
#   bash headscan/steps/step8_on_mesh.sh MESH.ply OUT_DIR "TITLE"
# Tag-based readouts (screw gaps, ear) come from the step-3/4 SfM and do not depend on the mesh; the hair/skin surface
# under every screw, the CAD-screw check and the face landmarks do.
set -euo pipefail
MESH=$1; OUT=$2; TITLE=${3:-}
W=/workspace/work; HS=/workspace/AliceVision/headscan; PY="/workspace/hsvenv/bin/python -I"
mkdir -p "$OUT"
$PY $HS/segment_head.py "$MESH" /workspace/arc "$OUT/seg" > "$OUT/segment.log"
$PY $HS/device_readout.py $W/step4/sfm_device.sfm $W/step3/run/feat_tags /workspace/arc/tags/frame_tags_layout.json \
    "$OUT/seg/head_closed_flags.ply" "$OUT/readout" --skin "$OUT/seg/head_skin.ply" > "$OUT/readout.log"
$PY $HS/check_screws.py "$OUT/readout.json" /workspace/arc "$OUT/seg/head_all.ply" "$OUT/screw_check.json" > "$OUT/screw_check.log"
$PY $HS/face_landmarks_render.py "$OUT/seg/head_closed.ply" "$OUT/facelm" 2>&1 | grep -v -E "^W0000|^I0000|absl|inference_feedback|TfLite|XNNPACK|gl_context|mediapipe/|^WARNING" > "$OUT/facelm.log"
$PY $HS/step8_report.py "$OUT/readout.json" "$OUT/facelm.json" "$OUT/step8_readouts" --screw-check "$OUT/screw_check.json" --title "$TITLE" > "$OUT/report.log"
R="$PY $HS/viz/render.py compare $W/step4/sfm_device.sfm"
$R "$OUT/step7_labels.png" "labels: frame blue, screws orange, ear green=$OUT/seg/head_labels.ply:noflip" \
   "head closed: magenta filled, not measured=$OUT/seg/head_closed_flags.ply:noflip" --textured --crop 0.45 --title "Steps 6-7 on: $TITLE" > /dev/null
$R "$OUT/step7_geometry.png" "head only, closed=$OUT/seg/head_closed.ply:noflip" "device only=$OUT/seg/device.ply:noflip" --crop 0.45 --title "Steps 6-7 geometry: $TITLE" > /dev/null
cat "$OUT/report.log"; tail -n +1 "$OUT/screw_check.log"; grep -E "face found|renders" "$OUT/facelm.log" | head -3
echo "STEP8 ON MESH DONE $OUT"
