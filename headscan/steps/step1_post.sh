#!/usr/bin/env bash
# Step 1 post-processing: SfM -> JSON, pictures, stats, upload to Drive.
#   bash headscan/steps/step1_post.sh OUT_DIR [DRIVE_DEST]
set -euo pipefail
OUT=$1; DEST=${2:-gdrive:frame_capture/6ba4b7d84f_recon/step1_rgb_meshroom}
M=/workspace/opt/Meshroom-2023.3.0
export ALICEVISION_ROOT=$M/aliceVision LD_LIBRARY_PATH=$M/aliceVision/lib:$M/lib
PY="/workspace/hsvenv/bin/python -I"
HS=/workspace/AliceVision/headscan
SFMDIR=$(dirname "$(ls "$OUT"/cache/StructureFromMotion/*/sfm.abc | head -1)")
$M/aliceVision/bin/aliceVision_convertSfMFormat --input "$SFMDIR/sfm.abc" --output "$OUT/sfm.json" \
  --describerTypes dspsift --views 1 --intrinsics 1 --extrinsics 1 --structure 1 --observations 0 --verboseLevel warning
$PY $HS/viz/render.py sfm "$OUT/sfm.json" "$OUT/step1_sfm.png" --title "Step 1 RGB-only SfM"
MESH=$(ls "$OUT"/publish/*.obj | head -1)
$PY $HS/viz/render.py mesh "$OUT/sfm.json" "$MESH" "$OUT/step1_mesh_head.png" --crop 0.55 --title "Step 1 textured mesh, head region"
$PY $HS/viz/render.py mesh "$OUT/sfm.json" "$MESH" "$OUT/step1_mesh_scene.png" --view-dist 2.2 --fov 60 --untextured --title "Step 1 mesh, whole scene"
$PY - "$OUT" "$SFMDIR" <<'PY'
import json, sys, re, glob
out, sfmdir = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/sfm.json"))
i = d["intrinsics"][0]
W = float(i["width"]); f_px = float(i["focalLength"]) * max(W, float(i["height"])) / float(i["sensorWidth"])
log = open(glob.glob(f"{sfmdir}/*log")[0]).read() if glob.glob(f"{sfmdir}/*log") else ""
rmse = re.findall(r"RMSE[^\d]*([\d.]+)", log)
s = {"views": len(d["views"]), "posed": len(d.get("poses", [])), "landmarks": len(d.get("structure", [])),
     "focal_px": round(f_px, 2), "principal_point_offset_px": [round(float(v), 2) for v in i["principalPoint"]],
     "distortion": [round(float(v), 5) for v in i.get("distortionParams", [])], "type": i["type"],
     "sfm_log_rmse_last": rmse[-1] if rmse else None}
json.dump(s, open(f"{out}/step1_stats.json", "w"), indent=1); print(json.dumps(s))
PY
rclone copy "$OUT" "$DEST" --exclude "cache/**" --transfers 8 2>&1 | grep -v NOTICE || true
rclone copy "$SFMDIR" "$DEST/sfm" 2>&1 | grep -v NOTICE || true
echo "uploaded to $DEST"
