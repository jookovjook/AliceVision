#!/usr/bin/env bash
# Step 1: RGB-only reconstruction baseline with the stock Meshroom 2023.3.0 release (AliceVision 3.2).
# No ARKit poses, no depth, no tags. Intrinsics only SEED the solver (from camera_matrix.csv; SfM refines them).
#   bash headscan/steps/step1_rgb_meshroom.sh CAPTURE_DIR FRAMES_DIR OUT_DIR
set -euo pipefail
CAP=$1; FRAMES=$2; OUT=$3
M=/workspace/opt/Meshroom-2023.3.0
export ALICEVISION_ROOT=$M/aliceVision LD_LIBRARY_PATH=$M/aliceVision/lib:$M/lib
mkdir -p "$OUT"

# fx, cx, cy of the rgb video (camera_matrix.csv = last frame; odometry fx varies < 0.2 %)
read FOV OFFX OFFY < <(python3 - "$CAP/camera_matrix.csv" <<'PY'
import sys, math
K = [[float(v) for v in l.split(",")] for l in open(sys.argv[1]) if l.strip()]
w, h = 1920, 1440
print(f"{2*math.degrees(math.atan(w/2/K[0][0])):.4f} {K[0][2]-w/2:.3f} {K[1][2]-h/2:.3f}")
PY
)
echo "seed intrinsics: hfov $FOV deg, principal point offset $OFFX $OFFY px"

$M/aliceVision/bin/aliceVision_cameraInit --imageFolder "$FRAMES" --output "$OUT/cameraInit_seed.sfm" \
  --sensorDatabase $M/aliceVision/share/aliceVision/cameraSensors.db \
  --defaultFieldOfView "$FOV" --defaultOffsetX "$OFFX" --defaultOffsetY "$OFFY" \
  --defaultCameraModel radial3 --allowedCameraModels radial3 --groupCameraFallback global \
  --viewIdMethod filename --viewIdRegex '.*?(\d+)' --verboseLevel warning

cd $M && ./meshroom_batch -i "$OUT/cameraInit_seed.sfm" -p photogrammetry \
  --cache "$OUT/cache" --save "$OUT/step1.mg" -o "$OUT/publish" -v info 2>&1 | tee "$OUT/meshroom_batch.log" | grep -E "^\[|Node|ERROR|error" | grep -vE "trace|debug" | head -200
ls -la "$OUT/publish"
