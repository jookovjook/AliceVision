#!/usr/bin/env bash
# Step 5: mesh the LiDAR-twin SfM (lidar_fuse_av.py output) with aliceVision_meshing, then texture the result with the
# full-resolution MVS images (visibilities remapped from the MVS dense point cloud).
#   bash headscan/steps/step5_fuse_mesh.sh MVS_DENSE_DIR FUSE_DIR
set -euo pipefail
DENSE=$1; F=$2
. /opt/av/env.sh
LOG="$F/cmd.log"; mkdir -p "$F/mesh" "$F/meshF" "$F/texturing"
run() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { tail -40 "$LOG"; echo "FAILED: $1"; exit 1; }; echo "  $(basename "$1") ok"; }
BBOX=$(python3 -c "import json; b=json.load(open('$DENSE/head_box.json')); h=b['half']; print(','.join(f'{v:.6f}' for v in [*b['centre'],0,0,0,h,h,h]))")
run aliceVision_meshing --input "$F/sfm.sfm" --depthMapsFolder "$F/depth" --outputMesh "$F/mesh/mesh.obj" \
    --output "$F/mesh/densePointCloud.abc" --boundingBox "$BBOX" --maxPoints 5000000 --minVis 2 --colorizeOutput 1 --verboseLevel info
run aliceVision_meshFiltering --inputMesh "$F/mesh/mesh.obj" --outputMesh "$F/meshF/mesh.obj" --keepLargestMeshOnly 1 \
    --smoothingIterations 3 --verboseLevel info
run aliceVision_texturing --input "$DENSE/mesh/densePointCloud.abc" --imagesFolder "$DENSE/images" --inputMesh "$F/meshF/mesh.obj" \
    --output "$F/texturing" --outputMeshFileType obj --colorMappingFileType png --textureSide 4096 --unwrapMethod Basic --verboseLevel info
echo "FUSE MESH DONE $F"
