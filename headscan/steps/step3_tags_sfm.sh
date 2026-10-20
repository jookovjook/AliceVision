#!/usr/bin/env bash
# Step 3: AprilTag 36h11 corners as SfM landmarks.
# Tags are matched over ALL image pairs (one-hot descriptors = exact id matching), filtered with the refined poses of
# step 2B (--matchFromKnownCameraPoses), so each tag corner becomes one long track instead of fragments that would
# duplicate the marker id. SIFT tracks of step 2 + tag tracks -> sfmExpanding from the step-2B poses (unlocked).
#   bash headscan/steps/step3_tags_sfm.sh STEP2_DIR OUT_DIR
set -euo pipefail
S2=$1; OUT=$2
. /opt/av/env.sh
mkdir -p "$OUT"/feat_tags "$OUT"/match_tags "$OUT"/sfm
LOG="$OUT/cmd.log"
run() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { tail -40 "$LOG"; echo "FAILED: $1"; exit 1; }; echo "  $(basename "$1") ok"; }
POSED="$S2/04_sfm/sfm_col.sfm"          # step-2B views + refined poses + intrinsics (+ SIFT landmarks)
INIT="$S2/04_sfm/cameras.sfm"           # same views/poses without landmarks
run aliceVision_featureExtraction --input "$S2/00_init/cameraInit.sfm" --output "$OUT/feat_tags" --describerTypes tag36h11 \
    --forceCpuExtraction 1 --verboseLevel info
run aliceVision_imageMatching --input "$S2/00_init/cameraInit.sfm" --featuresFolders "$OUT/feat_tags" --method Exhaustive \
    --output "$OUT/pairs_all.txt" --verboseLevel info
run aliceVision_featureMatching --input "$POSED" --featuresFolders "$OUT/feat_tags" --imagePairsList "$OUT/pairs_all.txt" \
    --describerTypes tag36h11 --matchFromKnownCameraPoses 1 --knownPosesGeometricErrorMax 8 --output "$OUT/match_tags" --verboseLevel info
run aliceVision_tracksBuilding --input "$S2/00_init/cameraInit.sfm" --featuresFolders "$S2/01_features" "$OUT/feat_tags" \
    --matchesFolders "$S2/02_matching" "$OUT/match_tags" --describerTypes dspsift,tag36h11 --output "$OUT/tracks.json" --verboseLevel info
run aliceVision_sfmExpanding --input "$INIT" --tracksFilename "$OUT/tracks.json" --output "$OUT/sfm/sfm.sfm" \
    --outputViewsAndPoses "$OUT/sfm/cameras.sfm" --lockScenePreviouslyReconstructed 0 --minAngleForTriangulation 2.0 --verboseLevel info
echo "STEP3 DONE"
