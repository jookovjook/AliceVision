#!/usr/bin/env bash
# Fresh vast.ai box (vastai/base-image:cuda-12.4.1-cudnn8-devel-ubuntu22.04) -> ready for the headscan pipeline:
# toolchain + headscan venv, our AliceVision build restored from Drive (no rebuild), captures unpacked,
# Stochastic Solids env. Needs ~/.config/rclone/rclone.conf (remote gdrive:) and this repo in /workspace/AliceVision.
#   bash headscan/box/restore_box.sh CAPTURE [CAPTURE ...]     e.g. calibration_1 calibration_2 calibration_3
set -euo pipefail
cd /workspace
HS=/workspace/AliceVision/headscan
BUILD=gdrive:frame_capture/6ba4b7d84f_recon/build/av_build_u2204_cuda124_feature-headscan-arc.tar.zst
mkdir -p dl data logs

rclone copy "$BUILD" dl/ & DL=$!
for c in "$@"; do rclone copy "gdrive:frame_capture/$c.zip" dl/ & DL="$DL $!"; done

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"   # uv (installed by setup_toolchain.sh)
/workspace/hsvenv/bin/python -c "import cv2, open3d, pupil_apriltags" 2>/dev/null || bash "$HS/box/setup_toolchain.sh" > logs/toolchain.log 2>&1
uv pip install --python /workspace/hsvenv/bin/python -q scikit-image rtree fast_simplification plyfile
# Stochastic Solids env in the background (CUDA extensions take ~15 min); apt is free again now
bash "$HS/gs/run_stochastic_solids.sh" setup > logs/ss_setup.log 2>&1 &
SS=$!
wait $DL

mkdir -p /opt/av && zstd -dc dl/$(basename "$BUILD") | tar -C /opt/av -xf -
(. /opt/av/env.sh && aliceVision_featureExtraction 2>&1 | grep -q "AliceVision featureExtraction") && echo "AliceVision restored"
for c in "$@"; do
  # every zip holds one capture folder (random name) -> data/CAPTURE
  rm -rf "data/.tmp_$c" && mkdir -p "data/.tmp_$c" && unzip -q "dl/$c.zip" -d "data/.tmp_$c"
  rm -rf "data/$c" && mv "data/.tmp_$c"/*/ "data/$c" && rmdir "data/.tmp_$c"
  echo "$c: $(ls data/$c | tr '\n' ' ')"
done
wait $SS && touch logs/ss_ready && echo "Stochastic Solids env ready"
