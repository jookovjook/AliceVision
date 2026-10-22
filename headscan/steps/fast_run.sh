#!/usr/bin/env bash
# Fast pipeline on the head capture + the 3 table calibration captures: CPU stages of all captures in parallel, then
# Stochastic Solids (7k) one capture at a time, then compare + fits + upload. Reference meshes (the full 30k runs) are
# fetched from Drive for the comparison.
#   bash headscan/steps/fast_run.sh
set -uo pipefail
HS=/workspace/AliceVision/headscan; L=/workspace/logs; mkdir -p $L /workspace/ref
HEAD_RULER=3.5,4,4,1.5,19.5,34,52.5,59,26,9.5,8,17.5
rclone copyto gdrive:frame_capture/6ba4b7d84f_recon/gs/stochastic_solids/head/tsdf.ply /workspace/ref/6ba4b7d84f_tsdf.ply &
for c in 1 2 3; do rclone copyto gdrive:frame_capture/calibration_run/calibration_$c/gs/ss/head/tsdf.ply /workspace/ref/calibration_${c}_tsdf.ply & done
env_of() {   # per-capture settings
  if [ "$1" = 6ba4b7d84f ]; then echo "SCALE=1 CENTER='0.02 0.04 0.04' HALF=0.22 RULER=$HEAD_RULER REF=/workspace/ref/6ba4b7d84f_tsdf.ply"
  else echo "RULER= SCALE=3 CENTER='0 0 0.1' HALF=0.26 REF=/workspace/ref/${1}_tsdf.ply"; fi
}
CAPS="6ba4b7d84f calibration_1 calibration_2 calibration_3"
for c in $CAPS; do
  (eval "$(env_of $c)"; export SCALE CENTER HALF REF RULER; AV_CORES=14 OMP_NUM_THREADS=14 \
    bash "$HS/steps/fast_capture.sh" $c frames,init,tags,poses,export > $L/$c.fast_cpu.log 2>&1; echo "$c cpu exit $?") &
done
wait
for c in $CAPS; do
  (eval "$(env_of $c)"; export SCALE CENTER HALF REF RULER; bash "$HS/steps/fast_capture.sh" $c ss,compare,fit,upload > $L/$c.fast_gpu.log 2>&1; echo "$c gpu exit $?")
done
echo "FAST RUN DONE $(date +%H:%M:%S)"
