#!/usr/bin/env bash
# Fast pipeline on head captures recorded with the headscan iOS app (Stray-format folders, already in /workspace/data/CAP):
# CPU stages of all captures in parallel, then Stochastic Solids (7k) + fits one capture at a time, then essentials to Drive.
# Blind: no ruler / caliper values go in (compared on the Mac afterwards); no reference mesh (no compare stage).
#   bash headscan/steps/app_run.sh CAP [CAP ...]          e.g. jack_1 jack_2 jack_3 yoni_1 yoni_2 yoni_3
set -uo pipefail
HS=/workspace/AliceVision/headscan; L=/workspace/logs; mkdir -p $L
export DRIVE_ROOT=${DRIVE_ROOT:-gdrive:frame_capture/app_scans_oct23}
N=$#; CORES=$(( $(nproc) < 64 ? $(nproc) : 64 )); PER=$(( CORES / N > 4 ? CORES / N : 4 ))
for c in "$@"; do
  (SCALE=1 AV_CORES=$PER OMP_NUM_THREADS=$PER bash "$HS/steps/fast_capture.sh" $c frames,init,tags,poses,export > $L/$c.app_cpu.log 2>&1; echo "$c cpu exit $?") &
done
wait
until [ -f /workspace/logs/ss_ready ]; do sleep 20; done
for c in "$@"; do
  (SCALE=1 bash "$HS/steps/fast_capture.sh" $c ss,fit,upload > $L/$c.app_gpu.log 2>&1; echo "$c gpu exit $?")
done
echo "APP RUN DONE $(date +%H:%M:%S)"
