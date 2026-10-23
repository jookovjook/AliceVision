#!/usr/bin/env bash
# Fast pipeline on app captures as they arrive (slow uplink): a capture is taken as soon as /workspace/data/CAP/.ready exists (any
# order); CPU stages one capture at a time with all cores, GPU stages (Stochastic Solids, fits, essentials to Drive) queued behind one
# lock, so the next capture's CPU stages overlap the current capture's GPU stages. Blind: no ruler, no reference mesh.
#   bash headscan/steps/app_stream.sh CAP [CAP ...]
set -uo pipefail
HS=/workspace/AliceVision/headscan; L=/workspace/logs; mkdir -p $L
export DRIVE_ROOT=${DRIVE_ROOT:-gdrive:frame_capture/app_scans_oct23}
CORES=$(( $(nproc) < 48 ? $(nproc) : 48 ))
until [ -f /workspace/logs/ss_ready ]; do sleep 20; done
todo=" $* "
while [ -n "${todo// /}" ]; do
  for c in $todo; do
    [ -f /workspace/data/$c/.ready ] || continue
    todo=${todo/ $c / }; echo "$c ready $(date +%H:%M:%S)"
    SCALE=1 AV_CORES=$CORES OMP_NUM_THREADS=$CORES bash "$HS/steps/fast_capture.sh" $c frames,init,tags,poses,export > $L/$c.app_cpu.log 2>&1; echo "$c cpu exit $? $(date +%H:%M:%S)"
    ( flock 8; SCALE=1 bash "$HS/steps/fast_capture.sh" $c ss,fit,upload > $L/$c.app_gpu.log 2>&1; echo "$c gpu exit $? $(date +%H:%M:%S)" ) 8>/workspace/gpu.lock &
  done
  sleep 10
done
wait
echo "APP RUN DONE $(date +%H:%M:%S)"
