#!/usr/bin/env bash
# All frame-on-the-table captures: CPU stages (keyframes .. export) one capture after the other with every core, and
# Stochastic Solids on the GPU for each capture as soon as its export exists (overlaps the next capture's SfM).
#   bash headscan/steps/calib_run.sh        env: CAPS (default calibration_1 calibration_2 calibration_3)
set -uo pipefail
HS=/workspace/AliceVision/headscan
L=/workspace/logs
CAPS=${CAPS:-"calibration_1 calibration_2 calibration_3"}
(
  for c in $CAPS; do
    bash "$HS/steps/calib_capture.sh" "$c" frames,sfm,tags,device,export > "$L/$c.cpu.log" 2>&1 || echo "$c CPU stages FAILED"
    touch "$L/$c.cpu.done"
  done
) &
# Stochastic Solids env (restore_box.sh): fused_ssim is the last CUDA extension it installs
until [ -f "$L/ss_ready" ] || /workspace/miniforge/envs/gggss/bin/python -c "import fused_ssim" 2>/dev/null; do sleep 30; done
for c in $CAPS; do
  until [ -f "$L/$c.cpu.done" ]; do sleep 30; done
  if [ -f "/workspace/work/$c/gs/dataset/export_check.json" ]; then
    bash "$HS/steps/calib_capture.sh" "$c" ss_train,ss_extract,fit,upload > "$L/$c.gpu.log" 2>&1 || echo "$c GPU stages FAILED"
  else
    echo "$c: no export, Stochastic Solids skipped"
  fi
done
wait
echo "CALIB RUN DONE $(date +%H:%M:%S)"
