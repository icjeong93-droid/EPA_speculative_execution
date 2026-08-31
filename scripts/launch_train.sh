#!/usr/bin/env bash
# Distribute EPA training runs across GPUs, one run per GPU.
#
# The upstream trainer has NO DDP and no AMP -- it is single-GPU fp32
# (src/training/*.py: just model.to(device)). That is fine here, because the
# work is embarrassingly parallel by construction: EPA-S trains one model per
# forecast horizon, so N horizons = N independent single-GPU jobs. Do NOT add
# DDP; just fill the GPUs.
#
#   bash scripts/launch_train.sh table1          # 3 runs: what Table 1 needs
#   bash scripts/launch_train.sh fig2            # 5 runs: + Figure 2 horizons
#   bash scripts/launch_train.sh all             # 8 runs: every horizon + EPA-M
#   NGPU=4 bash scripts/launch_train.sh table1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/EndpointAnticipation/anticipation-model"
CFG_DIR="$ROOT/configs/forecasting/mimi"
LOG_DIR="$ROOT/logs/train"
VENV="${VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY="$VENV/Scripts/python.exe"
NGPU="${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
SUFFIX="_transformer_mimi_12.5hz_loss1-01_m3.yaml"

case "${1:-table1}" in
  # Table 1 reports EPA-S and EPA-M at h=640 and h=1280. fcall IS EPA-M (it
  # carries all horizons in one model), so it covers both EPA-M rows.
  table1) RUNS=(fc640 fc1280 fcall) ;;
  # Figure 2 sweeps h in {960, 2560}. fc960 also matches the released
  # checkpoint's horizon, so it is the one run we can sanity-check externally.
  fig2)   RUNS=(fc640 fc1280 fcall fc960 fc2560) ;;
  all)    RUNS=(fc320 fc640 fc960 fc1280 fc1600 fc1920 fc2240 fc2560 fcall) ;;
  *)      RUNS=("$@") ;;
esac

mkdir -p "$LOG_DIR"
echo "root   : $ROOT"
echo "gpus   : $NGPU"
echo "runs   : ${RUNS[*]}  (${#RUNS[@]} jobs)"
echo "logs   : $LOG_DIR"
echo

i=0
for run in "${RUNS[@]}"; do
  cfg="$CFG_DIR/${run}${SUFFIX}"
  [ -f "$cfg" ] || { echo "!! missing config: $cfg"; exit 1; }
  gpu=$(( i % NGPU ))
  log="$LOG_DIR/${run}.log"
  echo "[gpu $gpu] $run -> $log"
  ( cd "$MODEL_DIR" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --config "$cfg" >"$log" 2>&1 ) &
  i=$(( i + 1 ))
  # fill all GPUs, then wait for the wave to drain before starting the next
  if (( i % NGPU == 0 )); then
    echo "  -- wave full ($NGPU jobs), waiting --"
    wait
  fi
done
wait
echo
echo "all runs finished. checkpoints under $ROOT/checkpoints"
