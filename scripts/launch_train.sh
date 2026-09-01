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
#   FORCE=1 bash scripts/launch_train.sh fc640   # retrain a run that already exists
#
# Re-submitting is safe: a run whose best_val_acc.pt already exists is SKIPPED and
# named in the summary, so a wave that lost one job can be resumed without
# retraining the rest and without silently overwriting good checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/EndpointAnticipation/anticipation-model"
CFG_DIR="${CFG_DIR:-$ROOT/configs/forecasting/mimi}"   # configs_smoke/... to rehearse
LOG_DIR="$ROOT/logs/train"
VENV="${VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY="$VENV/Scripts/python.exe"
NGPU="${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
# A missing/failing nvidia-smi yields 0, and $(( i % 0 )) is a bash division-by-zero
# that kills the script under `set -e` with no useful message. Floor it, as
# evaluate.sh already does.
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
SUFFIX="_transformer_mimi_12.5hz_loss1-01_m3.yaml"
FORCE="${FORCE:-0}"

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

# cfg_field <config> <key> -- first value of a top-level-ish `key: value` line
cfg_field() { grep -m1 "^[[:space:]]*$2:" "$1" | sed "s/.*$2:[[:space:]]*//; s/[[:space:]]*$//"; }

# run_dir <config> -- where upstream will write this run's checkpoints.
# src/utils/common.py builds run_name as <data-config stem>__<model-config stem>
# and setup_save_folder joins it under run_params.save_folder.
run_dir() {
  local cfg="$1" save data stem
  save=$(cfg_field "$cfg" save_folder)
  data=$(basename "$(cfg_field "$cfg" data_config)" .yaml)
  stem=$(basename "$cfg" .yaml)
  echo "$save/${data}__${stem}"
}

# Preprocessing must be COMPLETE before any GPU job starts, or run.py begins by
# resampling audio while holding the allocation. evaluate.sh guards its side; this
# is the more expensive one (a 48 h, 4-GPU reservation). The dump path comes from
# the run's own data config, so a configs_smoke rehearsal checks dump_smoke.
if [ "${SKIP_PREP_CHECK:-0}" != "1" ]; then
  first_cfg="$CFG_DIR/${RUNS[0]}${SUFFIX}"
  if [ -f "$first_cfg" ]; then
    data_cfg=$(cfg_field "$first_cfg" data_config)
    if [ -f "$data_cfg" ]; then
      dump=$(cfg_field "$data_cfg" dump)
      if [ -n "$dump" ] && ! "$PY" "$ROOT/scripts/check_preprocessed.py" --dump "$dump" --modes train val; then
        echo
        echo "   Refusing to start: training would resample audio while holding the GPUs."
        echo "   Run the CPU job first:  sbatch scripts/preprocess.sbatch"
        echo "   (SKIP_PREP_CHECK=1 overrides, but you almost certainly do not want that.)"
        exit 1
      fi
      echo
    fi
  fi
fi

mkdir -p "$LOG_DIR"
echo "root   : $ROOT"
echo "gpus   : $NGPU"
echo "runs   : ${RUNS[*]}  (${#RUNS[@]} jobs)"
echo "logs   : $LOG_DIR"
[ "$FORCE" = "1" ] && echo "force  : on (existing checkpoints will be retrained over)"
echo

# Parallel indexed arrays rather than an associative one -- no bash 4.3 dependency.
wave_pid=(); wave_run=()
failed=(); skipped=(); launched=()

# Wait for the current wave and record which jobs died. `wait` with no arguments
# ALWAYS returns 0, so the previous version reported "all runs finished" even when
# every job had crashed -- and evaluate.sh would then happily build Table 1 from a
# stale checkpoint. Wait per-PID instead.
drain() {
  [ "${#wave_pid[@]}" -eq 0 ] && return 0
  local k rc
  for k in "${!wave_pid[@]}"; do
    rc=0; wait "${wave_pid[$k]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  !! ${wave_run[$k]} FAILED (exit $rc) -- see $LOG_DIR/${wave_run[$k]}.log"
      failed+=("${wave_run[$k]}")
    fi
  done
  wave_pid=(); wave_run=()
}

i=0
for run in "${RUNS[@]}"; do
  cfg="$CFG_DIR/${run}${SUFFIX}"
  [ -f "$cfg" ] || { echo "!! missing config: $cfg"; exit 1; }

  # overwrite_prev_run: true only means "do not exit"; upstream never clears the
  # folder, so a re-run keeps the old best_val_acc.pt until some epoch beats it.
  # That is how a half-trained checkpoint reaches Table 1. Refuse by default.
  rd="$(run_dir "$cfg")"
  if [ "$FORCE" != "1" ] && [ -f "$rd/best_val_acc.pt" ]; then
    echo "[skip] $run -- already trained ($rd/best_val_acc.pt)"
    skipped+=("$run")
    continue
  fi

  gpu=$(( i % NGPU ))
  log="$LOG_DIR/${run}.log"
  echo "[gpu $gpu] $run -> $log"
  ( cd "$MODEL_DIR" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --config "$cfg" >"$log" 2>&1 ) &
  wave_pid+=("$!"); wave_run+=("$run")
  launched+=("$run")
  i=$(( i + 1 ))
  # fill all GPUs, then wait for the wave to drain before starting the next
  if (( i % NGPU == 0 )); then
    echo "  -- wave full ($NGPU jobs), waiting --"
    drain
  fi
done
drain

# ---- summary -------------------------------------------------------------
# train.json gets one object per completed epoch, rewritten every epoch, so its
# length says how far a run actually got. Early stopping calls exit() BEFORE the
# final write, so a run that stopped at patience shows one epoch fewer -- that is
# expected, a run that shows 1-2 epochs is not.
echo
echo "== summary =="
for run in "${RUNS[@]}"; do
  cfg="$CFG_DIR/${run}${SUFFIX}"
  rd="$(run_dir "$cfg")"
  ep="-"
  if [ -f "$rd/train.json" ]; then ep=$(grep -c '"epoch"' "$rd/train.json" 2>/dev/null) || ep="0"; fi
  ck="missing"; [ -f "$rd/best_val_acc.pt" ] && ck="ok"
  state="done"
  case " ${failed[*]-} "  in *" $run "*) state="FAILED" ;; esac
  case " ${skipped[*]-} " in *" $run "*) state="skipped" ;; esac
  printf '  %-8s %-8s epochs=%-4s checkpoint=%s\n' "$run" "$state" "$ep" "$ck"
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "!! ${#failed[@]} run(s) FAILED: ${failed[*]}"
  echo "   Do NOT evaluate yet -- a failed run can still leave an older best_val_acc.pt"
  echo "   behind, and report_table1.py cannot tell that apart from a finished run."
  echo "   Inspect: tail -50 $LOG_DIR/<run>.log"
  exit 1
fi

echo
echo "all runs finished. checkpoints under $ROOT/checkpoints"
echo "next: sbatch scripts/preprocess.sbatch test   (if not done)"
echo "      sbatch scripts/evaluate.sbatch"
