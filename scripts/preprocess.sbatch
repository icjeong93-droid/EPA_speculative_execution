#!/bin/bash
#SBATCH --job-name=epa-prep
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep_%j.out
# CPU-only. Do NOT request a GPU here -- this stage never touches one.
#
#   sbatch scripts/preprocess.sbatch          # train + val  (before training)
#   sbatch scripts/preprocess.sbatch test     # test split   (before evaluation)
#
# Run the first form to completion BEFORE sbatch scripts/train.sbatch, and the
# second BEFORE sbatch scripts/evaluate.sbatch -- otherwise each stage begins by
# resampling audio while holding four H100s idle.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
MODES=""
[ $# -gt 0 ] && MODES="--modes $*"
EPA_NUM_WORKERS=0 "$VENV/bin/python" scripts/preprocess_only.py \
  --config configs/forecasting/mimi/fc640_transformer_mimi_12.5hz_loss1-01_m3.yaml \
  $MODES
