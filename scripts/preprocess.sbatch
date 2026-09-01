#!/bin/bash
#SBATCH --job-name=epa-prep
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep_%j.out
# CPU-only. Do NOT request a GPU here -- this stage never touches one.
# Run this to completion BEFORE sbatch scripts/train.sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
EPA_NUM_WORKERS=0 "$VENV/bin/python" scripts/preprocess_only.py \
  --config configs/forecasting/mimi/fc640_transformer_mimi_12.5hz_loss1-01_m3.yaml
