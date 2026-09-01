#!/bin/bash
#SBATCH --job-name=epa-eval
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_eval_%j.out
# Run AFTER training. Produces MRA/HEA/PAR/ERC and prints the paper table.
# Requires the test split to be preprocessed first:
#   sbatch scripts/preprocess.sbatch test
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
bash scripts/evaluate.sh "$@"
