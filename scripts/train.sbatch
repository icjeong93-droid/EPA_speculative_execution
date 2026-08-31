#!/bin/bash
#SBATCH --job-name=epa
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
# SLURM wrapper. Adjust partition/account for your cluster.
# Data prep (Silero VAD + resampling) is CPU-bound and wants many cores --
# num_vad_workers / num_resample_workers in the data config should match
# --cpus-per-task.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/launch_train.sh "${1:-table1}"
