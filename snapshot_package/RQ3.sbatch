#!/bin/bash
#SBATCH --job-name=rq3_prediction
#SBATCH --output=logs/rq3_%j.out
#SBATCH --error=errors/rq3_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

export PYTHONUNBUFFERED=1

mkdir -p logs errors

# Load CUDA (module name may differ by cluster)
module load cuda
nvidia-smi

# Ensure uv is in PATH (installed via https://astral.sh/uv)
export PATH="$HOME/.local/bin:$PATH"

echo "🚀 Starting RQ3 prediction: $(date)"
echo "   Hostname : $(hostname)"
echo "   GPU      : $CUDA_VISIBLE_DEVICES"
uv --version

# Run from project root (the directory where sbatch was submitted)
uv run python -u snapshot_package/RQ3.py

echo "✅ RQ3 completed: $(date)"
