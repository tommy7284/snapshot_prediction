#!/bin/bash
#SBATCH --job-name=rq3_llama
#SBATCH --output=logs/rq3_llama_%j.out
#SBATCH --error=errors/rq3_llama_%j.err
#SBATCH --time=100:00:00
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

# Hugging Face cache — point to /work to avoid home quota issues
export HF_HOME="/work/$USER/huggingface_cache"
mkdir -p "$HF_HOME"

# Set HF token if not already logged in via `huggingface-cli login`
# export HF_TOKEN="your_token_here"

echo "🚀 Starting LLaMA fine-tuning: $(date)"
echo "   Hostname : $(hostname)"
echo "   GPU      : $CUDA_VISIBLE_DEVICES"
echo "   HF_HOME  : $HF_HOME"
uv --version

# Run from project root (the directory where sbatch was submitted)
uv run python -u snapshot_package/RQ3_llama.py

echo "✅ LLaMA fine-tuning completed: $(date)"
