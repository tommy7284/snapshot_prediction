#!/bin/bash
#SBATCH --job-name=finetune_llama
#SBATCH --output=logs/llama_f_cmdiff_%j.out   
#SBATCH --error=errors/llama_f_cmdiff_%j.err
#SBATCH --time=100:00:00
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

# Disable Python output buffering to write logs in real-time
export PYTHONUNBUFFERED=1

# 1. Load necessary modules
module load cuda
nvidia-smi

# 2. Configure Hugging Face cache directory (Important!)
# Change the download location for models and datasets to the /work directory
export HF_HOME="/work/$USER/huggingface_cache"
mkdir -p $HF_HOME

# 3. Hugging Face Access Token for Llama 3.1
# Not required if you have already run `poetry run huggingface-cli login`.
export HF_TOKEN=""

# 4. Output execution logs
echo "🚀 Starting training job: $(date)"
echo "Hostname: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Hugging Face Cache Dir: $HF_HOME"

# 5. Execute the Python training script using Poetry
poetry --version
echo "Starting Fine-tuning script..."

# Run the training script (train_llama.py)
poetry run python -u train_llama.py

echo "✅ Training job completed: $(date)"
