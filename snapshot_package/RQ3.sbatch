#!/bin/bash
#SBATCH --job-name=All_msg
#SBATCH --output=logs/All_diffonly%A_%a.out
#SBATCH --error=errors/All_msg%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=100:00:00
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
export PYTHONUNBUFFERED=1
module load cuda
nvidia-smi
#bash 1_clean_logs.sh
#cd snapshot_prediction
#pyenv activate py3
poetry --version
poetry run python -u predict_all.py
