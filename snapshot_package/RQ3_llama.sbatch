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

# Pythonの出力をバッファリングせず、リアルタイムでログファイルに書き出す
export PYTHONUNBUFFERED=1

# 1. 必要なモジュールのロード
module load cuda
nvidia-smi
# 2. Hugging Faceのキャッシュディレクトリ設定（重要！）
# モデルやデータセットのダウンロード先を容量の大きい /work 配下に変更します
export HF_HOME="/work/$USER/huggingface_cache"
mkdir -p $HF_HOME

# 3. Llama 3.1用のHugging Faceアクセストークン
# ※事前にターミナルで `poetry run huggingface-cli login` を済ませている場合は不要です。
# まだの場合は、以下のコメントアウトを外してご自身のトークン（Read権限）を貼り付けてください。
export HF_TOKEN=""

# 4. 実行ログの出力
echo "🚀 学習ジョブを開始します: $(date)"
echo "ホスト名: $(hostname)"
echo "割り当てられたGPU: $CUDA_VISIBLE_DEVICES"
echo "Hugging Face Cache Dir: $HF_HOME"

# 5. Poetryを使ってPythonの学習スクリプトを実行
poetry --version
echo "Starting Fine-tuning script..."

# 前回作成した学習スクリプト（train.py）を実行
poetry run python -u train_llama.py

echo "✅ 学習ジョブが完了しました: $(date)"
