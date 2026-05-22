# Replication Package: Analyzing and Predicting Snapshot Reversions in GitHub Repositories

This replication package contains the datasets and scripts necessary to reproduce the findings of our study on snapshot-related pull requests and their reversions in GitHub repositories. 

Our toolchain supports end-to-end replication, from dataset construction to statistical analysis and machine learning-based prediction.

## 📂 Repository Structure

The package is organized into the following main components:

### 1. Dataset Construction
These scripts are responsible for mining GitHub repositories, extracting snapshot-related pull requests, and building the datasets used for analysis.

*   **`dataset_file.py`**: Extracts both file-level and commit-level features in a single pass. Outputs per-snapshot-file data to `file_data/` and per-PR data to `commit_data/`.

### 2. Statistical Analysis & Visualization (Research Questions 1 & 2)
These scripts process the constructed datasets to answer our first two Research Questions (RQs) and generate corresponding plots.


*   **`RQ1.py`**: Conducts Mann-Whitney U tests and generates split violin plots to answer RQ1, comparing features (e.g., PR description length, number of revisions) between Reverted and non-Reverted PRs.
*   **`RQ2.py`**: Conducts statistical analysis and generates plots to answer RQ2 (File-level revert ratios). Outputs include distribution charts and stacked bar graphs.

### 3. Prediction Models (Research Question 3,4)
These scripts train and evaluate machine learning and Large Language Models (LLMs) to predict snapshot reversions.

*   **`RQ3.py`**: Trains and evaluates traditional machine learning models (e.g., Random Forest) and transformer-based models (e.g., CodeBERT, RoBERTa) to predict reversions. It generates evaluation metrics (Accuracy, ROC-AUC, PR-AUC, MCC, G-Mean) and saves prediction results.
*   **`RQ3_llama.py`**: Fine-tunes and evaluates a Large Language Model (`meta-llama/Meta-Llama-3.1-8B-Instruct`) using LoRA (Low-Rank Adaptation) to predict snapshot reversions based on commit messages and diff texts.

---

## 🚀 Environment Setup

We recommend using `uv` for fast and reliable environment setup. The project configuration is defined in `pyproject.toml`.

### Prerequisites
*   Python >= 3.10
*   `uv` (Fast Python package installer and resolver)

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
2.  Create a virtual environment and install dependencies using uv:
*   uv sync

*Note: For macOS (Apple Silicon) users, the environment will automatically install the standard version of TensorFlow. For Linux users, the GPU-enabled version (`tensorflow[and-cuda]`) will be installed automatically based on the `pyproject.toml` configuration.*

### Hugging Face Setup (Required for Llama & Transformers)
If you intend to run `RQ3.py` or `RQ3_llama.py`, you need to configure your Hugging Face authentication:

1. Request access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face.
2. Log in via the CLI:
   ```bash
   uv run huggingface-cli login

---

## 🖥️ GPU Cluster (SLURM)

`RQ3_llama.py` (Llama 3.1 8B + LoRA fine-tuning) requires a GPU with at least 24 GB VRAM and several hours of compute time.
We ran our experiments on a SLURM-managed HPC cluster. The provided sbatch scripts assume this environment.

### Requirements

- SLURM workload manager
- CUDA-capable GPU (≥ 24 GB VRAM recommended)
- `uv` installed in `$HOME/.local/bin` (run `curl -LsSf https://astral.sh/uv/install.sh | sh` on the cluster)
- Access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face

### Submission

Submit from the **project root**:

```bash
# ML & Transformer models (CodeBERT, RoBERTa, Random Forest)
sbatch snapshot_package/RQ3.sbatch

# Llama 3.1 fine-tuning
sbatch snapshot_package/RQ3_llama.sbatch
```

Logs are written to `logs/` and `errors/` (created automatically).

### Notes

- The `--partition=gpu_long` directive may need to be changed to match your cluster's partition name.
- `module load cuda` is cluster-specific; adjust or remove if CUDA is available without module loading.
- Hugging Face model weights are cached to `/work/$USER/huggingface_cache` to avoid home directory quota issues. Change `HF_HOME` in the sbatch script if your cluster uses a different scratch path.

