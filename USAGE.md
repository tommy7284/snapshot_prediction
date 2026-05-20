**`USAGE.md`** 
```markdown
# 🏃‍♂️ Usage Guide

Follow these steps to construct the dataset, perform statistical analysis, and train the prediction models.

## Step 0: Prerequisites

Ensure your environment variables are correctly set before running the scripts.

1. **GitHub API Token**: Required for data collection.

   ```bash
   export GITHUB_TOKEN="your_personal_access_token"

2.  Hugging Face Token (Optional): Required for RQ3_llama.py
    ```bash
    export HF_TOKEN="your_huggingface_token"
    # Or run: uv run huggingface-cli login

Step 1: Dataset Construction
Extract snapshot-related pull requests and commit histories.

    ```bash
    uv run python dataset_commit.py
    uv run python dataset_file.py

Step 2: Statistical Analysis & Visualization (RQ1 & RQ2)
Generate distribution plots and run statistical tests.

    ```bash
    uv run python RQ1.py
    uv run python RQ2.py

    ```bash
    # Run Baseline and Transformer models (CodeBERT, RoBERTa)
    uv run python RQ3.py

    # Fine-tune the Llama 3.1 model (Use the SLURM script if on a cluster)
    uv run python rq3_llama.py
