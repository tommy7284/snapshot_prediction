# Usage Guide

Follow these steps to construct the dataset, perform statistical analysis, and train the prediction models.
For a fully automated run, see [Full Replication](#full-replication).

## Prerequisites

Set the following environment variables before running any script.

```bash
export GITHUB_TOKEN="your_personal_access_token"   # Required for dataset construction
export HF_TOKEN="your_huggingface_token"            # Required for RQ3_llama.py only
```

## Full Replication

Run all steps in order:

```bash
bash run_replication.sh
```

All output is logged to `results/run.log`.

---

## Step-by-Step

### Step 1: Dataset Construction

Clones repositories, calls the GitHub GraphQL API, and writes per-snapshot-file and per-PR CSVs.

```bash
uv run python snapshot_package/dataset_file.py
```

Outputs:
- `file_data/<owner>_<repo>_reversions.csv` — one row per `.snap` file per PR
- `commit_data/<owner>_<repo>_reversions.csv` — one row per PR

### Step 2: RQ1 — Prevalence of Snapshot Updates

```bash
uv run python snapshot_package/RQ1.py
```

Outputs: `violinplot_01_pr_desc_length.pdf` – `violinplot_04_pr_final_lines.pdf`, `results/rq1_stats.txt`

### Step 3: RQ2 — Distribution of Reverted Snapshot Files

```bash
uv run python snapshot_package/RQ2.py
```

Outputs: `rq2_fig4_revert_ratio_distribution.pdf`, `rq2_fully_vs_partially.pdf`, `results/rq2_stats.txt`

### Step 4a: RQ3 — ML & Transformer Models

```bash
uv run python snapshot_package/RQ3.py
```

Outputs: `output_RQ3/all_models_roc_data.csv`, `output_RQ3/prediction_analysis.csv`, ROC curve and confusion matrix plots.

### Step 4b: RQ3 — LLaMA Fine-tuning (GPU required)

```bash
uv run python snapshot_package/RQ3_llama.py

# On a SLURM cluster:
sbatch snapshot_package/RQ3_llama.sbatch
```

---

## Changing Training Features (RQ3 and RQ3_llama)

Both `RQ3.py` and `RQ3_llama.py` accept a `--features` flag to select which columns from `file_data/`
are concatenated as the training input text.

### Available features

| Feature | Column in CSV | Description |
|---|---|---|
| `pr_description` | `pr_description` | Body text of the pull request |
| `commit_message` | `commit_message` | Commit message of the first snapshot-touching commit |
| `diff_text` | `diff_text` | Source-file diff from the first snapshot-touching commit |

### RQ3.py (CodeBERT / RoBERTa / Random Forest)

Default: `pr_description diff_text`

```bash
# Default (pr_description + diff_text)
uv run python snapshot_package/RQ3.py

# Commit message and diff only
uv run python snapshot_package/RQ3.py --features commit_message diff_text

# All three features
uv run python snapshot_package/RQ3.py --features pr_description commit_message diff_text

# Diff only
uv run python snapshot_package/RQ3.py --features diff_text
```

### RQ3_llama.py (Llama 3.1 8B + LoRA)

Default: `commit_message diff_text`

```bash
# Default (commit_message + diff_text)
uv run python snapshot_package/RQ3_llama.py

# PR description and diff
uv run python snapshot_package/RQ3_llama.py --features pr_description diff_text

# All three features
uv run python snapshot_package/RQ3_llama.py --features pr_description commit_message diff_text
```

Features are inserted into the training prompt as labeled sections in the order specified.
