#!/usr/bin/env bash
# =============================================================================
# Replication Script
# "Leveraging Snapshots in Code Review: An Empirical Study on
#  Snapshot Reversion Prediction"
#
# Usage:
#   export GITHUB_TOKEN="your_token"
#   bash run_replication.sh
#
# Optional (required only for RQ3 LLaMA fine-tuning):
#   export HF_TOKEN="your_huggingface_token"
#
# All output is saved to results/run.log in addition to stdout.
# =============================================================================

set -euo pipefail

RESULTS_DIR="results"
LOG_FILE="${RESULTS_DIR}/run.log"

mkdir -p "${RESULTS_DIR}"

# Redirect all stdout/stderr to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "  $(timestamp)"
    echo "============================================================"
}

# =============================================================================
# 0. Prerequisites check
# =============================================================================
section "Step 0: Prerequisites Check"

if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' is not installed."
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  [OK] uv: $(uv --version)"

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "WARNING: GITHUB_TOKEN is not set."
    echo "  Dataset construction will assume all candidate PRs are merged"
    echo "  and PR descriptions will be empty."
    echo "  Set GITHUB_TOKEN for accurate results:"
    echo "    export GITHUB_TOKEN=\"your_personal_access_token\""
else
    echo "  [OK] GITHUB_TOKEN is set."
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "INFO: HF_TOKEN is not set (only required for Step 4b - LLaMA fine-tuning)."
else
    echo "  [OK] HF_TOKEN is set."
fi

# =============================================================================
# 1. Dataset Construction
#    Outputs:
#      file_data/<repo>_reversions.csv   — one row per .snap file per PR
#      commit_data/<repo>_reversions.csv — one row per PR
# =============================================================================
section "Step 1: Dataset Construction (dataset_file.py)"

uv run python snapshot_package/dataset_file.py

echo ""
echo "  file_data/   : $(ls file_data/*.csv 2>/dev/null | wc -l | tr -d ' ') CSV files"
echo "  commit_data/ : $(ls commit_data/*.csv 2>/dev/null | wc -l | tr -d ' ') CSV files"

# =============================================================================
# 2. RQ1 — Prevalence of Snapshot Updates
#    Outputs:
#      violinplot_01_pr_desc_length.pdf
#      violinplot_02_pr_total_commits.pdf
#      violinplot_03_pr_final_files.pdf
#      violinplot_04_pr_final_lines.pdf
#      results/rq1_stats.txt
# =============================================================================
section "Step 2: RQ1 Analysis (RQ1.py)"

uv run python snapshot_package/RQ1.py

# =============================================================================
# 3. RQ2 — Distribution of Reverted Snapshot Files
#    Outputs:
#      rq2_fig4_revert_ratio_distribution.pdf
#      rq2_fully_vs_partially.pdf
#      results/rq2_stats.txt
# =============================================================================
section "Step 3: RQ2 Analysis (RQ2.py)"

uv run python snapshot_package/RQ2.py

# =============================================================================
# 4a. RQ3 — Reversion Prediction (ML / Transformer models)
#     Outputs: prediction CSVs and evaluation metrics printed to stdout
# =============================================================================
section "Step 4a: RQ3 Analysis — ML & Transformer Models (RQ3.py)"

uv run python snapshot_package/RQ3.py

# =============================================================================
# 4b. RQ3 (LLaMA) — Fine-tuning Llama 3.1 8B with LoRA
#     Requires: GPU, HF_TOKEN, access to meta-llama/Meta-Llama-3.1-8B-Instruct
#     On a SLURM cluster, use: sbatch snapshot_package/RQ3_llama.sbatch
# =============================================================================
if [ -n "${HF_TOKEN:-}" ]; then
    section "Step 4b: RQ3 Analysis — LLaMA Fine-tuning (RQ3_llama.py)"
    uv run python snapshot_package/RQ3_llama.py
else
    echo ""
    echo "  [Skip] Step 4b: HF_TOKEN not set."
    echo "  To run LLaMA fine-tuning:"
    echo "    export HF_TOKEN=\"your_token\""
    echo "    uv run python snapshot_package/RQ3_llama.py"
    echo "  On a SLURM cluster:"
    echo "    sbatch snapshot_package/RQ3_llama.sbatch"
fi

# =============================================================================
# Done
# =============================================================================
section "All Steps Complete"

echo ""
echo "  Output files:"
echo "    Figures  : violinplot_*.pdf, rq2_*.pdf"
echo "    Stats    : ${RESULTS_DIR}/rq1_stats.txt"
echo "               ${RESULTS_DIR}/rq2_stats.txt"
echo "    Full log : ${LOG_FILE}"
echo ""
