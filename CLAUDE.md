# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Uses `uv` for dependency management. Python >= 3.10 required.

```bash
uv sync
```

On macOS, TensorFlow CPU is installed; on Linux, the CUDA-enabled version is installed automatically per `pyproject.toml`.

## Environment Variables

```bash
export GITHUB_TOKEN="..."   # Required for GitHub API calls in dataset construction
export HF_TOKEN="..."       # Required for RQ3_llama.py (Llama 3.1 fine-tuning)
```

## Running Scripts

All scripts must be run from the project root:

```bash
# Dataset construction (outputs both file_data/ and commit_data/)
uv run python snapshot_package/dataset_file.py

# Statistical analysis
uv run python snapshot_package/RQ1.py
uv run python snapshot_package/RQ2.py

# ML prediction (Random Forest, CodeBERT, RoBERTa)
uv run python snapshot_package/RQ3.py

# LLM fine-tuning (Llama 3.1 — GPU cluster recommended)
uv run python snapshot_package/RQ3_llama.py
# SLURM: snapshot_package/RQ3.sbatch / RQ3_llama.sbatch
```

## Architecture

### Data Flow

```
jestable/*.json
    └─► dataset_file.py    ──► file_data/<owner>_<repo>_reversions.csv
    └─► dataset_commit.py  ──► commit_data/<owner>_<repo>_reversions.csv

file_data/*.csv
    └─► RQ1.py   → violin plots (PDF), Mann-Whitney U test results
    └─► RQ2.py   → file-level revert ratio plots
    └─► RQ3.py   → ML model metrics + prediction CSVs
    └─► RQ3_llama.py → LLM fine-tune + evaluation
```

### Key Modules

**`revert_Logic.py`** — Core git analysis library. The two critical functions:
- `check_reversion(repo, path, base_commit, head_commit, pr_commits)` — returns `True` if the `.snap` file ends up identical to the base (OID match) AND had at least one user-initiated change during the PR.
- `is_user_change(repo, commit, path)` — distinguishes intentional edits from automatic merge syncs by comparing content against both parents of a merge commit.

**`dataset_file.py`** — Main pipeline entry point. For each repo in `jestable/`:
1. Clones/fetches the repo into `repos/` via pygit2, including `refs/pull/*/head` refspecs.
2. Finds snapshot-relevant merged PRs using GitHub GraphQL API (batched in chunks of 50).
3. For each PR, iterates commits **once** to collect both per-`.snap`-file and per-PR state simultaneously.
4. Writes file-level results to `file_data/<owner>_<repo>_reversions.csv` and commit-level results to `commit_data/<owner>_<repo>_reversions.csv`.

### CSV Schema (file_data)

Each row is one `.snap` file × one PR:

| Column | Description |
|---|---|
| `pr_number` | GitHub PR number |
| `snap_path` | Path to `.snap` file |
| `source_path` | Corresponding source file (derived from `__snapshots__/` convention) |
| `pr_description` | PR body text |
| `commit_message` | First commit message touching this snapshot |
| `diff_text` | Source file diff from that commit |
| `reverted` | `1` if snapshot was reverted, `0` otherwise |
| `revert_type` | `"None=None"` (created+deleted) or `"OID=OID"` (updated+restored) |
| `first_commit_files` | Files changed in the first snapshot-touching commit |
| `first_commit_lines` | Lines changed in that commit |

### Reversion Detection Logic

A snapshot file is marked as **reverted** when:
1. `base_oid == head_oid` — the file's blob OID at PR head equals its blob OID at merge base.
2. At least one commit in the PR was a genuine user change to that file (not an automatic merge sync where `current_oid == parent2_oid`).

PRs containing any merge commits (2+ parents) are excluded from the dataset entirely.
