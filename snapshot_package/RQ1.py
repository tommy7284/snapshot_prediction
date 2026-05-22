import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- Configuration ---
# RQ1 uses the per-PR (commit-level) dataset: one row per PR.
# Columns needed: pr_description, pr_total_commits, pr_final_files, pr_final_lines,
#                 first_commit_files, first_commit_lines, first_commit_snaps, reverted
DATASETS_DIR = "commit_data"
RESULTS_DIR  = "results"

def read_all_prs(dir_path):
    """Load per-PR data from commit_data and compute derived features."""
    csv_files = glob.glob(os.path.join(dir_path, '**', '*.csv'), recursive=True)
    if not csv_files:
        return pd.DataFrame()

    dfs_list = []
    for f in csv_files:
        try:
            if os.path.getsize(f) == 0:
                continue
            df = pd.read_csv(f)
            if df.empty:
                continue
            repo_name = os.path.basename(f).replace('_reversions.csv', '').replace('_', '/', 1)
            df['repo_name'] = repo_name
            dfs_list.append(df)
        except Exception:
            continue

    if not dfs_list:
        return pd.DataFrame()

    final_df = pd.concat(dfs_list, ignore_index=True)
    final_df['reverted'] = pd.to_numeric(final_df['reverted'], errors='coerce').fillna(0).astype(int)
    final_df['desc_length'] = final_df['pr_description'].fillna('').astype(str).str.len()
    final_df['status'] = final_df['reverted'].map({0: 'non-Reverted', 1: 'Reverted'})
    return final_df

def run_mwu_tests(df):
    """Run Mann-Whitney U tests on all metrics and return results as a list of dicts."""
    metrics = [
        ('desc_length',       'PR Description Length'),
        ('pr_total_commits',  'Revisions (# of Commits)'),
        ('pr_final_files',    'Changed Files'),
        ('pr_final_lines',    'Changed Lines'),
        ('first_commit_files','First Commit: Changed Files'),
        ('first_commit_lines','First Commit: Changed Lines'),
        ('first_commit_snaps','First Commit: Changed Snapshots'),
    ]

    results = []
    for col, label in metrics:
        if col not in df.columns:
            continue
        g_non = df[df['status'] == 'non-Reverted'][col].dropna()
        g_rev = df[df['status'] == 'Reverted'][col].dropna()
        if len(g_non) == 0 or len(g_rev) == 0:
            continue
        stat, p = mannwhitneyu(g_non, g_rev, alternative='two-sided')
        results.append({
            'metric':          label,
            'n_non_reverted':  len(g_non),
            'n_reverted':      len(g_rev),
            'median_non':      g_non.median(),
            'median_rev':      g_rev.median(),
            'U':               stat,
            'p_value':         p,
            'significant':     p < 0.05,
        })
    return results

def print_prevalence_stats(df, out_lines):
    total    = len(df)
    reverted = int(df['reverted'].sum())
    non_rev  = total - reverted
    rate     = reverted / total if total > 0 else 0
    ratio    = round(1 / rate) if rate > 0 else float('inf')

    lines = [
        "",
        "=" * 60,
        "RQ1: Prevalence of Snapshot Updates",
        "=" * 60,
        f"  Snapshot-related merged PRs  : {total:>8,}",
        f"    non-Reverted               : {non_rev:>8,}  ({non_rev/total:.2%})",
        f"    Reverted (>=1 snap)        : {reverted:>8,}  ({rate:.2%}  ~1 in {ratio:,})",
        "=" * 60,
    ]
    for l in lines:
        print(l)
    out_lines.extend(lines)

def print_mwu_table(results, out_lines):
    header = (
        f"\n{'Metric':<35} {'N(non)':>6} {'N(rev)':>6} "
        f"{'Med(non)':>10} {'Med(rev)':>10} {'U':>12} {'p-value':>12} {'Sig':>4}"
    )
    sep = "-" * 100
    lines = ["", "Mann-Whitney U Test Results (RQ1)", sep, header, sep]
    for r in results:
        sig = "  * " if r['significant'] else "    "
        lines.append(
            f"{r['metric']:<35} {r['n_non_reverted']:>6} {r['n_reverted']:>6} "
            f"{r['median_non']:>10.1f} {r['median_rev']:>10.1f} "
            f"{r['U']:>12.0f} {r['p_value']:>12.3e} {sig}"
        )
    lines.append(sep)
    lines.append("  * p < 0.05")
    lines.append("")
    for l in lines:
        print(l)
    out_lines.extend(lines)

def save_results(out_lines):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "rq1_stats.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"  -> Numerical results saved to: {path}")

def main():
    print("📂 Loading per-PR data from commit_data/ ...")
    df = read_all_prs(DATASETS_DIR)

    if df.empty:
        print("❌ No valid data found in commit_data/.")
        return

    out_lines = []

    # --- Prevalence statistics ---
    print_prevalence_stats(df, out_lines)

    # --- Mann-Whitney U tests ---
    results = run_mwu_tests(df)
    print_mwu_table(results, out_lines)

    # --- Violin plots (Figure 3: 4 panels) ---
    # (column, human-readable title, y-axis label, output filename)
    plots = [
        ('desc_length',      'PR Description Length', 'Length',        'violinplot_01_pr_desc_length.pdf'),
        ('pr_total_commits', 'Revisions',             'Revisions',     'violinplot_02_pr_total_commits.pdf'),
        ('pr_final_files',   'Changed Files',         'Changed files', 'violinplot_03_pr_final_files.pdf'),
        ('pr_final_lines',   'Changed Lines',         'Changed lines', 'violinplot_04_pr_final_lines.pdf'),
    ]

    print("🎨 Generating violin plots (Figure 3) ...")
    sns.set_theme(style="whitegrid")
    palette = {'non-Reverted': 'lightblue', 'Reverted': 'salmon'}
    df['compare'] = ''  # dummy x-axis

    for col, ylabel, filename in [(c, y, fn) for c, _, y, fn in plots]:
        if col not in df.columns:
            print(f"  [Skip] Column '{col}' not found.")
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df, x='compare', y=col,
            hue='status', split=True,
            palette=palette,
            inner='box',
            hue_order=['non-Reverted', 'Reverted'],
            bw_adjust=1.5,
        )
        plt.yscale('symlog')
        plt.ylim(bottom=0)
        plt.xlabel('')
        plt.ylabel(ylabel, fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc='upper right', fontsize=20, frameon=True)
        plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {filename}")

    # --- Save numerical results ---
    save_results(out_lines)
    print("\n✅ RQ1 analysis complete.")

if __name__ == "__main__":
    main()
