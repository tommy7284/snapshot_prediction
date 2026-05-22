import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# RQ2 uses the per-snapshot-file dataset: one row per .snap file per PR.
# This enables computing per-PR file-level revert ratios.
FILE_DATA_DIR = "file_data"
RESULTS_DIR   = "results"

def load_file_data(dir_path):
    """Load per-snapshot-file data and return a merged DataFrame."""
    csv_files = glob.glob(os.path.join(dir_path, '*_reversions.csv'))
    if not csv_files:
        print("❌ No CSV files found in file_data/.")
        return pd.DataFrame()

    list_df = []
    print(f"📂 Loading per-snapshot-file data from file_data/ ({len(csv_files)} files) ...")
    for csv_file in csv_files:
        if os.path.getsize(csv_file) == 0:
            continue
        try:
            tmp_df = pd.read_csv(csv_file)
            if tmp_df.empty:
                continue
            repo_name = os.path.basename(csv_file).replace('_reversions.csv', '')
            tmp_df['repo_name'] = repo_name
            list_df.append(tmp_df)
        except Exception:
            continue

    if not list_df:
        print("❌ No valid data found.")
        return pd.DataFrame()

    df = pd.concat(list_df, ignore_index=True)
    df['reverted'] = pd.to_numeric(df['reverted'], errors='coerce').fillna(0).astype(int)
    return df

def save_results(out_lines):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "rq2_stats.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"  -> Numerical results saved to: {path}")

def main():
    df = load_file_data(FILE_DATA_DIR)
    if df.empty:
        return

    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    sep  = "=" * 60
    sep2 = "-" * 60

    emit()
    emit(sep)
    emit("RQ2: Distribution of Reverted Snapshot Files")
    emit(sep)

    # =================================================================
    # Statistic 1: Absolute file-level revert rate
    # =================================================================
    total_files    = len(df)
    reverted_files = int(df['reverted'].sum())
    abs_rate       = reverted_files / total_files if total_files > 0 else 0

    emit()
    emit("[Statistic 1] Absolute File-Level Revert Rate")
    emit(sep2)
    emit(f"  Total updated snapshot files  : {total_files:>8,}")
    emit(f"  Reverted snapshot files       : {reverted_files:>8,}  ({abs_rate:.4%})")
    emit(sep2)

    # =================================================================
    # Aggregate to PR level
    # =================================================================
    pr_metrics = df.groupby(['repo_name', 'pr_number']).agg(
        total_snapshots=('reverted', 'count'),
        reverted_snapshots=('reverted', 'sum'),
    ).reset_index()

    reverted_prs       = pr_metrics[pr_metrics['reverted_snapshots'] > 0].copy()
    total_reverted_prs = len(reverted_prs)

    emit(f"  Reverted PRs (>=1 snap)       : {total_reverted_prs:>8,}")
    emit()

    if reverted_prs.empty:
        print("❌ No reverted PRs found.")
        return

    # =================================================================
    # Statistic 2: File-level revert ratio per PR (→ Figure 4)
    # =================================================================
    reverted_prs['revert_ratio']     = reverted_prs['reverted_snapshots'] / reverted_prs['total_snapshots']
    reverted_prs['revert_ratio_pct'] = reverted_prs['revert_ratio'] * 100

    bins   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10%','10-20%','20-30%','30-40%','40-50%',
              '50-60%','60-70%','70-80%','80-90%','90-100%']

    reverted_prs['rate_category'] = pd.cut(
        reverted_prs['revert_ratio_pct'], bins=bins, labels=labels, include_lowest=True
    )

    emit("[Statistic 2] File-Level Revert Ratio Distribution per PR")
    emit(sep2)
    emit(f"  {'Bin':<10} {'PRs':>6}  {'%':>7}")
    emit(f"  {'-'*10} {'-'*6}  {'-'*7}")
    for lbl in labels:
        count = int((reverted_prs['rate_category'] == lbl).sum())
        pct   = count / total_reverted_prs * 100
        emit(f"  {lbl:<10} {count:>6,}  {pct:>6.1f}%")
    emit(sep2)

    median_r = reverted_prs['revert_ratio_pct'].median()
    mean_r   = reverted_prs['revert_ratio_pct'].mean()
    emit(f"  Median revert ratio : {median_r:.1f}%")
    emit(f"  Mean   revert ratio : {mean_r:.1f}%")
    emit()

    # =================================================================
    # Statistic 3: Fully vs. Partially reverted PRs
    # =================================================================
    # Use integer comparison to avoid floating-point precision issues
    reverted_prs['revert_type'] = np.where(
        reverted_prs['reverted_snapshots'] == reverted_prs['total_snapshots'],
        'Fully Reverted', 'Partially Reverted'
    )
    type_counts     = reverted_prs['revert_type'].value_counts()
    fully_count     = int(type_counts.get('Fully Reverted',     0))
    partially_count = int(type_counts.get('Partially Reverted', 0))
    fully_pct       = fully_count     / total_reverted_prs * 100
    partially_pct   = partially_count / total_reverted_prs * 100

    emit("[Statistic 3] Fully vs. Partially Reverted PRs")
    emit(sep2)
    emit(f"  (a) Fully Reverted     : {fully_count:>6,}  ({fully_pct:.1f}%)")
    emit(f"  (b) Partially Reverted : {partially_count:>6,}  ({partially_pct:.1f}%)")
    emit(f"  Total reverted PRs     : {total_reverted_prs:>6,}")
    emit(sep2)
    emit()

    # =================================================================
    # Plot settings
    # =================================================================
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size":       11,
        "axes.labelsize":  12,
        "axes.titlesize":  13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # =================================================================
    # Figure 4: File-level revert ratio distribution (Statistic 2)
    # =================================================================
    print("🎨 Creating Figure 4 (revert ratio distribution) ...")

    cat_pct = reverted_prs['rate_category'].value_counts(normalize=True).reset_index()
    cat_pct.columns = ['rate_category', 'proportion']
    cat_pct['percentage'] = cat_pct['proportion'] * 100

    base_df = pd.DataFrame({'rate_category': labels})
    cat_props = pd.merge(base_df, cat_pct, on='rate_category', how='left').fillna(0)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(
        data=cat_props, x='rate_category', y='percentage', order=labels,
        color="#4844CE", edgecolor='black', ax=ax1,
    )

    max_pct  = cat_props['percentage'].max()
    ylim_max = int(np.ceil(max_pct / 10) * 10)
    ylim_max = max(ylim_max, 50)
    ylim_max = min(ylim_max, 100)
    ticks    = np.arange(0, ylim_max + 1, 10)

    ax1.set_ylim(0, ylim_max)
    ax1.set_yticks(ticks)

    ax_twin = ax1.twinx()
    ax_twin.set_ylim(0, ylim_max)
    ax_twin.set_yticks(ticks)
    ax_twin.set_yticklabels([
        str(int(round(t / 100 * total_reverted_prs))) for t in ticks
    ])

    ax1.set_xlabel('File-level Revert Ratio per PR (%)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Percentage of Reverted PRs (%)',     fontweight='bold')
    ax_twin.set_ylabel('Number of Reverted PRs',         fontweight='bold', labelpad=10)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    out_fig4 = 'rq2_fig4_revert_ratio_distribution.pdf'
    plt.tight_layout()
    plt.savefig(out_fig4, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {out_fig4}")

    # =================================================================
    # Figure: Fully vs. Partially reverted (Statistic 3)
    # =================================================================
    print("🎨 Creating Figure: Fully vs. Partially reverted PRs ...")
    fig, ax2 = plt.subplots(figsize=(6.5, 2.2))

    ax2.barh([0], [fully_pct],     color='#55A868', edgecolor='black', height=0.5,
             label=f'(a) Fully Reverted ({fully_count} PRs)')
    ax2.barh([0], [partially_pct], left=[fully_pct], color='#E75257', edgecolor='black', height=0.5,
             label=f'(b) Partially Reverted ({partially_count} PRs)')

    if fully_pct > 5:
        ax2.text(fully_pct / 2, 0, f'{fully_pct:.1f}%',
                 va='center', ha='center', color='white', weight='bold', fontsize=11)
    if partially_pct > 5:
        ax2.text(fully_pct + partially_pct / 2, 0, f'{partially_pct:.1f}%',
                 va='center', ha='center', color='white', weight='bold', fontsize=11)

    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel('Percentage of Reverted PRs (%)', fontweight='bold')
    ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False)

    out_fig_type = 'rq2_fully_vs_partially.pdf'
    plt.tight_layout()
    plt.savefig(out_fig_type, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {out_fig_type}")

    # --- Save numerical results ---
    save_results(out_lines)
    print("\n✅ RQ2 analysis complete.")

if __name__ == "__main__":
    main()
