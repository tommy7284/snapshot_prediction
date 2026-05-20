import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
REVERSION_DATA_DIR = "file_data" # Directory containing CSV files

def calculate_and_plot_separately():
    # 🌟 1. Scan and merge all CSV files
    csv_files = glob.glob(os.path.join(REVERSION_DATA_DIR, '*_reversions.csv'))
    if not csv_files:
        print("❌ No CSV files found. Please check the configured path.")
        return

    list_df = []
    print("📂 Starting to read CSV data...")
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        repo_name = file_name.replace('_reversions.csv', '')
        
        if os.path.exists(csv_file) and os.path.getsize(csv_file) == 0:
            continue
            
        try:
            tmp_df = pd.read_csv(csv_file)
            if tmp_df.empty:
                continue
            tmp_df['repo_name'] = repo_name
            list_df.append(tmp_df)
        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"⚠️ Warning: File read error in {file_name}: {e}")
            continue
            
    if not list_df:
        print("❌ No CSV files containing valid data were found.")
        return
        
    df = pd.concat(list_df, ignore_index=True)
    df['reverted'] = pd.to_numeric(df['reverted'], errors='coerce').fillna(0).astype(int)
    
    # -----------------------------------------------------------------
    # [Approach 2] Absolute file-level revert rate
    # -----------------------------------------------------------------
    total_snapshot_files = len(df)
    total_reverted_files = df['reverted'].sum()
    absolute_revert_rate = total_reverted_files / total_snapshot_files if total_snapshot_files > 0 else 0
    
    print(f"\n📈 === [Approach 2] Overall File-level Statistics ===")
    print(f"Total updated snapshot files: {total_snapshot_files:,}")
    print(f"Reverted snapshot files: {total_reverted_files:,}")
    print(f"Absolute file revert rate: {absolute_revert_rate:.4%}")

    # Aggregate per PR
    pr_metrics = df.groupby(['repo_name', 'pr_number']).agg(
        total_snapshots=('reverted', 'count'),
        reverted_snapshots=('reverted', 'sum')
    ).reset_index()
    
    reverted_prs = pr_metrics[pr_metrics['reverted_snapshots'] > 0].copy()
    total_reverted_prs_count = len(reverted_prs)
    
    print(f"\n🔍 === PR-level Filtering ===")
    print(f"Identified reverted PRs: {total_reverted_prs_count}")

    if reverted_prs.empty:
        print("❌ No Reverted PRs found for analysis.")
        return

    # -----------------------------------------------------------------
    # [Approach 1] File-level revert ratio per PR
    # -----------------------------------------------------------------
    reverted_prs['revert_ratio'] = reverted_prs['reverted_snapshots'] / reverted_prs['total_snapshots']
    reverted_prs['revert_ratio_pct'] = reverted_prs['revert_ratio'] * 100

    # -----------------------------------------------------------------
    # [Approach 3] Fully vs. Partially reverted PRs
    # -----------------------------------------------------------------
    reverted_prs['revert_type'] = np.where(reverted_prs['revert_ratio'] >= 1.0, 'Fully Reverted', 'Partially Reverted')
    type_counts = reverted_prs['revert_type'].value_counts()
    fully_count = type_counts.get('Fully Reverted', 0)
    partially_count = type_counts.get('Partially Reverted', 0)
    
    fully_pct = (fully_count / total_reverted_prs_count) * 100
    partially_pct = (partially_count / total_reverted_prs_count) * 100

    # -----------------------------------------------------------------
    # 🎨 Common plot settings for the paper
    # -----------------------------------------------------------------
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    # =================================================================
    # 🎨 Figure 1: Approach 1 (Left Y-axis: Percentage / Right Y-axis: Actual Count)
    # =================================================================
    print("\n🎨 Creating Figure 1 (Approach 1: Twin Y-axis graph)...")
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    reverted_prs['rate_category'] = pd.cut(reverted_prs['revert_ratio_pct'], bins=bins, labels=labels, include_lowest=True)
    category_pct = reverted_prs['rate_category'].value_counts(normalize=True).reset_index()
    category_pct.columns = ['rate_category', 'proportion']
    category_pct['percentage'] = category_pct['proportion'] * 100
    
    base_df = pd.DataFrame({'rate_category': labels})
    category_proportions = pd.merge(base_df, category_pct, on='rate_category', how='left').fillna(0)
    
    # 1. Draw percentage bar plot based on the left axis (ax1)
    sns.barplot(
        data=category_proportions, x='rate_category', y='percentage', order=labels,
        color="#4844CE", edgecolor='black', ax=ax1
    )
    
    # 2. Scale the Y-axis range neatly (in 10% increments) according to the max percentage
    max_pct = category_proportions['percentage'].max()
    ylim_max_pct = int(np.ceil(max_pct / 10) * 10) # Round up to the nearest 10%
    if ylim_max_pct < 50:  # Ensure at least 50% is displayed for better appearance
        ylim_max_pct = 50
    if ylim_max_pct > 100:
        ylim_max_pct = 100
        
    ax1.set_ylim(0, ylim_max_pct)
    tick_values = np.arange(0, ylim_max_pct + 1, 10)
    ax1.set_yticks(tick_values)
    
    # 3. Create right axis (secondary axis) and sync ticks completely with the left axis
    ax_twin = ax1.twinx()
    ax_twin.set_ylim(0, ylim_max_pct)
    ax_twin.set_yticks(tick_values)
    
    # Rewrite only the "display labels" of the right axis to the "actual number of PRs (rounded to integer)"
    # Formula: Count = (Percentage / 100) * Total reverted PRs
    twin_labels = [f"{int(round((x / 100) * total_reverted_prs_count))}" for x in tick_values]
    ax_twin.set_yticklabels(twin_labels)
    
    # 4. Set axis labels and title
    ax1.set_xlabel('File-level Revert Ratio per PR (%)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Percentage of Reverted PRs (%)', fontweight='bold')
    ax_twin.set_ylabel('Number of Reverted PRs', fontweight='bold', labelpad=10)
    # plt.title('Distribution of File-level Revert Ratios (Approach 1)', pad=12)
    
    # Workaround for UserWarning
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Sync background grid lines with the left percentage axis
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    output_fig1 = 'approach1_revert_ratio_distribution.pdf'
    plt.tight_layout()
    plt.savefig(output_fig1, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   -> Saved: {output_fig1}")

    # =================================================================
    # 🎨 Figure 2: Approach 3 (Horizontal Stacked Bar Chart)
    # =================================================================
    print("🎨 Creating Figure 2 (Approach 3: Horizontal stacked bar chart)...")
    fig, ax2 = plt.subplots(figsize=(6.5, 2.2))
    
    ax2.barh([0], [fully_pct], color='#55A868', edgecolor='black', height=0.5, label=f'(a) Fully Reverted ({fully_count} PRs)')
    ax2.barh([0], [partially_pct], left=[fully_pct], color="#E75257", edgecolor='black', height=0.5, label=f'(b) Partially Reverted ({partially_count} PRs)')
    
    if fully_pct > 5:
        ax2.text(fully_pct / 2, 0, f'{fully_pct:.1f}%', va='center', ha='center', color='white', weight='bold', fontsize=11)
    if partially_pct > 5:
        ax2.text(fully_pct + (partially_pct / 2), 0, f'{partially_pct:.1f}%', va='center', ha='center', color='white', weight='bold', fontsize=11)
        
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel('Percentage of Reverted PRs (%)', fontweight='bold')
    # ax2.set_title('Proportion of Fully vs. Partially Reverted PRs (Approach 3)', pad=12)
    ax2.legend(
        loc='lower right', 
        bbox_to_anchor=(1.0, 1.02), 
        ncol=2, 
        frameon=False
    )
    output_fig2 = 'approach3_fully_vs_partially.pdf'
    plt.tight_layout()
    plt.savefig(output_fig2, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   -> Saved: {output_fig2}")
    
    print("\n🎉 All independent figures have been successfully generated!")

if __name__ == "__main__":
    calculate_and_plot_separately()