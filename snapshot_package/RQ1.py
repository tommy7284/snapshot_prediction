import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- Configuration ---
DATASETS_DIR = "commit_data" # Folder containing PR data

def read_all_prs(dir_path):
    """Load all commit data and assign labels for comparison."""
    search_pattern = os.path.join(dir_path, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)
    if not csv_files: return pd.DataFrame()

    dfs_list = []
    for f in csv_files:
        try:
            if os.path.getsize(f) == 0: continue
            df = pd.read_csv(f)
            if df.empty: continue
            
            repo_name = os.path.basename(f).replace('_reversions.csv', '').replace('_', '/', 1)
            df['repo_name'] = repo_name
            dfs_list.append(df)
        except Exception:
            continue

    if not dfs_list: return pd.DataFrame()
    
    final_df = pd.concat(dfs_list, ignore_index=True)
    final_df['reverted'] = pd.to_numeric(final_df['reverted'], errors='coerce').fillna(0).astype(int)
    
    final_df['desc_length'] = final_df['pr_description'].fillna('').astype(str).str.len()
    final_df['commit_msg_length'] = final_df['commit_message'].fillna('').astype(str).str.len()
    final_df['status'] = final_df['reverted'].map({0: 'non-Reverted', 1: 'Reverted'})
    
    return final_df

def print_mwu_test(df, col_name, title):
    group_normal = df[df['status'] == 'non-Reverted'][col_name].dropna()
    group_reverted = df[df['status'] == 'Reverted'][col_name].dropna()
    
    if len(group_normal) == 0 or len(group_reverted) == 0:
        return

    median_normal = group_normal.median()
    median_reverted = group_reverted.median()
    stat, p_value = mannwhitneyu(group_normal, group_reverted, alternative='two-sided')
    
    print(f"\n🔬 [U-Test] {title}")
    print(f"  - Median: Normal = {median_normal:.1f} / Reverted = {median_reverted:.1f}")
    print(f"  - U-Statistic: {stat}")
    print(f"  - p-value    : {p_value:.3e}")
    
    if p_value < 0.05:
        print("  => Conclusion: [Statistically significant difference] between the two groups (p < 0.05) ⭐")
    else:
        print("  => Conclusion: [No statistically significant difference] between the two groups (p >= 0.05)")

def main():
    print("📂 Loading CSV data...")
    df = read_all_prs(DATASETS_DIR)
    
    if df.empty: 
        print("❌ No valid data found.")
        return

    print("\n" + "="*50)
    print("📊 Statistical Test Results (Mann-Whitney U test)")
    print("="*50)

    print_mwu_test(df, 'desc_length', 'PR Description Length')
    print_mwu_test(df, 'commit_msg_length', 'Commit Message Length')
    print_mwu_test(df, 'first_commit_files', 'First Commit: Changed Files')
    print_mwu_test(df, 'first_commit_lines', 'First Commit: Changed Lines')
    print_mwu_test(df, 'first_commit_snaps', 'First Commit: Changed Snapshots')
    print_mwu_test(df, 'pr_total_commits', 'PR Total: Number of Commits')
    print_mwu_test(df, 'pr_final_files', 'PR Total: Final Changed Files')
    print_mwu_test(df, 'pr_final_lines', 'PR Total: Final Changed Lines')
    print("="*50)

    # Note: Removed plots 2, 3, 4, 5 as requested
    plots = [
        ('desc_length', 'PR Description Length', 'violinplot_01_pr_desc_length.pdf'),
        ('pr_total_commits', 'PR Total: Number of Commits', 'violinplot_02_pr_total_commits.pdf'),
        ('pr_final_files', 'PR Total: Final Changed Files', 'violinplot_03_pr_final_files.pdf'),
        ('pr_final_lines', 'PR Total: Final Changed Lines', 'violinplot_04_pr_final_lines.pdf')
    ]

    print("\n🎨 Drawing split violin plots and saving them as individual images...")
    sns.set_theme(style="whitegrid")
    
    palette = {'non-Reverted': 'lightblue', 'Reverted': 'salmon'}

    # 🌟 Add: Prepare a dummy column to fix the X-axis in one place
    df['compare'] = ''

    for col, title, filename in plots:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 6))
        
        # 🌟 Change: Set split=True to draw Normal on the left and Reverted on the right
        sns.violinplot(
            data=df, x='compare', y=col, 
            hue='status', split=True, 
            # Draw a box inside the violin
            palette=palette, 
            inner='box',
            hue_order=['non-Reverted', 'Reverted'],
            bw_adjust=1.5
        )
        
        plt.yscale('symlog')
        plt.ylim(bottom=0)
        
        # Overall graph decoration
        # plt.title(title, fontsize=24, pad=15)
        plt.xlabel('') # Remove the dummy column label as it is unnecessary
        y_label_text = 'Length' if 'length' in col else 'Changed lines' if 'lines' in col else 'Revisions' if 'commits' in col else 'Changed files'
        plt.ylabel(y_label_text, fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # Place legend in the upper right
        plt.legend(loc='upper right', fontsize=24, frameon=True)
        
        plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close() 
        
        print(f"  -> Saved: {filename}")

    print("\n✅ All graphs have been successfully generated!")

if __name__ == "__main__":
    main()