import pygit2
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import requests
import time

import revert_Logic

# --- Configuration ---
REPOS_DIR = "repos"
JSON_DIR = "jestable"
FILE_DATA_DIR = "file_data"      # per-snapshot-file dataset (one row per .snap × PR)
COMMIT_DATA_DIR = "commit_data"  # per-PR dataset (one row per PR)
SNAPSHOT_EXTENSION = ".snap"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Cache to speed up tree traversal
tree_snapshot_cache = {}

def load_all_json_files(directory):
    json_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        json_data.append(data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to read file '{file_path}'. Skipping. Error: {e}")
    return json_data

def setup_repo(repo_url, clone_dir):
    if os.path.exists(clone_dir):
        repo = pygit2.Repository(clone_dir)
    else:
        print(f"Cloning repository '{repo_url}'...")
        repo = pygit2.clone_repository(repo_url, clone_dir)
        print("✅ Clone complete.")

    remote = repo.remotes["origin"]
    refs = ["+refs/heads/*:refs/remotes/origin/*","+refs/pull/*/head:refs/remotes/origin/pr/*"]

    print(f"📡 Fetching from remote '{remote.name}'...")
    try:
        remote.fetch(refspecs=refs)
        print("✅ Fetch complete.")
    except pygit2.GitError as e:
        print(f"⚠️ Error occurred during fetch, but continuing: {e}")

    return repo

def get_default_branch(repo):
    head_ref = repo.references.get('refs/remotes/origin/HEAD')
    if not head_ref:
        for branch in ['main', 'master', 'develop']:
            if f'refs/remotes/origin/{branch}' in repo.references:
                return branch
        raise RuntimeError("Could not determine the default branch.")
    target_ref_name = head_ref.target
    return target_ref_name.split('/')[-1]

def get_snapshot_introduction_time(repo, default_branch):
    try:
        head_commit = repo.revparse_single(f"origin/{default_branch}")
    except pygit2.GitError:
        return None

    walker = repo.walk(head_commit.id, pygit2.GIT_SORT_REVERSE)
    for commit in walker:
        if not commit.parents: continue
        parent_commit = commit.parents[0]
        diff = repo.diff(parent_commit.tree, commit.tree)
        for delta in diff.deltas:
            if delta.new_file.path.endswith(".snap") and delta.status == pygit2.GIT_DELTA_ADDED:
                return commit.commit_time
    return None

def get_snapshot_files_from_tree(repo, tree):
    if tree.id in tree_snapshot_cache:
        return tree_snapshot_cache[tree.id]

    snapshot_files = []
    def _walk(current_tree, path_so_far):
        for entry in current_tree:
            current_path = os.path.join(path_so_far, entry.name) if path_so_far else entry.name
            if entry.type == pygit2.GIT_OBJECT_TREE:
                _walk(repo.get(entry.id), current_path)
            elif entry.type == pygit2.GIT_OBJECT_BLOB and entry.name.endswith(".snap"):
                snapshot_files.append(current_path)

    _walk(tree, "")
    tree_snapshot_cache[tree.id] = snapshot_files
    return snapshot_files

def get_source_file_from_snap(snap_path):
    dir_name = os.path.dirname(snap_path)
    base_name = os.path.basename(snap_path)
    if base_name.endswith('.snap'):
        base_name = base_name[:-5]
    if os.path.basename(dir_name) == '__snapshots__':
        parent_dir = os.path.dirname(dir_name)
        return os.path.join(parent_dir, base_name).replace('\\', '/')
    return os.path.join(dir_name, base_name).replace('\\', '/')

def get_merged_prs_data_via_graphql(repo_full_name: str, pr_numbers: list) -> dict:
    if not GITHUB_TOKEN:
        print("\n[Warning] GITHUB_TOKEN is not set. Assuming all PRs are merged and have no body.")
        return {pr: {"body": "", "changedFiles": 0, "additions": 0, "deletions": 0} for pr in pr_numbers}

    owner, name = repo_full_name.split('/')
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    merged_pr_data = {}
    chunk_size = 50

    for i in range(0, len(pr_numbers), chunk_size):
        chunk = pr_numbers[i:i+chunk_size]
        query_parts = []
        for pr in chunk:
            query_parts.append(f'''
            pr_{pr}: pullRequest(number: {pr}) {{
                merged
                body
                baseRefName
                changedFiles
                additions
                deletions
                commits(first: 100) {{ nodes {{ commit {{ oid }} }} }}
            }}''')

        query_body = "\n".join(query_parts)
        query = f"""
        query {{
          repository(owner: "{owner}", name: "{name}") {{
            {query_body}
          }}
        }}
        """

        retry_count = 0
        while retry_count < 3:
            try:
                response = requests.post("https://api.github.com/graphql", json={"query": query}, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and data["data"] is not None:
                        repo_data = data["data"].get("repository") or {}
                        for pr in chunk:
                            pr_data = repo_data.get(f"pr_{pr}")
                            if pr_data and pr_data.get("merged"):
                                commit_nodes = pr_data.get("commits", {}).get("nodes", [])
                                expected_oids = [node["commit"]["oid"] for node in commit_nodes if "commit" in node]

                                merged_pr_data[pr] = {
                                    "body": pr_data.get("body") or "",
                                    "base_branch": pr_data.get("baseRefName"),
                                    "expected_oids": expected_oids,
                                    "changedFiles": pr_data.get("changedFiles") or 0,
                                    "additions": pr_data.get("additions") or 0,
                                    "deletions": pr_data.get("deletions") or 0,
                                }
                    break
                elif response.status_code in (403, 429):
                    print(f"\n[API Rate Limit] Limit reached. Waiting 60 seconds...")
                    time.sleep(60)
                    retry_count += 1
                else:
                    print(f"\n[API Error] Status code: {response.status_code}")
                    break
            except Exception as e:
                print(f"\n[Network Error] {e}. Retrying in 10 seconds...")
                time.sleep(10)
                retry_count += 1
        time.sleep(0.5)
    return merged_pr_data

def process_unified_pr_data(repo, repo_path, repo_full_name):
    """Build both the file-level and commit-level datasets in a single pass."""
    pr_refs = [r for r in repo.references if r.startswith("refs/remotes/origin/pr/") and r.split('/')[-1].isdigit()]

    if not pr_refs:
        print(f"Warning: No PR references found in '{repo_path}'.")
        return [], [], [], 0

    try:
        default_branch = get_default_branch(repo)
        target_commit = repo.revparse_single(f"origin/{default_branch}")
    except (RuntimeError, pygit2.GitError) as e:
        return [], [], [], 0

    intro_time = get_snapshot_introduction_time(repo, default_branch)
    if intro_time is None: intro_time = 0

    print(f"Analyzing {len(pr_refs)} pull requests in '{repo_path}'...")

    # === [Step 1] Extract candidate PRs ===
    candidate_pr_numbers = []
    candidate_pr_data = {}
    filtered_pr_count = 0

    for ref_name in pr_refs:
        ref = repo.references.get(ref_name)
        if not ref or not ref.target: continue
        pr_head_commit = repo.get(ref.target)
        pr_number = ref_name.split('/')[-1]

        if pr_head_commit.commit_time < intro_time:
            continue
        filtered_pr_count += 1

        try:
            base_oid = repo.merge_base(pr_head_commit.id, target_commit.id)
            if base_oid is None: continue
            base_commit = repo.get(base_oid)

            base_snap_files = get_snapshot_files_from_tree(repo, base_commit.tree)
            head_snap_files = get_snapshot_files_from_tree(repo, pr_head_commit.tree)
            current_snapshot_files = set(base_snap_files + head_snap_files)

            snapshot_dirs = set()
            source_to_snap = {}
            for snap_path in current_snapshot_files:
                snapshot_dirs.add(os.path.dirname(snap_path))
                src_path = get_source_file_from_snap(snap_path)
                source_to_snap[src_path] = snap_path

            if not snapshot_dirs: continue

            diff = repo.diff(base_commit.tree, pr_head_commit.tree)
            modified_paths = set()
            for patch in diff:
                path = patch.delta.new_file.path or patch.delta.old_file.path
                if path: modified_paths.add(path)

            is_relevant_pr = False
            for path in modified_paths:
                modified_dir = os.path.dirname(path)
                if any(snap_dir == modified_dir or snap_dir.startswith(modified_dir + os.sep) for snap_dir in snapshot_dirs):
                    is_relevant_pr = True
                    break

            if is_relevant_pr:
                candidate_pr_numbers.append(pr_number)
                candidate_pr_data[pr_number] = {
                    'pr_head_id': pr_head_commit.id,
                    'base_id': base_commit.id,
                    'modified_paths': modified_paths,
                    'source_to_snap': source_to_snap,
                    'diff': diff
                }
        except pygit2.GitError:
            continue

    print(f"✅ Snapshot-related candidate PRs: {len(candidate_pr_numbers)}")

    # === [Step 2] Fetch merge status via GraphQL ===
    if not candidate_pr_numbers:
        return [], [], [], filtered_pr_count

    print("📡 Fetching merge statuses in bulk via GitHub API (GraphQL)...")
    merged_pr_info = get_merged_prs_data_via_graphql(repo_full_name, candidate_pr_numbers)
    print(f"✅ Actually merged related PRs: {len(merged_pr_info)}")

    # === [Step 3] Build both datasets in a single pass ===
    print("🧠 Executing detailed analysis and dataset extraction simultaneously...")
    pr_analysis_list = []
    file_dataset_list = []
    commit_dataset_list = []

    for pr_number, pr_info in merged_pr_info.items():
        data = candidate_pr_data[pr_number]
        try:
            pr_head_commit = repo.get(data['pr_head_id'])
            base_commit = repo.get(data['base_id'])
            modified_paths = data['modified_paths']
            source_to_snap = data['source_to_snap']
            diff = data['diff']
            expected_oids = pr_info.get("expected_oids", [])

            raw_pr_commits = revert_Logic.get_pr_commits_first_parent(repo, pr_head_commit.id, base_commit.id)
            if expected_oids:
                raw_pr_commits = [c for c in raw_pr_commits if str(c.id) in expected_oids]

            if not raw_pr_commits: continue

            # Skip PRs entirely if they contain merge commits with 2 or more parents
            has_merge_commit = any(len(commit.parents) >= 2 for commit in raw_pr_commits)
            if has_merge_commit:
                continue

            pr_commits = raw_pr_commits

            last_non_merge = revert_Logic.get_last_non_merge_commit(repo, pr_head_commit.id, base_commit.id)
            if not last_non_merge: continue

            # Commit-level PR metrics (from GitHub API)
            pr_total_commits = len(raw_pr_commits)
            pr_final_files = pr_info.get('changedFiles', 0)
            pr_final_lines = pr_info.get('additions', 0) + pr_info.get('deletions', 0)

            # --- Per-snapshot-file state (file-level dataset) ---
            snap_first_updates = {}

            # --- Per-PR state (commit-level dataset) ---
            touched_ss_files = set()
            first_ss_diff_text = ""
            first_ss_commit_msg = ""
            first_ss_commit_hash = ""
            commit_first_commit_files = 0
            commit_first_commit_lines = 0
            first_commit_snaps = 0

            # --- Shared state ---
            total_ss_updates_in_pr = 0
            first_update_position = -1

            pr_commits_chronological = sorted(raw_pr_commits, key=lambda c: c.commit_time)

            # Single pass over commits: collect data for both datasets simultaneously
            for i, commit in enumerate(pr_commits_chronological):
                if not commit.parents: continue
                commit_diff = repo.diff(commit.parents[0], commit)
                patches = list(commit_diff)
                deltas = commit_diff.deltas
                _commit_files = len(deltas)
                _commit_lines = sum(p.line_stats[1] + p.line_stats[2] for p in patches)

                is_valid_ss_update = False
                ss_files_in_this_commit = set()

                for delta in deltas:
                    path = delta.new_file.path
                    if not (path and path.endswith(SNAPSHOT_EXTENSION)):
                        continue
                    if not revert_Logic.is_user_change(repo, commit, path):
                        continue

                    is_valid_ss_update = True
                    ss_files_in_this_commit.add(path)
                    touched_ss_files.add(path)

                    # File-level: record the first update per snap path
                    if path not in snap_first_updates:
                        src_path = get_source_file_from_snap(path)
                        src_diff_text = ""
                        for patch in patches:
                            fp = patch.delta.new_file.path or patch.delta.old_file.path
                            if fp == src_path:
                                src_diff_text = patch.data.decode('utf-8', 'replace')
                                break
                        snap_first_updates[path] = {
                            'commit_hash': str(commit.id),
                            'commit_msg': commit.message,
                            'src_path': src_path,
                            'diff_text': src_diff_text,
                            'first_commit_files': _commit_files,
                            'first_commit_lines': _commit_lines,
                        }

                if is_valid_ss_update:
                    total_ss_updates_in_pr += 1
                    if first_update_position == -1:
                        first_update_position = i + 1

                # Commit-level: record metrics from the first SS-touching commit only
                if ss_files_in_this_commit and not first_ss_diff_text:
                    first_ss_commit_msg = commit.message
                    first_ss_commit_hash = str(commit.id)
                    commit_first_commit_files = _commit_files
                    commit_first_commit_lines = _commit_lines
                    first_commit_snaps = len(ss_files_in_this_commit)
                    expected_src_files = {get_source_file_from_snap(s) for s in ss_files_in_this_commit}
                    commit_diffs = []
                    for patch in patches:
                        fp = patch.delta.new_file.path or patch.delta.old_file.path
                        if fp in expected_src_files:
                            commit_diffs.append(patch.data.decode('utf-8', 'replace'))
                    first_ss_diff_text = "\n".join(commit_diffs)

            # --- Count lines and changes for analysis ---
            pr_source_change_count = 0
            pr_snap_update_count = 0
            local_source_to_snap = source_to_snap.copy()
            for path in modified_paths:
                if path.endswith('.snap'):
                    src = get_source_file_from_snap(path)
                    local_source_to_snap[src] = path

            for path in modified_paths:
                if path in local_source_to_snap:
                    pr_source_change_count += 1
                    corr_snap = local_source_to_snap[path]
                    if corr_snap in modified_paths:
                        pr_snap_update_count += 1

            changed_snaps = [p for p in modified_paths if p.endswith('.snap')]
            target_source_files = {get_source_file_from_snap(p) for p in changed_snaps}

            snap_lines_changed = 0
            source_lines_changed = 0

            for patch in diff:
                path = patch.delta.new_file.path or patch.delta.old_file.path
                if not path: continue
                _, adds, dels = patch.line_stats
                total_lines = adds + dels
                if path in changed_snaps:
                    snap_lines_changed += total_lines
                elif path in target_source_files:
                    source_lines_changed += total_lines

            has_update_in_first_commit = 1 if first_update_position == 1 else 0

            if pr_source_change_count > 0:
                pr_analysis_list.append({
                    'ss_update_count': total_ss_updates_in_pr,
                    'has_update_in_first_commit': has_update_in_first_commit,
                    'first_update_position': first_update_position,
                    'snap_lines_changed': snap_lines_changed,
                    'source_lines_changed': source_lines_changed,
                    'source_change_count': pr_source_change_count,
                    'snap_update_count': pr_snap_update_count
                })

                # --- File-level dataset: one row per .snap file per PR ---
                if snap_first_updates:
                    for snap_path, info in snap_first_updates.items():
                        is_reverted = False
                        revert_type = ""
                        if revert_Logic.check_reversion(repo, snap_path, base_commit, last_non_merge, pr_commits):
                            is_reverted = True
                            base_oid = revert_Logic.get_blob_oid(repo, base_commit, snap_path)
                            head_oid = revert_Logic.get_blob_oid(repo, last_non_merge, snap_path)
                            revert_type = "None=None" if (base_oid is None and head_oid is None) else "OID=OID"

                        file_dataset_list.append({
                            "pr_number": pr_number,
                            "pr_head_commit": str(pr_head_commit.id),
                            "commit_hash": info['commit_hash'],
                            "snap_path": snap_path,
                            "source_path": info['src_path'],
                            "pr_description": pr_info.get("body", ""),
                            "commit_message": info['commit_msg'],
                            "diff_text": info['diff_text'],
                            "reverted": 1 if is_reverted else 0,
                            "revert_type": revert_type,
                            "first_commit_files": info['first_commit_files'],
                            "first_commit_lines": info['first_commit_lines'],
                        })

                # --- Commit-level dataset: one row per PR ---
                if touched_ss_files:
                    is_reverted = False
                    revert_type = ""
                    changed_snapshots = revert_Logic.get_actual_pr_changes(repo, pr_commits, base_commit)
                    for snap_path in changed_snapshots:
                        if revert_Logic.check_reversion(repo, snap_path, base_commit, last_non_merge, pr_commits):
                            is_reverted = True
                            base_oid = revert_Logic.get_blob_oid(repo, base_commit, snap_path)
                            head_oid = revert_Logic.get_blob_oid(repo, last_non_merge, snap_path)
                            revert_type = "None=None" if (base_oid is None and head_oid is None) else "OID=OID"
                            break

                    commit_dataset_list.append({
                        "pr_number": pr_number,
                        "pr_head_commit": str(pr_head_commit.id),
                        "commit_hash": first_ss_commit_hash,
                        "pr_description": pr_info.get("body", ""),
                        "commit_message": first_ss_commit_msg,
                        "diff_text": first_ss_diff_text,
                        "reverted": 1 if is_reverted else 0,
                        "revert_type": revert_type,
                        "first_commit_files": commit_first_commit_files,
                        "first_commit_lines": commit_first_commit_lines,
                        "first_commit_snaps": first_commit_snaps,
                        "pr_total_commits": pr_total_commits,
                        "pr_final_files": pr_final_files,
                        "pr_final_lines": pr_final_lines,
                    })

        except Exception as e:
            print(f"Error processing PR #{pr_number}: {e}")
            continue

    # Clear cache to prevent memory leaks
    tree_snapshot_cache.clear()

    return pr_analysis_list, file_dataset_list, commit_dataset_list, filtered_pr_count

def write_to_csv(data_list, filename):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            if not data_list:
                print(f"✅ Since data is empty, created an empty CSV '{filename}'.")
                return
            headers = data_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_list)
        print(f"✅ Successfully saved {len(data_list)} records to '{filename}'.")
    except IOError as e:
        print(f"❌ Error occurred while writing to file '{filename}': {e}")

def calculate_and_print_final_stats(all_pr_data, total_scanned_prs):
    relevant_pr_count = len(all_pr_data)
    if total_scanned_prs == 0:
        print("\nNo pull requests found for analysis.")
        return

    prs_with_any_ss_update = sum(1 for pr in all_pr_data if pr['ss_update_count'] > 0)
    prs_with_update_in_first = sum(pr['has_update_in_first_commit'] for pr in all_pr_data)
    total_source_changes = sum(pr['source_change_count'] for pr in all_pr_data)
    total_snap_updates_for_sources = sum(pr['snap_update_count'] for pr in all_pr_data)

    print("\n" + "="*20 + " Final Analysis Results " + "="*20)
    print(f"Total PRs investigated (Post-Snapshot Intro): {total_scanned_prs}")
    print(f"Snapshot-related PRs (Merged only): {relevant_pr_count}")
    print(f"PRs containing SS updates: {prs_with_any_ss_update}")
    print("-" * 64)
    if total_scanned_prs > 0:
        print(f"0. % of Snapshot-related PRs among investigated: {(relevant_pr_count / total_scanned_prs) * 100:.2f}%")
    if relevant_pr_count > 0:
        print(f"1. % of PRs with any SS update (in related PRs): {(prs_with_any_ss_update / relevant_pr_count) * 100:.2f}%")
        print(f"2. % of PRs with SS update in 1st commit: {(prs_with_update_in_first / relevant_pr_count) * 100:.2f}%")
    if prs_with_any_ss_update > 0:
        print(f"3. % of '1st commit updates' among updated PRs: {(prs_with_update_in_first / prs_with_any_ss_update) * 100:.2f}%")
    print("-" * 64)
    print("🌟 --- Snapshot Update Probability (Change Counts) ---")
    print(f"Total Source File Changes: {total_source_changes} times")
    print(f"Corresponding Snapshot Updates: {total_snap_updates_for_sources} times")
    if total_source_changes > 0:
        update_prob = (total_snap_updates_for_sources / total_source_changes) * 100
        print(f"👉 Snapshot Update Probability: {update_prob:.2f}%")
    else:
        print("👉 Snapshot Update Probability: N/A")
    print("="*64)

def plot_source_vs_snap_changes(all_pr_data):
    data = [pr for pr in all_pr_data if pr['snap_lines_changed'] > 0 or pr['source_lines_changed'] > 0]
    if not data: return
    source_changes = [pr['source_lines_changed'] for pr in data]
    snap_changes = [pr['snap_lines_changed'] for pr in data]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.scatter(source_changes, snap_changes, alpha=0.6, color='coral', edgecolors='k')
    if len(data) > 1:
        correlation = np.corrcoef(source_changes, snap_changes)[0, 1]
        plt.title(f'Relationship Between Source & Snapshot Line Changes\n(Correlation: {correlation:.2f})', fontsize=15, pad=15)
    else:
        plt.title('Relationship Between Source & Snapshot Line Changes', fontsize=15, pad=15)

    plt.xlabel('Source File Line Changes (Additions + Deletions)', fontsize=12)
    plt.ylabel('Snapshot File Line Changes (Additions + Deletions)', fontsize=12)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("ss_vs_source_changes.png")
    print("\n📈 Saved scatter plot as 'ss_vs_source_changes.png'.")

def plot_first_update_position(all_pr_data):
    positions = [pr['first_update_position'] for pr in all_pr_data if pr.get('first_update_position', -1) > 0]

    if not positions:
        print("\n[Skip Graph] No PRs found with snapshot updates.")
        return

    df = pd.DataFrame({'position': positions})
    counts = df['position'].value_counts().sort_index()
    max_display_pos = min(max(counts.index), 15)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color='cornflowerblue', edgecolor='black', alpha=0.8)
    plt.title('At Which Commit Are Snapshots First Updated in a PR?', fontsize=15, pad=15)
    plt.xlabel('Commit Position (1 = First Commit in PR)', fontsize=12)
    plt.ylabel('Number of PRs', fontsize=12)
    plt.xticks(range(1, max_display_pos + 1))
    plt.xlim(0.5, max_display_pos + 0.5)
    for i, v in counts.items():
        if i <= max_display_pos:
            plt.text(i, v + (max(counts.values)*0.01), str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig("first_snapshot_update_position.png", dpi=300)
    print("\n📈 Saved bar chart as 'first_snapshot_update_position.png'.")

# --- Main Process ---
if __name__ == "__main__":
    grand_all_pr_analysis_results = []
    grand_all_file_dataset_results = []
    grand_all_commit_dataset_results = []
    grand_total_scanned_prs = 0

    if not os.path.isdir(JSON_DIR):
        print(f"Error: JSON directory '{JSON_DIR}' not found.")
    else:
        jsons = load_all_json_files(JSON_DIR)
        processed_repos = set()

        for data in jsons:
            name_value = data.get('name')
            if not name_value: continue
            if name_value in processed_repos: continue
            processed_repos.add(name_value)

            repo_url = 'https://github.com/' + name_value
            repo_dir_name = name_value.replace('/', '_')
            clone_dir = os.path.join(REPOS_DIR, repo_dir_name)
            file_output_path = os.path.join(FILE_DATA_DIR, f"{repo_dir_name}_reversions.csv")
            commit_output_path = os.path.join(COMMIT_DATA_DIR, f"{repo_dir_name}_reversions.csv")

            print("\n" + "="*60)
            print(f"Starting unified analysis: {name_value}")
            print("="*60)

            try:
                repo = setup_repo(repo_url, clone_dir)
            except (pygit2.GitError, RuntimeError) as e:
                print(f"Error: Failed to setup repository '{repo_url}'. Skipping. Details: {e}")
                continue

            pr_analysis_list, file_dataset_list, commit_dataset_list, scanned_count = \
                process_unified_pr_data(repo, clone_dir, name_value)

            grand_total_scanned_prs += scanned_count

            if pr_analysis_list:
                grand_all_pr_analysis_results.extend(pr_analysis_list)
                grand_all_file_dataset_results.extend(file_dataset_list)
                grand_all_commit_dataset_results.extend(commit_dataset_list)
                write_to_csv(file_dataset_list, file_output_path)
                write_to_csv(commit_dataset_list, commit_output_path)
            else:
                write_to_csv([], file_output_path)
                write_to_csv([], commit_output_path)

    if grand_total_scanned_prs > 0:
        calculate_and_print_final_stats(grand_all_pr_analysis_results, grand_total_scanned_prs)

        total_reverts = sum(1 for d in grand_all_file_dataset_results if d.get('reverted') == 1)
        none_none_count = sum(1 for d in grand_all_file_dataset_results if d.get('revert_type') == "None=None")
        oid_oid_count = sum(1 for d in grand_all_file_dataset_results if d.get('revert_type') == "OID=OID")

        print("\n" + "="*64)
        print("🌟 --- Revert Type Breakdown ---")
        print(f"Total Reverted PRs : {total_reverts}")
        print(f"  -> Created then deleted (None=None) : {none_none_count}")
        print(f"  -> Updated then restored (OID=OID)   : {oid_oid_count}")
        print("="*64)

        plot_source_vs_snap_changes(grand_all_pr_analysis_results)
        plot_first_update_position(grand_all_pr_analysis_results)
    else:
        print("\nNo data found for analysis.")

    print("\nAll processing completed.")
