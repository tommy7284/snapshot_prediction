import json
import os
import csv
import requests
from typing import List, Set, Dict, Optional, Tuple
import pygit2

# --- 設定 ---
REPOS_DIR = "repos"
JSON_DIR = "collect/jestable"
SNAPSHOT_EXTENSION = ".snap"
# 必要に応じてTokenを設定してください
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "") 

# ★デバッグモードをONに設定（詳細ログが出ます）
DEBUG = True

# --- デバッグ用ヘルパー ---
def log_debug(msg: str, indent: int = 0):
    """デバッグログ出力用"""
    if DEBUG:
        print(" " * indent + f"[DEBUG] {msg}")

def short_oid(oid) -> str:
    """OIDを短縮表示（None対応）"""
    return str(oid)[:7] if oid else "None"

def get_commit_desc(commit: pygit2.Commit) -> str:
    """コミットの簡単な説明（ハッシュ + メッセージ1行目）"""
    msg = commit.message.split('\n')[0][:50]
    return f"{short_oid(commit.id)} ('{msg}')"

# --- コアロジック ---

def get_blob_oid(repo: pygit2.Repository, commit: pygit2.Commit, path: str) -> Optional[pygit2.Oid]:
    """コミットからファイルのOIDを取得"""
    try:
        return commit.tree[path].id
    except (KeyError, ValueError):
        try:
            tree_entry = commit.tree
            for part in path.split('/'):
                tree_entry = repo[tree_entry[part].id]
            return tree_entry.id
        except:
            return None

def is_user_change(repo: pygit2.Repository, commit: pygit2.Commit, path: str) -> bool:
    """
    そのコミットでユーザーが能動的にファイルを変更したかを判定する
    (マージによる自動的な取り込みは除外する)
    """
    indent = 4
    commit_desc = get_commit_desc(commit)
    
    if not commit.parents:
        log_debug(f"Commit {commit_desc}: Root commit (New file)", indent)
        return True

    parent1 = commit.parents[0]
    
    # OID取得（ログ用）
    current_oid = get_blob_oid(repo, commit, path)
    p1_oid = get_blob_oid(repo, parent1, path)

    # 1. Parent1 との比較（Diffチェック）
    # is_user_changeの判定自体はDiffの有無で行うが、
    # ログで見やすくするためにOIDも表示する
    if current_oid == p1_oid:
        log_debug(f"Commit {commit_desc}: No change from Parent 1 ({short_oid(p1_oid)}) -> Skip", indent)
        return False
    
    # 念のためDiffAPIでも確認（ロジックの堅牢性のため）
    diff = repo.diff(parent1, commit)
    found_in_diff = any(d.new_file.path == path for d in diff.deltas)
            
    if not found_in_diff:
        log_debug(f"Commit {commit_desc}: OID changed but not in Diff (Rename/Mode change?) -> Skip", indent)
        return False

    log_debug(f"Commit {commit_desc}: Diff detected vs Parent 1. Checking if Merge...", indent)

    # 2. マージコミットの場合のノイズ除去
    if len(commit.parents) > 1:
        parent2 = commit.parents[1]
        p2_oid = get_blob_oid(repo, parent2, path)

        # 「変更後の状態」が「取り込んだブランチ(Parent2)」と完全に一致する場合
        if current_oid == p2_oid:
            log_debug(f"    -> Merge Sync Detected! Content matches Parent 2 ({short_oid(p2_oid)}).", indent)
            log_debug(f"    -> Ignoring as 'Automatic Sync'.", indent)
            return False
        else:
            log_debug(f"    -> Merge Conflict Resolution or Custom Edit.", indent)
            log_debug(f"    -> Current({short_oid(current_oid)}) != Parent2({short_oid(p2_oid)})", indent)

    log_debug(f"  ✅ User Change CONFIRMED at {commit_desc}", indent)
    return True

def check_reversion(repo: pygit2.Repository,
                    path: str,
                    base_commit: pygit2.Commit,
                    head_commit: pygit2.Commit,
                    pr_commits: List[pygit2.Commit]) -> bool:
    """
    Diffベースの厳密なReversion判定
    """
    log_debug(f"--- Checking Reversion for: {path} ---")
    
    base_oid = get_blob_oid(repo, base_commit, path)
    head_oid = get_blob_oid(repo, head_commit, path)
    
    log_debug(f"Base OID: {short_oid(base_oid)} | Head OID: {short_oid(head_oid)}")
    
    # 条件1: 最終状態(Head)がBaseと同じであること
    if base_oid != head_oid:
        log_debug(f"Result: FALSE (Base != Head). Final state is different.")
        return False
    
    log_debug("Condition 1 Met: Base == Head. Scanning history for changes...")

    # 条件2: PRの過程で、ユーザーによる「能動的な変更」が一度でも行われたか？
    has_user_edit = False
    
    for commit in pr_commits:
        if is_user_change(repo, commit, path):
            has_user_edit = True
            log_debug(f"Result: TRUE (Reversion Detected). Found user edit at {short_oid(commit.id)}.")
            break
    
    if not has_user_edit:
        log_debug("Result: FALSE. No effective user changes found (only Syncs or No-ops).")

    return has_user_edit

# --- その他のヘルパー関数（変更なし） ---

def has_keyword_in_pr_discussion(owner: str, name: str, pr_num: int, keyword: str = "snapshot") -> bool:
    if not GITHUB_TOKEN: return False
    # (省略: 元のコードと同じ実装)
    return True 

def get_pr_commits_first_parent(repo: pygit2.Repository, pr_head_oid: pygit2.Oid, 
                                  base_oid: pygit2.Oid) -> List[pygit2.Commit]:
    """
    PRに含まれるコミットを収集する（修正版：Walker使用）
    
    手動のParent探索をやめ、pygit2のWalkerを使用します。
    walker.hide(base_oid) を使うことで、Baseおよびその祖先（init含む）を
    グラフ構造に基づいて確実に除外します。
    これにより、マージベースがParent 2側にある場合でも、突き抜けるのを防ぎます。
    """
    try:
        # トポロジカル順（子 -> 親）で探索設定
        walker = repo.walk(pr_head_oid, pygit2.GIT_SORT_TOPOLOGICAL)
        
        # 【重要】Baseコミットとその全ての祖先を「隠す（除外する）」
        # これにより、PR固有のコミットだけが残ります。
        walker.hide(base_oid)
        
        commits = list(walker)
        
        # 時系列順（古い -> 新しい）に並べ替える
        # （ロジック上で時系列に沿って処理したほうが分かりやすいため）
        commits.reverse()
        
        if DEBUG:
            # 念のため、initが含まれていないかログで確認可能にする
            if commits:
                first = commits[0]
                log_debug(f"Collected {len(commits)} commits. First: {short_oid(first.id)}, Last: {short_oid(commits[-1].id)}")
            else:
                log_debug("Collected 0 commits (PR matches Base).")

        return commits

    except Exception as e:
        if DEBUG: print(f"[DEBUG] Error in walker: {e}")
        return []

def get_last_non_merge_commit(repo: pygit2.Repository, pr_head_oid: pygit2.Oid, 
                               base_oid: pygit2.Oid) -> Optional[pygit2.Commit]:
    head = repo.get(pr_head_oid)
    if len(head.parents) <= 1: return head
    current = head
    while len(current.parents) > 0:
        if current.id == base_oid: break
        if len(current.parents) == 1: return current
        current = current.parents[0]
    return None

def get_actual_pr_changes(repo: pygit2.Repository,
                          pr_commits: List[pygit2.Commit],
                          base_commit: pygit2.Commit) -> Set[str]:
    pr_changed_files = set()
    for commit in pr_commits:
        if len(commit.parents) > 1: continue 
        diff = repo.diff(base_commit, commit)
        for delta in diff.deltas:
            path = delta.new_file.path
            if path and path.endswith(SNAPSHOT_EXTENSION):
                pr_changed_files.add(path)
    return pr_changed_files
