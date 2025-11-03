# -*- coding: utf-8 -*-
"""
Atlas Timeline AutoCommit v3
-----------------------------------
🧭 目的:
Atlasの「時系列キャッシュ」(atlas_timeline.json) と「現行スナップショット」(atlas_session_cache.json)
を自動的にGitHubリポジトリへcommit・pushし、知的作業履歴を恒久保存する。

依存:
- git CLI（Mac標準インストールでOK）
- .env に GIT_REPO_PATH と GIT_BRANCH を記載しておく

.env例:
GIT_REPO_PATH=/Users/tsuyoshi/Desktop/python_lesson
GIT_BRANCH=main
"""

import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# === 設定読込 ===
load_dotenv(override=True)

REPO_PATH = os.getenv("GIT_REPO_PATH", "/Users/tsuyoshi/Desktop/python_lesson").strip()
BRANCH = os.getenv("GIT_BRANCH", "main").strip()

FILES_TO_COMMIT = [
    "config/atlas_timeline.json",
    "config/atlas_session_cache.json",
]

# === 関数群 ===
def run(cmd, cwd=None):
    """コマンド実行（エラー出力含む）"""
    result = subprocess.run(cmd, cwd=cwd, text=True, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"⚠️ コマンド失敗: {cmd}")
        print(result.stderr)
    return result.stdout.strip()


def ensure_repo_clean():
    """Gitリポジトリ状態確認"""
    status = run("git status --porcelain", cwd=REPO_PATH)
    if status:
        print("📦 変更があります。コミットを準備します。")
        return True
    else:
        print("✅ 変更なし。スキップします。")
        return False


def commit_and_push():
    """自動コミット＋プッシュ"""
    print("🚀 GitHub AutoCommit 実行中...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # add
    for f in FILES_TO_COMMIT:
        full_path = os.path.join(REPO_PATH, f)
        if os.path.exists(full_path):
            run(f"git add {f}", cwd=REPO_PATH)
        else:
            print(f"⚠️ ファイル未検出: {f}")

    # commit & push
    msg = f"🧭 Atlas timeline auto-update ({now})"
    run(f'git commit -m "{msg}"', cwd=REPO_PATH)
    run(f"git push origin {BRANCH}", cwd=REPO_PATH)
    print("✅ GitHubへ自動バックアップ完了。")


# === メイン ===
if __name__ == "__main__":
    print("🌐 Atlas Timeline AutoCommit v3 起動中...")
    print(f"📁 リポジトリ: {REPO_PATH}")
    print(f"🌿 ブランチ: {BRANCH}")

    if ensure_repo_clean():
        commit_and_push()
    else:
        print("⏸️ 自動コミットをスキップしました。")
import atlas_autosave_core
