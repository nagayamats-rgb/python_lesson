# -*- coding: utf-8 -*-
"""
freeze_manager_extended.py
KOTOHA 凍結管理要（かんなめ） - コード凍結・環境記録ユーティリティ

目的:
  - 実行されたスクリプトを自動スナップショット化し、後から完全再現できるようにする。
  - コード破損・仕様逸脱を防ぐ。
  - ログ・環境情報・ハッシュを自動記録。

動作:
  1. auto_freeze_on_start(__file__, note="...") を呼ぶだけで、
     frozen_versions/<script_name>/timestamp/ に3ファイルを作成する。
  2. 生成物:
       - <script_name>.py … 実行コードそのまま
       - meta.json … 実行時環境・ハッシュ・補足メモ
       - stamp.txt … 日時・ユーザー名・ノート等の簡易記録

再現:
  - 凍結ディレクトリから任意バージョンをコピーすれば、当時のコードで再実行可能。

互換性:
  - Python 3.8〜3.12
  - 外部依存なし（標準ライブラリのみ）
"""

import os
import json
import hashlib
import platform
from datetime import datetime

def _sha256_of_file(path: str) -> str:
    """ファイルのSHA256ハッシュを返す"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return "unavailable"

def _safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_", ".")).rstrip()

def auto_freeze_on_start(file_path: str, note: str = ""):
    """
    スクリプト起動時に凍結スナップショットを作成
    """
    try:
        base_name = os.path.basename(file_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(
            os.path.dirname(file_path),
            "frozen_versions",
            os.path.splitext(base_name)[0],
            stamp,
        )
        os.makedirs(folder, exist_ok=True)

        # コードコピー
        dest_code = os.path.join(folder, base_name)
        with open(file_path, "r", encoding="utf-8") as src, open(dest_code, "w", encoding="utf-8") as dst:
            dst.write(src.read())

        # メタ情報
        meta = {
            "script": base_name,
            "timestamp": stamp,
            "note": note,
            "sha256": _sha256_of_file(file_path),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            "cwd": os.getcwd(),
        }
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 簡易スタンプ
        stamp_path = os.path.join(folder, "stamp.txt")
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(f"KOTOHA 凍結管理要（かんなめ）\n")
            f.write(f"実行日時: {stamp}\n")
            f.write(f"スクリプト: {base_name}\n")
            if note:
                f.write(f"ノート: {note}\n")
            f.write(f"環境: {platform.platform()} / Python {platform.python_version()}\n")
            f.write(f"ユーザー: {meta['user']}\n")
            f.write(f"ワークディレクトリ: {meta['cwd']}\n")
            f.write(f"SHA256: {meta['sha256']}\n")

        print(f"🔒 凍結完了: {folder}")
    except Exception as e:
        print(f"⚠️ 凍結処理で例外が発生しました（継続します）: {e}")
import atlas_autosave_core
