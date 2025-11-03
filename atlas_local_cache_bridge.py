# -*- coding: utf-8 -*-
"""
atlas_local_cache_bridge.py
──────────────────────────────
Atlas ↔ Python 双方向キャッシュブリッジ
 - AI処理の知見・設定・履歴をローカルに保存・再利用
 - TTL付きセッションキャッシュ（デフォルト72時間）
 - CLI引数 --flush でキャッシュ削除
"""

import os
import json
import time
import sys
from datetime import datetime, timedelta

# === 定数設定 ===
CACHE_DIR = "/Users/tsuyoshi/Desktop/python_lesson/config"
CACHE_FILE = os.path.join(CACHE_DIR, "atlas_session_cache.json")
CACHE_TTL_HOURS = 72  # キャッシュ寿命

# === 共通ユーティリティ ===
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def now_ts():
    return int(time.time())

def expired(ts):
    """TTL判定"""
    return now_ts() - ts > CACHE_TTL_HOURS * 3600

# === キャッシュ操作 ===
def load_cache():
    """キャッシュ読み込み"""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if expired(data.get("_timestamp", 0)):
            print("🕒 キャッシュ期限切れ → 新規生成します。")
            os.remove(CACHE_FILE)
            return None
        print(f"✅ キャッシュ読込: {CACHE_FILE}")
        return data
    except Exception as e:
        print(f"⚠️ キャッシュ読込エラー: {e}")
        return None

def save_cache(context: dict):
    """キャッシュ保存"""
    ensure_dir(CACHE_DIR)
    payload = {
        "_timestamp": now_ts(),
        "_saved_at": datetime.now().isoformat(),
        "context": context,
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"💾 キャッシュ保存: {CACHE_FILE}")
    except Exception as e:
        print(f"⚠️ キャッシュ保存失敗: {e}")

def flush_cache():
    """キャッシュ削除"""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("🧹 キャッシュ削除完了。")
    else:
        print("⚠️ キャッシュファイルは存在しません。")

# === CLI動作 ===
def main():
    if "--flush" in sys.argv:
        flush_cache()
        return

    cache = load_cache()
    if cache:
        print(json.dumps(cache["context"], ensure_ascii=False, indent=2))
    else:
        # テスト用ダミーコンテキスト
        demo_context = {
            "phase": "ALT生成フェーズ",
            "current_model": "gpt-5-turbo",
            "persona_engine": "KOTOHA_v1.0",
            "knowledge_loaded": True,
            "notes": "ローカルキャッシュ初期化テスト完了"
        }
        save_cache(demo_context)

if __name__ == "__main__":
    main()
import atlas_autosave_core
