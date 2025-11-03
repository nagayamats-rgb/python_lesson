# -*- coding: utf-8 -*-
"""
Atlas Local Cache Bridge v2
-----------------------------------
🧠 Time-aware Cache（時間を意識するセッション記憶）

目的:
- Atlasのキャッシュを単一スナップショットから時系列構造へ拡張
- 各フェーズ（ALT生成・知見構築など）ごとに履歴を蓄積
- KOTOHA人格・モデル設定・実行中スクリプト情報なども自動記録

出力:
  /Users/tsuyoshi/Desktop/python_lesson/config/atlas_session_cache.json     （現行スナップショット）
  /Users/tsuyoshi/Desktop/python_lesson/config/atlas_timeline.json           （時系列履歴）
"""

import os
import json
from datetime import datetime

# === 設定 ===
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson/config"
CACHE_PATH = os.path.join(BASE_DIR, "atlas_session_cache.json")
TIMELINE_PATH = os.path.join(BASE_DIR, "atlas_timeline.json")

# === 初期データ ===
DEFAULT_STATE = {
    "phase": "未定義フェーズ",
    "current_model": "gpt-5-turbo",
    "persona_engine": "KOTOHA_v1.0",
    "knowledge_loaded": False,
    "notes": "セッション初期化",
}


def ensure_dir(path):
    """ディレクトリが存在しない場合は作成"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_snapshot(state: dict):
    """現在状態をキャッシュ＋履歴に保存"""
    ensure_dir(TIMELINE_PATH)
    now = datetime.now().isoformat(timespec="seconds")

    # キャッシュ書き込み
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 履歴読み込み
    if os.path.exists(TIMELINE_PATH):
        with open(TIMELINE_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"timeline": []}

    # 履歴追記
    entry = {"timestamp": now, **state}
    history["timeline"].append(entry)
    history["current_snapshot"] = now

    # 保存
    with open(TIMELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"🕓 スナップショット保存: {now}")
    print(f"📁 現在のフェーズ: {state.get('phase', '不明')} | モデル: {state.get('current_model', '-')}")


def load_timeline():
    """履歴を読み込む"""
    if not os.path.exists(TIMELINE_PATH):
        print("⚠️ 履歴が存在しません。初回セッションかもしれません。")
        return {"timeline": []}
    with open(TIMELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def show_timeline(limit=5):
    """最近のスナップショットを表示"""
    data = load_timeline()
    timeline = data.get("timeline", [])
    if not timeline:
        print("📭 記録なし")
        return
    print(f"🧭 最新 {limit} 件の履歴:")
    for e in timeline[-limit:]:
        ts = e.get("timestamp", "-")
        ph = e.get("phase", "-")
        model = e.get("current_model", "-")
        note = e.get("notes", "")
        print(f"  {ts} | {ph} | {model} | {note}")


# === メイン ===
if __name__ == "__main__":
    print("🧠 Atlas Local Cache Bridge v2 起動中...")
    ensure_dir(CACHE_PATH)

    # 現在状態を仮設定（テスト用）
    current_state = {
        "phase": "知見再構築フェーズ",
        "current_model": "gpt-5-turbo",
        "persona_engine": "KOTOHA_v1.0",
        "knowledge_loaded": True,
        "notes": "semantic_extractor_rebuilder_v1_1_unified 実行中",
    }

    save_snapshot(current_state)
    show_timeline(limit=3)
import atlas_autosave_core
