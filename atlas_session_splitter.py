# -*- coding: utf-8 -*-
"""
Atlas Session Splitter v1.0
---------------------------------------
KOTOHA / Atlas の巨大セッションを3分割 (persona / dev / ops)
構造:
  config/atlas_session_cache.json → 各種スナップショット生成
  config/atlas_session_index.json にインデックス登録
環境変数 (.env):
  ATLAS_SPLIT_CACHE, ATLAS_SNAPSHOT_DIR, ATLAS_MAX_SNAPSHOTS
"""

import os, json, time, shutil, glob
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# =========================
# 設定とユーティリティ
# =========================
BASE = Path("/Users/tsuyoshi/Desktop/python_lesson")
CFG_DIR = BASE / "config"
SESSION_PATH = CFG_DIR / "atlas_session_cache.json"
TIMELINE_PATH = CFG_DIR / "atlas_timeline.json"
INDEX_PATH = CFG_DIR / "atlas_session_index.json"

load_dotenv(BASE / ".env")

def getenv(key, default=""):
    v = os.getenv(key, "").strip()
    return v if v else default

def load_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# セッション分割ロジック
# =========================
def pick_context_blocks(session_obj: dict):
    """Atlasセッションを persona / dev / ops に分離"""
    persona_keys = {"persona_engine", "kotoha", "values", "style", "ethos", "tone"}
    dev_keys     = {"current_model", "knowledge_loaded", "notes", "modules", "freeze", "router", "writer"}
    ops_keys     = {"phase", "files", "runs", "metrics", "snapshots"}

    persona, dev, ops = {}, {}, {}

    for k, v in session_obj.items():
        k_l = str(k).lower()
        if k in persona_keys or "persona" in k_l or "kotoha" in k_l:
            persona[k] = v
        elif k in dev_keys or any(s in k_l for s in ["writer","router","module","prompt","semantic","json","kb"]):
            dev[k] = v
        elif k in ops_keys or any(s in k_l for s in ["phase","run","stat","timeline","cache","atlas"]):
            ops[k] = v
        else:
            ops[k] = v

    return persona, dev, ops

# =========================
# メイン処理
# =========================
def main():
    split_on = getenv("ATLAS_SPLIT_CACHE", "OFF").upper() == "ON"
    snap_dir = Path(getenv("ATLAS_SNAPSHOT_DIR", str(CFG_DIR / "atlas_snapshots")))
    max_keep = int(getenv("ATLAS_MAX_SNAPSHOTS", "12") or "12")

    if not split_on:
        print("🔕 ATLAS_SPLIT_CACHE=OFF（何もしません）")
        return

    session = load_json(SESSION_PATH)
    timeline = load_json(TIMELINE_PATH)
    persona, dev, ops = pick_context_blocks(session)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = snap_dir / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    dump_json(outdir / "persona.json", persona)
    dump_json(outdir / "dev.json", dev)
    dump_json(outdir / "ops.json", ops)

    index = {
        "active_stamp": stamp,
        "paths": {
            "persona": str(outdir / "persona.json"),
            "dev": str(outdir / "dev.json"),
            "ops": str(outdir / "ops.json"),
        },
        "meta": {
            "snapshot_root": str(snap_dir),
            "max_keep": max_keep,
            "timeline_hint": timeline.get("summary", {}) if isinstance(timeline, dict) else {}
        }
    }
    dump_json(INDEX_PATH, index)

    # 古いスナップショット削除
    snaps = sorted(glob.glob(str(snap_dir / "*")), reverse=True)
    for old in snaps[max_keep:]:
        try:
            shutil.rmtree(old)
        except Exception:
            pass

    print("✅ Atlas 分割スナップ完了")
    print(f"📁 最新スナップショット: {outdir}")
    print(f"🧭 インデックス: {INDEX_PATH}")

if __name__ == "__main__":
    main()