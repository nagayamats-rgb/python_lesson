# -*- coding: utf-8 -*-
"""
atlas_local_cache_bridge.py
────────────────────────────────────────
Atlas ↔ Python ローカルキャッシュ・ブリッジ（拡張）
- セッション状態（モデル/進捗/ログ）に加え、
  ・KOTOHA人格（kotoha_persona.json）
  ・知見バランサー成果（knowledge_fused_structured_v2_1.json）
  を自動ロード＋セッションへ統合。
- TTL管理・フラッシュ・イベント追記・スナップショット対応。
- どのスクリプトからも import して使える軽量API。
"""

from __future__ import annotations
import os
import json
import time
import sys
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from datetime import datetime

# ====== 固定パス（要として記憶・全コードで共通）======
BASE_DIR   = "/Users/tsuyoshi/Desktop/python_lesson"
CONFIG_DIR = os.path.join(BASE_DIR, "config")
OUT_DIR    = os.path.join(BASE_DIR, "output")
SEM_DIR    = os.path.join(OUT_DIR, "semantics")

CACHE_FILE = os.path.join(CONFIG_DIR, "atlas_session_cache.json")
CACHE_TTL_HOURS = 72

# KOTOHA人格（生成済）
PERSONA_FILE = os.path.join(CONFIG_DIR, "kotoha_persona.json")

# 知見バランサー成果（v2.1で生成）
FUSED_KNOWLEDGE_FILE = os.path.join(SEM_DIR, "knowledge_fused_structured_v2_1.json")

# 将来の拡張に備えて明示
CSV_RAKUTEN = "/Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv"
CSV_YAHOO   = "/Users/tsuyoshi/Desktop/python_lesson/sauce/yahoo.csv"
FMT_RAKUTEN = "/Users/tsuyoshi/Desktop/python_lesson/sauce/Rakuten_Format.csv"
FMT_YAHOO   = "/Users/tsuyoshi/Desktop/python_lesson/sauce/YAHOO_Format.csv"


# ====== ユーティリティ ======
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def _now_ts() -> int:
    return int(time.time())

def _expired(ts: int, hours: int = CACHE_TTL_HOURS) -> bool:
    return (_now_ts() - int(ts)) > hours * 3600

def _safe_load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _atomic_write_json(path: str, data: Any):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(tmp, path)


# ====== データ構造 ======
@dataclass
class PersonaState:
    enabled: bool = True         # 人格エンジンを使うか
    version: str = "KOTOHA_v1.0" # 任意の表示用
    payload: Dict[str, Any] = None  # kotoha_persona.json の中身

@dataclass
class KnowledgeState:
    fused_loaded: bool = False
    fused_payload: Dict[str, Any] = None
    sources: List[str] = None  # どのファイルから構築したかの痕跡

@dataclass
class SessionContext:
    phase: str = "INIT"
    current_model: str = ""        # 例: gpt-5-turbo
    openai_mode: str = "chat"      # chat/responses
    temperature: float = 1.0
    notes: str = ""
    events: List[Dict[str, Any]] = None

@dataclass
class AtlasCache:
    _timestamp: int
    _saved_at: str
    session: SessionContext
    persona: PersonaState
    knowledge: KnowledgeState


# ====== ローダー ======
def _load_persona() -> PersonaState:
    data = _safe_load_json(PERSONA_FILE) or {}
    return PersonaState(
        enabled=True,
        version=data.get("version") or "KOTOHA_v1.0",
        payload=data
    )

def _load_knowledge() -> KnowledgeState:
    fused = _safe_load_json(FUSED_KNOWLEDGE_FILE) or {}
    loaded = bool(fused)
    sources = []
    if loaded:
        sources.append(os.path.basename(FUSED_KNOWLEDGE_FILE))
    return KnowledgeState(
        fused_loaded=loaded,
        fused_payload=fused if loaded else {},
        sources=sources
    )

def _new_cache(context_hint: Optional[Dict[str, Any]] = None) -> AtlasCache:
    # 初期セッション
    sess = SessionContext(
        phase = (context_hint or {}).get("phase", "ALT生成フェーズ"),
        current_model = (context_hint or {}).get("current_model", os.getenv("OPENAI_MODEL", "")),
        openai_mode = os.getenv("OPENAI_MODE", "chat"),
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "1.0")),
        notes = "初期化",
        events = []
    )
    persona = _load_persona()
    knowledge = _load_knowledge()
    return AtlasCache(
        _timestamp=_now_ts(),
        _saved_at=datetime.now().isoformat(),
        session=sess,
        persona=persona,
        knowledge=knowledge
    )


# ====== パブリックAPI ======
def load_cache(auto_refresh: bool = True) -> AtlasCache:
    """
    キャッシュを読み込む。存在しない/期限切れなら新規生成。
    """
    _ensure_dir(CONFIG_DIR)
    data = _safe_load_json(CACHE_FILE)
    if not data or (auto_refresh and _expired(data.get("_timestamp", 0))):
        cache = _new_cache()
        save_cache(cache)
        print("🕒 キャッシュ初期化（新規）")
        return cache

    try:
        # 復元
        session = SessionContext(**(data.get("session") or {}))
        persona = PersonaState(**(data.get("persona") or {}))
        knowledge = KnowledgeState(**(data.get("knowledge") or {}))
        cache = AtlasCache(
            _timestamp=data.get("_timestamp", _now_ts()),
            _saved_at=data.get("_saved_at", datetime.now().isoformat()),
            session=session,
            persona=persona,
            knowledge=knowledge
        )
        print(f"✅ キャッシュ読込: {CACHE_FILE}")
        return cache
    except Exception:
        # 壊れていたら作り直す
        cache = _new_cache()
        save_cache(cache)
        print("⚠️ キャッシュ破損 → 再生成")
        return cache

def save_cache(cache: AtlasCache):
    payload = {
        "_timestamp": _now_ts(),
        "_saved_at": datetime.now().isoformat(),
        "session": asdict(cache.session),
        "persona": asdict(cache.persona),
        "knowledge": asdict(cache.knowledge),
    }
    _ensure_dir(CONFIG_DIR)
    _atomic_write_json(CACHE_FILE, payload)
    # 率直なフィードバック
    print(f"💾 キャッシュ保存: {CACHE_FILE}")

def flush_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("🧹 キャッシュ削除完了")
    else:
        print("ℹ️ キャッシュなし")

def set_persona_enabled(cache: AtlasCache, enabled: bool):
    cache.persona.enabled = enabled
    save_cache(cache)

def update_session(cache: AtlasCache, **kwargs):
    """
    例:
      update_session(cache, phase="ALT生成", current_model="gpt-5-turbo")
    """
    for k, v in kwargs.items():
        if hasattr(cache.session, k):
            setattr(cache.session, k, v)
    save_cache(cache)

def record_event(cache: AtlasCache, tag: str, detail: Dict[str, Any]):
    if cache.session.events is None:
        cache.session.events = []
    cache.session.events.append({
        "ts": datetime.now().isoformat(),
        "tag": tag,
        "detail": detail
    })
    # 直ちに保存（障害時も追跡できるように）
    save_cache(cache)

def snapshot_paths() -> Dict[str, str]:
    """ 他モジュールが参照できる基礎パス群 """
    return {
        "BASE_DIR": BASE_DIR,
        "CONFIG_DIR": CONFIG_DIR,
        "OUT_DIR": OUT_DIR,
        "SEM_DIR": SEM_DIR,
        "CACHE_FILE": CACHE_FILE,
        "PERSONA_FILE": PERSONA_FILE,
        "FUSED_KNOWLEDGE_FILE": FUSED_KNOWLEDGE_FILE,
        "CSV_RAKUTEN": CSV_RAKUTEN,
        "CSV_YAHOO": CSV_YAHOO,
        "FMT_RAKUTEN": FMT_RAKUTEN,
        "FMT_YAHOO": FMT_YAHOO,
    }


# ====== CLI ======
def _print(cache: AtlasCache):
    obj = {
        "_timestamp": cache._timestamp,
        "_saved_at": cache._saved_at,
        "session": asdict(cache.session),
        "persona": {
            "enabled": cache.persona.enabled,
            "version": cache.persona.version,
            "payload_keys": list((cache.persona.payload or {}).keys())
        },
        "knowledge": {
            "fused_loaded": cache.knowledge.fused_loaded,
            "source_files": cache.knowledge.sources or [],
            # 中身は大きいので鍵だけ
            "payload_keys": list((cache.knowledge.fused_payload or {}).keys())
        }
    }
    print(json.dumps(obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 使い方:
    #   python atlas_local_cache_bridge.py           # 読み込み（無ければ新規作成）
    #   python atlas_local_cache_bridge.py --flush   # 削除
    #   python atlas_local_cache_bridge.py --disable-persona / --enable-persona
    if "--flush" in sys.argv:
        flush_cache()
        sys.exit(0)

    cache = load_cache(auto_refresh=True)

    if "--disable-persona" in sys.argv:
        set_persona_enabled(cache, False)
        print("🔕 人格エンジン: OFF")
        sys.exit(0)
    if "--enable-persona" in sys.argv:
        set_persona_enabled(cache, True)
        print("🔔 人格エンジン: ON")
        sys.exit(0)

    _print(cache)
import atlas_autosave_core
