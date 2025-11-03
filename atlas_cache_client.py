# -*- coding: utf-8 -*-
"""
Atlas Cache Client v1.0
---------------------------------------
Atlas Session Splitterで作成された分割スナップショットを遅延読み込みし、
ライターやAIモジュールに動的注入します。
"""

import json
from pathlib import Path
from typing import Dict, Any

BASE = Path("/Users/tsuyoshi/Desktop/python_lesson")
INDEX_PATH = BASE / "config/atlas_session_index.json"

def _load_json(p: Path) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_context(kind: str) -> Dict[str, Any]:
    """persona, dev, ops のいずれかを読み込む"""
    idx = _load_json(INDEX_PATH)
    path = idx.get("paths", {}).get(kind)
    if not path:
        return {}
    return _load_json(Path(path))

def attach_atlas_context(payload: Dict[str, Any], wants=("persona","dev")) -> Dict[str, Any]:
    """任意のpayloadにAtlas文脈を注入"""
    merged = dict(payload)
    for k in wants:
        ctx = load_context(k)
        if not ctx:
            continue
        if k == "persona":
            merged["kotoha_persona"] = ctx
        elif k == "dev":
            merged["atlas_dev"] = ctx
        elif k == "ops":
            merged["atlas_ops"] = ctx
        else:
            merged[f"atlas_{k}"] = ctx
    return merged