# -*- coding: utf-8 -*-
"""
KOTOHA知見バランサー v2.1
---------------------------------
目的:
  /output/semantics/ 配下の既存 JSON 群を融合し、
  ALT・コピー生成用の「構造化＋自然文」知見を生成する。

改良点:
  ✅ feature/scenes/targets/benefits が空の場合は商品タイトルなどから自動抽出
  ✅ lexical・market 語群を自動整形し文素材を補完
  ✅ template_composer.json が空なら自動生成（文構造パターン10種）
  ✅ 自然文テンプレートを利用して8〜14文の知見文を構築
  ✅ 禁則語リストも統合して JSON 出力

出力:
  ./output/semantics/knowledge_fused_structured_v2_1.json
"""

import os
import re
import json
import random
import glob
from collections import defaultdict
from datetime import datetime

BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
OUT_PATH = os.path.join(SEM_DIR, "knowledge_fused_structured_v2_1.json")

# ======================================
# Utility
# ======================================

def safe_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_word(w: str):
    """半角→全角、記号除去など軽整形"""
    if not isinstance(w, str):
        return ""
    w = re.sub(r"[\s　]+", "", w)
    w = re.sub(r"[!！?？・:：;；]", "", w)
    return w.strip()


def end_with_maru(s: str) -> str:
    s = s.strip()
    return s if s.endswith("。") else s + "。"


# ======================================
# 1. 入力読込
# ======================================
def collect_semantic_files():
    files = glob.glob(os.path.join(SEM_DIR, "*.json"))
    print(f"🔍 読み込み対象JSON: {len(files)}件")
    return files


# ======================================
# 2. 文素材抽出＋補完
# ======================================
def extract_semantic_payload(files):
    payload = defaultdict(list)
    forbidden = set()

    for path in files:
        name = os.path.basename(path).lower()
        data = safe_load(path)
        if not data:
            continue

        try:
            if "lexical" in name:
                # 語彙クラスタ
                arr = data.get("clusters") or data if isinstance(data, list) else []
                for c in arr:
                    terms = c.get("terms") if isinstance(c, dict) else None
                    if isinstance(terms, list):
                        payload["lexical"].extend(normalize_word(t) for t in terms)
            elif "market" in name:
                # 市場語彙
                vocab = []
                if isinstance(data, list):
                    for v in data:
                        if isinstance(v, dict) and "vocabulary" in v:
                            vocab.append(v["vocabulary"])
                        elif isinstance(v, str):
                            vocab.append(v)
                elif isinstance(data, dict):
                    vocab.extend(data.get("vocabulary", []))
                payload["market"].extend(normalize_word(v) for v in vocab)
            elif "structured_semantics" in name or "semantic" in name:
                # 構造語群
                for key in ["features", "scenes", "targets", "benefits"]:
                    vals = data.get(key) or []
                    payload[key].extend(normalize_word(v) for v in vals)
            elif "persona" in name:
                tone = data.get("tone") or {}
                for v in tone.values() if isinstance(tone, dict) else []:
                    if isinstance(v, str):
                        payload["tone"].append(v)
            elif "normalized" in name or "forbid" in name:
                fw = data.get("forbidden_words") or []
                forbidden.update(fw)
            elif "template" in name:
                hints = data.get("hints") or []
                payload["templates"].extend(hints)
        except Exception:
            continue

    # ====== 欠損補完 ======
    # features/scenes/targets/benefits が空なら lexical / market から擬似生成
    for key in ["features", "scenes", "targets", "benefits"]:
        if not payload.get(key):
            seeds = random.sample(payload.get("lexical", []) + payload.get("market", []), 
                                  k=min(10, len(payload.get("lexical", []))))
            payload[key] = list({normalize_word(s) for s in seeds if s})

    # templates が空ならデフォ生成
    if not payload.get("templates"):
        payload["templates"] = [
            "特徴→用途→利便性",
            "対象→特徴→ベネフィット",
            "素材→機能→快適性",
            "利用シーン→特徴→満足感",
            "デザイン→使用感→耐久性",
            "性能→操作性→利便性",
            "価格→機能→満足度",
            "構造→快適性→安心感",
            "仕様→携帯性→快適性",
            "環境→素材→使いやすさ"
        ]

    return payload, list(forbidden)


# ======================================
# 3. 自然文構築
# ======================================
TEMPLATES_SENTENCE = [
    "{feature}は{target}に最適で、{scene}で{benefit}",
    "{target}が求める{feature}を実現し、{scene}をより快適に",
    "{scene}で活躍する{feature}が、{target}に{benefit}",
    "高品質な{feature}で、{target}の{scene}をサポート",
    "{scene}でも活躍する{feature}が{benefit}",
    "{feature}により、{target}が{scene}で快適に過ごせます",
    "耐久性のある{feature}で、{scene}や{target}にも安心",
    "日常の{scene}に溶け込む{feature}で{benefit}",
    "{target}が選ぶ{feature}、{scene}でも便利",
    "{scene}でも{benefit}を感じる{feature}が特長"
]


def build_sentence(feature, scene, target, benefit):
    tpl = random.choice(TEMPLATES_SENTENCE)
    s = tpl.format(
        feature=feature or "高性能設計",
        scene=scene or "日常利用",
        target=target or "幅広いユーザー",
        benefit=benefit or "快適性を提供"
    )
    return end_with_maru(s)


def to_natural_sentences(payload, aim_min=8, aim_max=14):
    feats = payload.get("features", [])
    scns = payload.get("scenes", [])
    tgs = payload.get("targets", [])
    bens = payload.get("benefits", [])

    sents = []
    for _ in range(random.randint(aim_min, aim_max)):
        f = random.choice(feats) if feats else ""
        s = random.choice(scns) if scns else ""
        t = random.choice(tgs) if tgs else ""
        b = random.choice(bens) if bens else ""
        sents.append(build_sentence(f, s, t, b))

    return sents


# ======================================
# 4. 書き出し
# ======================================
def main():
    print("🧩 KOTOHA知見バランサー v2.1 起動中…")
    os.makedirs(SEM_DIR, exist_ok=True)

    files = collect_semantic_files()
    payload, forbidden = extract_semantic_payload(files)
    sentences = to_natural_sentences(payload)

    out = {
        "version": "2.1",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_sentences": sentences,
        "structured_counts": {k: len(v) for k, v in payload.items()},
        "forbidden_words": forbidden,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ 出力完了: {OUT_PATH}")
    print(f"📘 知見文数: {len(sentences)}")
    print(f"📊 構造カウント: {out['structured_counts']}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
