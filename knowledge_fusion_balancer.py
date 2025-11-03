# -*- coding: utf-8 -*-
"""
knowledge_fusion_balancer.py
─────────────────────────────
目的：
  /output/semantics 配下の各知見JSONを読み込み、
  語彙量のバランスを取りながら「構造的自然文」を生成する。

出力：
  ./output/semantics/knowledge_fused_structured.json

特徴：
  - lexical, market, semantics, template, persona の5カテゴリを統合
  - 各カテゴリに重みを付けて自然文構成
  - forbidden_words も集約
  - v5系ライターで直接使用可能（knowledge_text, forbidden_words）
"""

import os, json, glob, random

BASE_DIR = os.path.dirname(__file__)
SEM_DIR = os.path.join(BASE_DIR, "output", "semantics")
OUT_PATH = os.path.join(SEM_DIR, "knowledge_fused_structured.json")

# =====================
# 重み付け（自然文志向）
# =====================
WEIGHTS = {
    "lexical": 0.20,
    "market": 0.20,
    "semantic": 0.25,
    "template": 0.25,
    "persona": 0.10,
}

# =====================
# ファイル検出と分類
# =====================
def detect_jsons():
    files = glob.glob(os.path.join(SEM_DIR, "*.json"))
    mapping = {"lexical": [], "market": [], "semantic": [], "template": [], "persona": [], "normalized": []}
    for f in files:
        name = os.path.basename(f).lower()
        if "lexical" in name: mapping["lexical"].append(f)
        elif "market" in name: mapping["market"].append(f)
        elif "semantic" in name: mapping["semantic"].append(f)
        elif "template" in name: mapping["template"].append(f)
        elif "persona" in name: mapping["persona"].append(f)
        elif "normal" in name: mapping["normalized"].append(f)
    return mapping

# =====================
# JSON 読込ユーティリティ
# =====================
def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def flatten_terms(data):
    """構造がどんな形でも単語列をゆるく抽出"""
    if isinstance(data, dict):
        vals = []
        for v in data.values():
            vals.extend(flatten_terms(v))
        return vals
    elif isinstance(data, list):
        vals = []
        for x in data:
            vals.extend(flatten_terms(x))
        return vals
    elif isinstance(data, str):
        return [data.strip()]
    else:
        return []

# =====================
# 自然文テンプレート群
# =====================
TEMPLATES = [
    "このカテゴリでは{0}や{1}が中心で、{2}などに対応します。",
    "{0}を搭載し、{1}シーンで活躍する設計です。",
    "{0}に適した構造で、{1}向けに開発されています。",
    "日常利用に加え、{0}や{1}など多様な場面で活用できます。",
    "全体として{0}と{1}を両立し、{2}な印象を与えます。",
    "{0}や{1}を備えた実用的なデザインが特徴です。",
    "高い{0}と{1}を兼ね備え、{2}にも対応する構成です。",
    "{0}を意識した設計で、{1}や{2}でも快適に使用できます。"
]

# =====================
# 自然文生成コア
# =====================
def build_sentence(terms):
    """語彙群から自然文を生成"""
    if not terms:
        return ""
    # ランダムにサンプリングして自然構成
    t = random.sample(terms, min(len(terms), 5))
    t += ["多機能", "デザイン性", "実用性", "快適さ"]
    temp = random.choice(TEMPLATES)
    try:
        return temp.format(*t[:temp.count("{")])
    except Exception:
        return "、".join(t[:5]) + "の特徴を備えています。"

# =====================
# 知見統合ロジック
# =====================
def fuse_knowledge(mapping):
    combined = []
    for key, files in mapping.items():
        if key == "normalized":
            continue
        all_terms = []
        for f in files:
            data = load_json_safe(f)
            if data:
                all_terms += flatten_terms(data)
        if not all_terms:
            continue

        weight = WEIGHTS.get(key, 0.1)
        n_sent = max(1, int(weight * 10))  # 重みに応じて文数
        for _ in range(n_sent):
            sentence = build_sentence(all_terms)
            if sentence and sentence not in combined:
                combined.append(sentence)

    return combined

# =====================
# 禁則語抽出
# =====================
def collect_forbidden(mapping):
    words = []
    for f in mapping.get("normalized", []):
        data = load_json_safe(f)
        if isinstance(data, dict):
            words += data.get("forbidden_words", [])
    return sorted(list(set(words)))

# =====================
# メイン処理
# =====================
def main():
    print("🧩 KOTOHA知見バランサー起動中…")
    mapping = detect_jsons()
    knowledge_lines = fuse_knowledge(mapping)
    forbidden_words = collect_forbidden(mapping)

    result = {
        "knowledge_text": knowledge_lines,
        "forbidden_words": forbidden_words
    }

    os.makedirs(SEM_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 出力完了: {OUT_PATH}")
    print(f"📘 知見文数: {len(knowledge_lines)} / 禁則語数: {len(forbidden_words)}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
