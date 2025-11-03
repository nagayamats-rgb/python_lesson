#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 KOTOHA ENGINE — SEO Grammar Normalizer v1.0
Natural Language Refinement + SEO Keyword Harmonizer
---------------------------------------------------------
入力:  ./output/styled/styled_persona_*.json
出力:  ./output/normalized/normalized_YYYYMMDD_HHMM.json
"""

import os
import json
import re
import unicodedata
from datetime import datetime
from tqdm import tqdm

INPUT_DIR = "./output/styled"
OUTPUT_DIR = "./output/normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 共通パターン定義 ===
REPLACE_PATTERNS = {
    "マグセーフ": "MagSafe",
    "アイフォン": "iPhone",
    "スマホ": "スマートフォン",
    "おしゃれ": "上品で知的なデザイン",
    "かわいい": "さりげなく愛らしい",
    "高級感": "静かに上質を感じさせる",
    "シンプル": "理にかなったシンプルさ",
    "便利": "考え抜かれた便利さ",
    "軽量": "軽やかな軽さ",
}

# === 文末語尾バリエーション ===
ENDINGS = [
    "。", "。", "。",  # 通常率を高めて安定
    "。上質を支える構造。",
    "。誠実に仕上げた設計。",
    "。静かな佇まいで魅せる。",
    "。丁寧なものづくり。",
    "。長く愛される品質。",
]

# === 重複・冗長語を整形 ===
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[!！]+", "", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"(\s{2,}|　+)", " ", text)
    text = text.strip()
    return text

# === SEO語彙統一 ===
def harmonize_keywords(text: str) -> str:
    for k, v in REPLACE_PATTERNS.items():
        text = text.replace(k, v)
    return text

# === 自然語構文へ変換 ===
def normalize_sentence(text: str) -> str:
    text = clean_text(text)
    text = harmonize_keywords(text)
    if not text.endswith("。"):
        text += "。"
    if len(text) < 40 and "。" in text:
        text += " " + re.sub(r"^。", "", text)  # 軽い自然文連結
    if random_chance := hash(text) % len(ENDINGS):
        text = re.sub(r"。$", ENDINGS[random_chance], text)
    return text

def find_latest_styled():
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith("styled_persona_") and f.endswith(".json")]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(INPUT_DIR, f)))
    return os.path.join(INPUT_DIR, latest)

def main():
    input_file = find_latest_styled()
    if not input_file:
        print("🚫 styled_persona_*.json が見つかりません。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    refined = []
    for cluster in tqdm(data, desc="🪄 Grammar Normalizing"):
        new_catch = normalize_sentence(cluster.get("catch_copy", ""))
        new_alts = [normalize_sentence(a) for a in cluster.get("alts", [])]

        refined.append({
            "cluster_id": cluster.get("cluster_id"),
            "persona": cluster.get("persona"),
            "brand_tone": cluster.get("brand_tone"),
            "catch_copy": new_catch,
            "alts": new_alts
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(OUTPUT_DIR, f"normalized_{ts}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    print(f"\n✅ SEO Grammar Normalizer 完了: {output_file}")
    print(f"📊 クラスタ数: {len(refined)}")
    print("💡 次は Quality Filter で自然度スコアリングへ。")


if __name__ == "__main__":
    main()
import atlas_autosave_core
