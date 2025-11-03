#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 KOTOHA ENGINE — Persona Styler v1.4
Refined Persona Mapper（誠実でウィットに富む＋語尾・カテゴリ拡張）
----------------------------------------------------------
入力:  ./output/polished/polished_pro_*.json
出力:  ./output/styled/styled_persona_YYYYMMDD_HHMM.json
"""

import os
import json
import random
import re
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INPUT_DIR = "./output/polished"
OUTPUT_DIR = "./output/styled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ブランドトーン定義 ===
GLOBAL_TONE = {
    "name": "Honest_Wit",
    "style": "誠実でウィットに富む",
    "phrase_blends": {
        "シンプル": "理にかなったシンプルさ",
        "おしゃれ": "上品で知的なおしゃれさ",
        "かわいい": "さりげなく愛らしい",
        "軽い": "軽やかで誠実な仕上がり",
        "便利": "考え抜かれた便利さ",
        "高級": "静かに上質を感じさせる"
    }
}

# === 終止バリエーション ===
ENDING_VARIANTS = [
    "心地よい使い勝手。",
    "誠実なつくり。",
    "上質を感じさせる仕上がり。",
    "長く寄り添う設計。",
    "落ち着きあるデザイン。",
    "丁寧な手触り。",
    "理にかなった構造。",
    "品のある佇まい。",
]

# === テンプレート分類 ===
TONE_TEMPLATES = {
    "tech_male": {
        "keywords": ["充電", "ケーブル", "Apple", "MacBook", "スマート", "ガジェット"],
        "patterns": ["性能を研ぎ澄ませた設計。", "機能美が光る仕上がり。", "洗練された構造。"]
    },
    "fashion_female": {
        "keywords": ["バッグ", "ドレス", "スカート", "かわいい", "ピンク", "レディース"],
        "patterns": ["上品で柔らかな印象。", "軽やかに魅せるデザイン。", "知的で穏やかな華やかさ。"]
    },
    "home_life": {
        "keywords": ["キッチン", "収納", "生活", "インテリア", "家庭"],
        "patterns": ["暮らしに寄り添う設計。", "やさしく整う空間づくり。", "穏やかな日常を支える品質。"]
    },
    "beauty_care": {
        "keywords": ["美容", "ケア", "スキン", "香り", "ボディ", "コスメ"],
        "patterns": ["素肌にやさしい感触。", "自然体の美しさを引き出す。", "上品で穏やかな印象を与える。"]
    },
    "kids_toy": {
        "keywords": ["キッズ", "子供", "おもちゃ", "ベビー", "教育", "ジュニア"],
        "patterns": ["遊びながら学べる構造。", "安心して使える設計。", "笑顔を生むデザイン。"]
    },
    "outdoor": {
        "keywords": ["アウトドア", "キャンプ", "登山", "ハイキング", "防水", "耐久"],
        "patterns": ["自然と調和する設計。", "アクティブな時間を支える品質。", "軽やかに持ち運べる構造。"]
    },
    "office": {
        "keywords": ["ビジネス", "オフィス", "スーツ", "ノート", "会議", "仕事"],
        "patterns": ["知的な印象を与える設計。", "集中を支える静かなデザイン。", "信頼を生む佇まい。"]
    },
    "default": {
        "keywords": [],
        "patterns": ["理にかなったシンプルさ。", "上質な日常使い。", "穏やかに寄り添う仕上がり。"]
    }
}


def guess_persona_tone(text):
    for tone, info in TONE_TEMPLATES.items():
        for kw in info["keywords"]:
            if kw.lower() in text.lower():
                return tone
    return "default"


def apply_global_tone(text):
    for k, v in GLOBAL_TONE["phrase_blends"].items():
        text = text.replace(k, v)
    text = re.sub(r"[!！]+", "", text).strip().rstrip("。")
    text += "。"
    return text


def diversify_alt(alt, patterns):
    """ALT文の軽微差分生成"""
    base = apply_global_tone(alt)
    base += " " + random.choice(patterns)
    if random.random() < 0.4:
        base += " " + random.choice(ENDING_VARIANTS)
    return base


def apply_persona_style(catch, alts, tone_key):
    tone_data = TONE_TEMPLATES[tone_key]
    patterns = tone_data["patterns"]

    new_catch = apply_global_tone(catch)
    if len(new_catch) < 30:
        new_catch += " " + random.choice(patterns)

    new_alts = [diversify_alt(a, patterns) for a in alts]
    while len(new_alts) < 20:
        new_alts.append(diversify_alt("", patterns))

    return new_catch, new_alts[:20]


def find_latest_polished():
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith("polished_pro_") and f.endswith(".json")]
    if not files:
        return None
    return os.path.join(INPUT_DIR, max(files, key=lambda x: os.path.getmtime(os.path.join(INPUT_DIR, x))))


def main():
    input_file = find_latest_polished()
    if not input_file:
        print("🚫 polished_pro_*.json が見つかりません。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    styled = []
    for cluster in tqdm(data, desc="🎨 Persona Refining 中"):
        name = cluster.get("cluster_name", "")
        catch = cluster.get("catch_copy", "")
        alts = cluster.get("alts", [])
        tone = guess_persona_tone(name + catch)
        new_catch, new_alts = apply_persona_style(catch, alts, tone)

        styled.append({
            "cluster_id": cluster.get("cluster_id", ""),
            "persona": tone,
            "catch_copy": new_catch,
            "alts": new_alts,
            "brand_tone": GLOBAL_TONE["name"]
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(OUTPUT_DIR, f"styled_persona_{ts}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(styled, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Persona Styler v1.4 完了: {output_file}")
    print(f"🧩 クラスタ数: {len(styled)}")
    print(f"💫 ブランドトーン: {GLOBAL_TONE['style']}")


if __name__ == "__main__":
    main()
import atlas_autosave_core
