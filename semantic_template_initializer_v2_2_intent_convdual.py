# -*- coding: utf-8 -*-
"""
semantic_template_initializer_v2_2_intent_convdual.py
------------------------------------------------------
KOTOHA二層構造テンプレート自動生成器（Traffic × Conversion）

- output/semantics 配下に構造JSONをカテゴリ別出力
- 各カテゴリでtraffic層とconversion層を保持
- .envのOPENAI_MODEL, USE_KOTOHA_PERSONAを自動読込
"""

import os
import json
import datetime
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# =============================
# 0️⃣ 依存ライブラリの確認
# =============================
def ensure_module(package: str):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for mod in ["dotenv"]:
    ensure_module(mod)

# =============================
# 1️⃣ 初期設定
# =============================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "semantics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
PERSONA = os.getenv("USE_KOTOHA_PERSONA", "OFF")
FUSION_TRAFFIC = 0.6
FUSION_CONVERSION = 0.4

CATEGORIES = [
    "smartphone",
    "pc",
    "lifestyle",
    "fashion"
]

# =============================
# 2️⃣ ベース構造
# =============================
def base_template(category: str):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    data = {
        "meta": {
            "version": "2.2",
            "generated_at": now,
            "categories": [category],
            "source": {
                "rakuten_api": "IchibaItem/Search/20220601",
                "yahoo_api": "ShoppingWebService/V3/itemSearch"
            },
            "fusion_ratio": {
                "traffic": FUSION_TRAFFIC,
                "conversion": FUSION_CONVERSION
            },
            "description": f"カテゴリ「{category}」向け。流入層×転換層の二層構造。",
            "model": MODEL,
            "persona_engine": PERSONA
        },
        "layer_traffic": {
            "core_keywords": [],
            "feature_terms": [],
            "category_terms": [],
            "brand_terms": [],
            "related_devices": [],
            "technical_specs": []
        },
        "layer_conversion": {
            "intent_words": [],
            "benefit_terms": [],
            "usage_scenes": [],
            "target_personas": [],
            "value_words": [],
            "conversion_triggers": []
        },
        "shared_attributes": {
            "prohibited_words": [
                "画像", "写真", "レビュー", "最安", "No.1", "リンク", "購入はこちら"
            ],
            "sentence_templates": [
                "この{feature}で、{scene}でも{benefit}。",
                "{target}向けに設計された{feature}で、{benefit}を実現。",
                "{scene}にぴったりの{feature}で、{value_word}が魅力。",
                "使う人を選ばない{feature}で、{conversion_trigger}にも最適。"
            ],
            "ending_variations": [
                "です。", "します。", "できます。", "しやすいです。", "がポイント。", "が魅力。"
            ]
        },
        "extension": {
            "language_model_hint": MODEL,
            "persona_profile": PERSONA,
            "semantic_density_score": None
        }
    }
    return data

# =============================
# 3️⃣ ファイル生成
# =============================
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ {path.name}")

def main():
    print("🧩 semantic_template_initializer_v2_2_intent_convdual 起動")
    all_templates = {}

    for cat in CATEGORIES:
        obj = base_template(cat)
        file_path = OUTPUT_DIR / f"structured_semantics_v2_intent_convdual_{cat}.json"
        save_json(obj, file_path)
        all_templates[cat] = obj

    # 統合版
    unified_path = OUTPUT_DIR / "structured_semantics_v2_intent_convdual.json"
    unified = {
        "meta": {
            "version": "2.2",
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": "全カテゴリ統合版テンプレート"
        },
        "categories": all_templates
    }
    save_json(unified, unified_path)

    print("🎯 完了: /output/semantics にカテゴリ別テンプレートを生成しました。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
