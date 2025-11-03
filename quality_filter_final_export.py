#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 KOTOHA ENGINE — Quality Filter & Final Export v1.0
Naturalness Scoring + CSV Export Integrator
---------------------------------------------------------
入力:  ./output/normalized/normalized_*.json
出力:  ./output/final/output_final_YYYYMMDD_HHMM.csv
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import unicodedata
from tqdm import tqdm

INPUT_JSON_DIR = "./output/normalized"
INPUT_CSV = "./input.csv"
OUTPUT_DIR = "./output/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === スコアリング設定 ===
def naturalness_score(text: str) -> float:
    if not text:
        return 0
    text = unicodedata.normalize("NFKC", text)
    length = len(text)
    unique_ratio = len(set(text)) / max(1, len(text))
    has_comma = "、" in text or "," in text
    balance = 1.0 - abs(length - 80) / 80  # ALT基準80字
    score = 0.4 * unique_ratio + 0.4 * balance + (0.2 if has_comma else 0)
    return round(max(0, min(score, 1.0)), 3)

# === 補完句パターン ===
COMPLETION_PHRASES = [
    "上質な使い心地を届けます。",
    "理にかなった美しさです。",
    "毎日を支える機能です。",
    "暮らしに寄り添う設計です。",
    "細部にまで誠実さが宿ります。"
]

def complete_sentence(text: str) -> str:
    text = text.strip()
    if len(text) < 40:
        text += " " + np.random.choice(COMPLETION_PHRASES)
    if not text.endswith("。"):
        text += "。"
    return text

# === 最新JSON探索 ===
def find_latest_normalized():
    files = [f for f in os.listdir(INPUT_JSON_DIR) if f.startswith("normalized_") and f.endswith(".json")]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(INPUT_JSON_DIR, f)))
    return os.path.join(INPUT_JSON_DIR, latest)

def main():
    json_file = find_latest_normalized()
    if not json_file:
        print("🚫 normalized_*.json が見つかりません。")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    print(f"📘 読み込み: {json_file}")
    print(f"📊 クラスタ数: {len(clusters)}")

    # === スコアリング＋補完 ===
    refined = []
    for c in tqdm(clusters, desc="✨ Scoring & Refining"):
        c["catch_copy_score"] = naturalness_score(c.get("catch_copy", ""))
        c["alts_scores"] = [naturalness_score(a) for a in c.get("alts", [])]
        c["catch_copy"] = complete_sentence(c.get("catch_copy", ""))

        # 低スコアALT除去＋補完
        alts = [a for a, s in zip(c["alts"], c["alts_scores"]) if s > 0.3]
        while len(alts) < 20:
            alts.append(alts[-1] if alts else "上質な仕上がりです。")
        c["alts"] = alts[:20]
        refined.append(c)

    # === DataFrame化 for CSV Export ===
    df = pd.DataFrame(refined)
    alt_cols = [f"ALT{i+1}" for i in range(20)]
    df_csv = pd.DataFrame({
        "商品名": [f"Cluster_{c['cluster_id']}" for c in refined],
        "キャッチコピー": [c["catch_copy"] for c in refined],
        **{alt_cols[i]: [c["alts"][i] for c in refined] for i in range(20)},
        "自然度スコア": [round(np.mean(c["alts_scores"]), 3) for c in refined]
    })

    # === 出力 ===
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_out = os.path.join(OUTPUT_DIR, f"filtered_normalized_{ts}.json")
    csv_out = os.path.join(OUTPUT_DIR, f"output_final_{ts}.csv")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    df_csv.to_csv(csv_out, index=False, encoding="cp932")

    print(f"\n✅ Final Export 完了!")
    print(f"📄 JSON: {json_out}")
    print(f"📊 CSV : {csv_out}")
    print(f"🧩 商品群: {len(refined)} 件")


if __name__ == "__main__":
    main()
import atlas_autosave_core
