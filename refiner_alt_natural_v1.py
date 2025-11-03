import os
import json
import csv
import re
import random
from pathlib import Path
from collections import Counter

# ========= 基本設定 =========
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_FILE = f"{BASE_DIR}/output/ai_writer/alt_text_refined_final_stable2_fixed.csv"
SEMANTICS_DIR = f"{BASE_DIR}/output/semantics"
OUTPUT_FILE = f"{BASE_DIR}/output/ai_writer/alt_text_refined_final_natural.csv"
DIFF_REPORT = f"{BASE_DIR}/output/ai_writer/alt_text_refined_diff_report.csv"

MIN_LEN = 80
MAX_LEN = 110

# ========= 同義語辞書（初期） =========
SYNONYMS = {
    "特徴": ["魅力", "強み", "こだわり", "特長"],
    "活かし": ["引き出し", "際立たせ", "高め", "表現し"],
    "高品質": ["上質", "高精度", "耐久性", "プレミアム"],
    "使いやす": ["扱いやす", "快適", "スムーズ", "シンプル"],
    "デザイン": ["フォルム", "設計", "スタイル"],
    "便利": ["快適", "スムーズ", "効率的", "使い勝手の良い"],
    "軽量": ["コンパクト", "持ち運びやすい", "携帯性に優れた"],
    "性能": ["機能", "スペック", "パフォーマンス"],
    "シーン": ["場面", "状況", "ライフスタイル"],
}

# ========= 禁則語 =========
FORBIDDEN = [
    "競合", "優位性", "他社", "写真", "画像", "映える", "見た目", 
    "〜のように見える", "比べて", "比較", "ランキング", "No.1", "レビュー"
]

# ========= ユーティリティ関数 =========
def load_semantics():
    vocab = set()
    for file in Path(SEMANTICS_DIR).glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for d in data:
                        for key in ["keywords", "vocabulary", "concepts"]:
                            if isinstance(d, dict) and key in d:
                                if isinstance(d[key], list):
                                    vocab.update(d[key])
                                elif isinstance(d[key], str):
                                    vocab.add(d[key])
        except Exception:
            continue
    return list(vocab)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.replace("。。", "。")
    text = re.sub(r"、{2,}", "、", text)
    for word in FORBIDDEN:
        text = text.replace(word, "")
    return text.strip()

def synonym_replace(text, vocab):
    for k, vals in SYNONYMS.items():
        if k in text:
            rep = random.choice(vals)
            text = text.replace(k, rep)
    for v in random.sample(vocab, min(5, len(vocab))):
        if v not in text and random.random() < 0.05:
            text += f"、{v}"
    return text

def trim_sentence(text):
    if len(text) > MAX_LEN:
        cut = text[:MAX_LEN]
        cut = re.sub(r"。[^。]*$", "。", cut)  # 文途中切断防止
        return cut
    if len(text) < MIN_LEN:
        text += "。"  # 短文補完
    return text

# ========= メイン処理 =========
def main():
    vocab = load_semantics()
    results = []
    diffs = []
    total_before, total_after = 0, 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_row = row.copy()
            for col in row.keys():
                if "ALT" in col and row[col].strip():
                    before = row[col].strip()
                    after = clean_text(before)
                    after = synonym_replace(after, vocab)
                    after = trim_sentence(after)
                    new_row[col] = after
                    total_before += len(before)
                    total_after += len(after)
                    if before != after:
                        diffs.append({
                            "列名": col,
                            "変更前": before,
                            "変更後": after
                        })
            results.append(new_row)

    # 出力1: 自然化ALT本体
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # 出力2: 差分レポート
    with open(DIFF_REPORT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["列名", "変更前", "変更後"])
        writer.writeheader()
        writer.writerows(diffs)

    avg_before = total_before / max(1, len(diffs))
    avg_after = total_after / max(1, len(diffs))
    print(f"✅ 出力完了: {OUTPUT_FILE}")
    print(f"✅ 差分レポート: {DIFF_REPORT}")
    print(f"平均文字数 Before: {avg_before:.1f} → After: {avg_after:.1f}")
    print(f"修正件数: {len(diffs)} / {len(results)} 件")

# ========= 実行 =========
if __name__ == "__main__":
    main()
import atlas_autosave_core
