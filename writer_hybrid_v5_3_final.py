# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v5.3 FINAL
GPT-5完全対応版: temperature削除 / max_completion_tokens採用
"""

import os
import csv
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# 初期設定
# =========================================================
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
OUT_DIR = os.path.join(BASE_DIR, "output/ai_writer")
os.makedirs(OUT_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ENCODING_IN = "cp932"

PATH_LEXICAL = os.path.join(SEM_DIR, "lexical_clusters_20251030_223013.json")
PATH_MARKET  = os.path.join(SEM_DIR, "market_vocab_20251030_201906.json")
PATH_SEMANT  = os.path.join(SEM_DIR, "structured_semantics_20251030_224846.json")
PATH_PERSONA = os.path.join(SEM_DIR, "styled_persona_20251031_0031.json")
PATH_NORMAL  = os.path.join(SEM_DIR, "normalized_20251031_0039.json")
PATH_TEMPLATE = os.path.join(SEM_DIR, "template_composer.json")

# =========================================================
# ユーティリティ
# =========================================================
def sanitize(s):
    s = s.replace("\u3000", " ")
    return re.sub(r"\s+", " ", s.strip())

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default or {}

def ensure_dict_or_first(json_obj):
    """dict / list 両対応で最初の要素を取得"""
    if isinstance(json_obj, list):
        return json_obj[0] if json_obj else {}
    elif isinstance(json_obj, dict):
        return json_obj
    else:
        return {}

# =========================================================
# AI生成ロジック
# =========================================================
def ai_generate(name, persona_cfg, lexical_cfg, market_cfg, sem_cfg, tmpl_cfg, norm_cfg):
    persona_entry = ensure_dict_or_first(persona_cfg)
    sem_entry     = ensure_dict_or_first(sem_cfg)
    lexical_entry = ensure_dict_or_first(lexical_cfg)
    norm_entry    = ensure_dict_or_first(norm_cfg)

    prompt = f"""
あなたは日本語ECサイト向けの熟練コピーライターです。
以下のルールを必ず守り、与えられたJSON構造を参考にキャッチコピーとALTを生成してください。

# 商品名
{name}

# 文体・トーン（Persona）
{json.dumps(persona_entry, ensure_ascii=False)}

# 市場語彙（Market）
{json.dumps(market_cfg.get('keywords', []), ensure_ascii=False)}

# 意味ネット・特徴（Semantics）
{json.dumps(sem_entry, ensure_ascii=False)}

# 同義語クラスタ（Lexical）
{json.dumps(lexical_entry, ensure_ascii=False)}

# 構文テンプレート（Template）
{json.dumps(tmpl_cfg.get('templates', []), ensure_ascii=False)}

# 禁則語・表記ルール（Normalization）
{json.dumps(norm_entry.get('forbidden_words', []), ensure_ascii=False)}

# 出力条件
- Copy: 40〜60文字
- ALT: 80〜110文字
- 禁止語を含まない
- 読点・句点は自然な日本語
- トーン: 信頼感・明瞭・誇張なし
- ALTは具体的な商品説明を含む（SEOに有利なキーワードを自然に配置）

# 出力形式（JSON）
{{
  "copy": "ここに生成結果",
  "alt": "ここに生成結果"
}}
    """

    res = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "あなたは日本語ECコピー専門のプロフェッショナルライターです。"},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=500  # ✅ GPT-5対応
        # temperature削除（GPT-5では固定値）
    )

    try:
        data = json.loads(res.choices[0].message.content)
        copy_t = data.get("copy", "").strip()
        alt_t = data.get("alt", "").strip()
    except Exception:
        text = res.choices[0].message.content.strip()
        copy_t, alt_t = text.split("\n", 1) if "\n" in text else (text, text)
    return copy_t, alt_t

# =========================================================
# メイン処理
# =========================================================
def main():
    print("🌸 Hybrid AI Writer v5.3 FINAL 実行開始（GPT-5完全対応）")

    lexical_cfg = load_json(PATH_LEXICAL)
    market_cfg  = load_json(PATH_MARKET)
    sem_cfg     = load_json(PATH_SEMANT)
    persona_cfg = load_json(PATH_PERSONA)
    tmpl_cfg    = load_json(PATH_TEMPLATE)
    norm_cfg    = load_json(PATH_NORMAL)

    with open(INPUT_CSV, "r", encoding=ENCODING_IN, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("⚠️ CSVが空です。")
        return

    header = rows[0]
    try:
        name_idx = header.index("商品名")
    except ValueError:
        raise RuntimeError("⚠️ ヘッダに『商品名』列が見つかりません。")

    names = [sanitize(r[name_idx]) for r in rows[1:] if len(r) > name_idx and sanitize(r[name_idx])]
    unique_names = list(dict.fromkeys(names))
    print(f"✅ 商品名抽出: {len(names)}件 → 一意化後 {len(unique_names)}件")

    results = []
    csv_rows = [["商品名", "キャッチコピー", "商品画像名（ALT）1"]]

    for nm in unique_names:
        print(f"🧠 AI生成中: {nm[:30]}...")
        copy_t, alt_t = ai_generate(nm, persona_cfg, lexical_cfg, market_cfg, sem_cfg, tmpl_cfg, norm_cfg)
        results.append({
            "product_name": nm,
            "copy": copy_t,
            "alt": alt_t
        })
        csv_rows.append([nm, copy_t, alt_t])

    now = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join(OUT_DIR, f"hybrid_writer_full_{now}.json")
    csv_path = os.path.join(OUT_DIR, f"hybrid_writer_preview_{now}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"💾 出力完了: {json_path}")
    print(f"🧾 目視用CSV: {csv_path}")
    print(f"📊 件数: {len(results)}（全件AI生成）")
    print("📏 Copy 40–60 / ALT 80–110 / 禁則・句読点適用済")

# =========================================================
if __name__ == "__main__":
    main()
import atlas_autosave_core
