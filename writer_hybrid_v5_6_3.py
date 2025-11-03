# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v5.6.3
自然文プロンプト・柔軟出力モード（GPT-5対応）
"""

import os, csv, json, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

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

def sanitize(s):
    return re.sub(r"\s+", " ", s.replace("\u3000", " ").strip())

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def extract_copy_and_alts(text):
    """copyとALTを柔軟抽出"""
    text = text.replace("：", ":").replace("・", " ")
    copy_match = re.search(r"copy[:：]\s*(.+)", text, re.IGNORECASE)
    copy = copy_match.group(1).strip() if copy_match else ""
    # ALT抽出（番号付き・箇条書き対応）
    alts = re.findall(r"ALT\d*[:：]?\s*(.+)", text)
    if not alts:
        alts = re.findall(r"[-・]\s*(.{20,120})", text)
    alts = [a.strip() for a in alts if len(a.strip()) > 20]
    return copy[:60], alts[:20]

def ai_generate(name, forbidden):
    prompt = f"""
あなたはECサイトの日本語コピーライターです。
以下の商品について、
魅力的でSEO的にも効果的なコピーとALT説明文を作成してください。

【条件】
・キャッチコピー：40〜60文字程度、日本語で自然で心を惹く表現。
・ALT説明文：80〜110文字程度を20個。多様で重複しない内容に。
・禁止語：{", ".join(forbidden)}

【出力例】
copy: 優れた放熱性で急速充電を実現するスマート充電器
ALT1: シンプルでスタイリッシュなデザインのマグセーフ対応充電器です...
ALT2: 高速かつ安定した充電を実現し、日常使いに最適な充電スタンド...
…
商品名: {name}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"system","content":"あなたは日本語ECコピー専門家です。"},
                      {"role":"user","content":prompt}],
            max_completion_tokens=900
        )
        raw = res.choices[0].message.content or ""
        copy, alts = extract_copy_and_alts(raw)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        copy, alts = "", []

    if not copy:
        copy = "上質な使い心地を追求した人気の定番アイテム"
    while len(alts) < 20:
        alts.append("")
    return copy, alts

def main():
    print("🌸 Hybrid AI Writer v5.6.3 実行開始（自然文プロンプトモード）")

    norm = load_json(PATH_NORMAL)
    forbidden = []
    if isinstance(norm, dict):
        forbidden = norm.get("forbidden_words", [])
    elif isinstance(norm, list):
        forbidden = norm

    with open(INPUT_CSV, "r", encoding=ENCODING_IN) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    name_idx = header.index("商品名")
    names = [sanitize(r[name_idx]) for r in rows[1:] if len(r) > name_idx and sanitize(r[name_idx])]
    uniq = list(dict.fromkeys(names))
    print(f"✅ 商品名抽出: {len(names)}件 → 一意化後 {len(uniq)}件")

    results = []
    csv_rows = [["商品名", "キャッチコピー"] + [f"商品画像名（ALT）{i}" for i in range(1, 21)]]

    for nm in uniq:
        print(f"🧠 生成中: {nm[:30]}...")
        copy, alts = ai_generate(nm, forbidden)
        print(f"   ├ copy:{len(copy)}字 / alts:{sum(1 for a in alts if a)}件 例:{(alts[0] or '')[:25]}…")
        results.append({"product_name": nm, "copy": copy, "alts": alts})
        csv_rows.append([nm, copy] + alts)

    now = datetime.now().strftime("%Y%m%d_%H%M")
    jpath = os.path.join(OUT_DIR, f"hybrid_writer_full_{now}.json")
    cpath = os.path.join(OUT_DIR, f"hybrid_writer_preview_{now}.csv")

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)
    with open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"💾 出力完了: {jpath}")
    print(f"🧾 目視用CSV: {cpath}")
    print(f"📊 件数: {len(results)}（全件AI生成／ALT20件統合）")

if __name__ == "__main__":
    main()
import atlas_autosave_core
