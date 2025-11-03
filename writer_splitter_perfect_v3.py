# ============================================
# 🌸 writer_splitter_perfect_v3_1.py
# 安定版：空応答・壊れたJSON・再試行保護付き
# ============================================

import json, csv, os, re, time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

MODEL = "gpt-5"
MAX_TOKENS = 800
OUTPUT_DIR = "./output/ai_writer"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_CSV = "./input.csv"
RETRY_WAIT = 5

SEED_JSONS = {
    "persona": "./output/semantics/styled_persona_20251031_0031.json",
    "lexical": "./output/semantics/lexical_clusters_20251030_223013.json",
    "market": "./output/semantics/market_vocab_20251030_201906.json",
    "semantic": "./output/semantics/structured_semantics_20251030_224846.json",
    "template": "./output/semantics/template_composer.json",
    "norm": "./output/semantics/normalized_20251031_0039.json"
}

client = OpenAI()

def load_jsons():
    cfg = {}
    for k, path in SEED_JSONS.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    cfg[k] = json.load(f)
                except:
                    cfg[k] = {}
        else:
            cfg[k] = {}
    return cfg

def summarize_knowledge(cfg):
    persona = ", ".join([v.get("tone", "") for v in cfg["persona"] if isinstance(v, dict)])
    lexical = ", ".join([v.get("keyword", "") for v in cfg["lexical"] if isinstance(v, dict)])
    market = ", ".join([v.get("vocabulary", "") for v in cfg["market"] if isinstance(v, dict)])
    semantic = ", ".join([v.get("concept", "") for v in cfg["semantic"] if isinstance(v, dict)])
    template = ", ".join([v.get("pattern", "") for v in cfg["template"] if isinstance(v, dict)])
    norm = ", ".join([v.get("forbidden_words", "") for v in cfg["norm"] if isinstance(v, dict)])
    return f"""
【ローカル知見要約】
- トーン: {persona}
- 市場語彙: {market}
- 概念群: {semantic}
- 構文パターン: {template}
- 禁止語: {norm}
- 代表キーワード: {lexical}
"""

def call_openai_json(messages, retries=3):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=MAX_TOKENS,
                messages=messages
            )
            content = res.choices[0].message.content
            if not content or not content.strip():
                print(f"⚠️ 空応答を検出（{attempt+1}/{retries}）→再試行中…")
                time.sleep(RETRY_WAIT)
                # knowledgeが長すぎる場合は短縮して再送
                for msg in messages:
                    if msg["role"] == "user" and len(msg["content"]) > 1500:
                        msg["content"] = msg["content"][:1000] + "\n（要約短縮版）"
                continue

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                if "{" in content and "}" in content:
                    data = json.loads(content[:content.rfind("}")+1])
                else:
                    print(f"⚠️ JSON変換失敗（{attempt+1}/{retries}）→再試行")
                    time.sleep(RETRY_WAIT)
                    continue
            return data
        except Exception as e:
            print(f"⚠️ OpenAIエラー: {e}（{attempt+1}/{retries}）")
            time.sleep(RETRY_WAIT)
    print("❌ 応答取得失敗（すべての再試行が失敗）")
    return None

def ai_generate(product_name, knowledge):
    system_prompt = f"""
あなたはEC向けコピーライターです。
以下の知見と商品名をもとに、楽天とYahoo向けキャッチコピー、そしてSEOに最適化されたALT20件を生成してください。

出力フォーマット(JSON):
{{
  "rakuten_copy": "全角60〜80文字、上限87文字",
  "yahoo_copy": "全角25〜30文字、上限30文字",
  "alt_texts": ["80〜110文字×20"]
}}

構文ガイド:
- 商品スペック(spec)
- コアコンピタンス(competence)
- どんな人が(user)
- どんなシーンで(scene)
- 使うとどう便利・困りごと解決(benefit)
この構成要素で自然に文章を作成しなさい。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"商品名: {product_name}\n{knowledge}"}
    ]

    data = call_openai_json(messages)
    if not data:
        return "", "", [""] * 20

    rakuten = data.get("rakuten_copy", "")[:87]
    yahoo = data.get("yahoo_copy", "")[:30]
    alts = data.get("alt_texts", [])
    if not isinstance(alts, list):
        alts = [""] * 20
    elif len(alts) < 20:
        alts += [""] * (20 - len(alts))
    return rakuten, yahoo, alts[:20]

def extract_names():
    names = []
    with open(INPUT_CSV, "r", encoding="shift_jis", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("商品名", "").strip()
            if name:
                names.append(name)
    names = list(dict.fromkeys(names))
    return names

def main():
    print("🌸 writer_splitter_perfect_v3.1 実行開始（安定モード＋再試行保護）")
    cfg = load_jsons()
    knowledge = summarize_knowledge(cfg)
    names = extract_names()
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")

    results = []
    for nm in tqdm(names, desc="🧠 商品別AI生成中"):
        rak, yah, alts = ai_generate(nm, knowledge)
        results.append({
            "product_name": nm,
            "rakuten_copy": rak,
            "yahoo_copy": yah,
            "alt_texts": alts
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = os.path.join(OUTPUT_DIR, f"split_full_{timestamp}")
    jsonl_path = f"{base}.jsonl"
    csv_path_r = f"{OUTPUT_DIR}/rakuten_copy_{timestamp}.csv"
    csv_path_y = f"{OUTPUT_DIR}/yahoo_copy_{timestamp}.csv"
    csv_path_alt = f"{OUTPUT_DIR}/alt_text_{timestamp}.csv"

    # JSONL出力
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for item in results:
            jf.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 楽天・Yahoo
    with open(csv_path_r, "w", encoding="utf-8-sig", newline="") as rf, \
         open(csv_path_y, "w", encoding="utf-8-sig", newline="") as yf, \
         open(csv_path_alt, "w", encoding="utf-8-sig", newline="") as af:

        writer_r = csv.writer(rf)
        writer_y = csv.writer(yf)
        writer_a = csv.writer(af)

        writer_r.writerow(["商品名", "楽天コピー"])
        writer_y.writerow(["商品名", "Yahooコピー"])
        writer_a.writerow(["商品名"] + [f"ALT{i}" for i in range(1, 21)])

        for item in results:
            writer_r.writerow([item["product_name"], item["rakuten_copy"]])
            writer_y.writerow([item["product_name"], item["yahoo_copy"]])
            writer_a.writerow([item["product_name"]] + item["alt_texts"])

    print(f"✅ 出力完了:\n   - 楽天: {csv_path_r}\n   - Yahoo: {csv_path_y}\n   - ALT20: {csv_path_alt}\n   - JSONL: {jsonl_path}")
    print("✅ 共通ALT20は『alt_text_*.csv』に全商品ぶんを横持ちで書き出します。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
