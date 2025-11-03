# ===========================================
# 🌸 KOTOHA ENGINE — Hybrid AI Writer v5.8 Fixed2
# GPT-5仕様準拠（temperature/max_completion_tokens対応）
# ===========================================

import os, json, re, random, datetime
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# ====== 基本設定 ======
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_CSV = "./input.csv"
OUT_JSON = "./output/ai_writer/"
SEM_PATH = "./output/semantics/"

os.makedirs(OUT_JSON, exist_ok=True)

# ====== JSONロード ======
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_knowledge_files():
    files = {
        "cluster": f"{SEM_PATH}lexical_clusters_20251030_223013.json",
        "market": f"{SEM_PATH}market_vocab_20251030_201906.json",
        "semantic": f"{SEM_PATH}structured_semantics_20251030_224846.json",
        "persona": f"{SEM_PATH}styled_persona_20251031_0031.json",
        "normalized": f"{SEM_PATH}normalized_20251031_0039.json",
        "template": f"{SEM_PATH}template_composer.json"
    }
    return {k: load_json(v) for k, v in files.items()}

# ====== クラスタマッチ ======
def find_related_cluster(product_name, clusters):
    for c in clusters:
        if any(k in product_name for k in c.get("keywords", [])):
            return c
    return random.choice(clusters)

# ====== 知見要約 ======
def make_knowledge_block(cluster, market, sem, persona, normalized, template):
    kws = "、".join(cluster.get("keywords", [])[:5])
    trend = ""
    if isinstance(market, dict):
        for v in market.values():
            if isinstance(v, list):
                trend += "、".join(v)
    tone = persona[0].get("tone", "誠実で知的") if isinstance(persona, list) else "誠実で知的"
    forbid = "、".join(normalized[0].get("forbidden_words", [])) if isinstance(normalized, list) else ""
    tmpl = "・".join(template.get("templates", [])[:3]) if isinstance(template, dict) else ""
    concept = "＋".join(sem.get("concepts", [])[:3]) if isinstance(sem, dict) else ""
    return f"""
主要語群：{kws}
市場語：{trend[:120]}
構文指針：{concept}
文体トーン：{tone}
禁止語：{forbid}
テンプレート型：{tmpl}
"""

# ====== AI生成 ======
def ai_generate(product_name, knowledge_block):
    prompt = f"""
あなたは熟練した日本語コピーライターです。
以下の商品について、SEOと自然文を両立した文章を生成してください。

【商品名】{product_name}

出力仕様：
・キャッチコピー（40〜60文字）
・ALTテキスト（80〜110文字 ×20件）

【知見要約】
{knowledge_block}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "あなたは日本語マーケティングライターです。"},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1500  # ← GPT-5仕様に準拠
        )

        txt = res.choices[0].message.content.strip()
        copy_match = re.findall(r'キャッチコピー[:：]?(.*)', txt)
        copy_text = copy_match[0].strip() if copy_match else txt.split("\n")[0][:60]
        alts = re.findall(r'ALT[0-9０-９]?[：:]\s*(.*)', txt)
        alt_texts = [a.strip() for a in alts if len(a) > 0][:20]
        while len(alt_texts) < 20:
            alt_texts.append(f"{product_name} の魅力を伝える商品画像")
        return copy_text, alt_texts

    except Exception as e:
        print(f"⚠️ 生成エラー: {e}")
        return "生成失敗", [f"{product_name} の商品画像" for _ in range(20)]

# ====== メイン ======
def main():
    print("🌸 Hybrid AI Writer v5.8 Fixed2 実行開始（GPT-5仕様対応）")

    df = pd.read_csv(INPUT_CSV, encoding="cp932")
    products = [p for p in df["商品名"].dropna().unique().tolist()]
    print(f"✅ 商品名抽出: {len(products)}件")

    cfg = load_knowledge_files()
    clusters = cfg["cluster"]

    out_records = []
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    for nm in tqdm(products, desc="🧠 商品別AI生成中"):
        cluster = find_related_cluster(nm, clusters)
        kb = make_knowledge_block(cluster, cfg["market"], cfg["semantic"],
                                  cfg["persona"], cfg["normalized"], cfg["template"])
        copy, alts = ai_generate(nm, kb)
        print(f"🧠 {nm[:25]}... → Copy:{len(copy)}字 / ALT:{len(alts)}件")
        out_records.append({"商品名": nm, "キャッチコピー": copy, **{f"ALT{i+1}": alts[i] for i in range(20)}})

    out_json_path = f"{OUT_JSON}hybrid_writer_full_{now}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    out_csv_path = f"{OUT_JSON}hybrid_writer_preview_{now}.csv"
    pd.DataFrame(out_records).to_csv(out_csv_path, encoding="cp932", index=False)

    print(f"✅ 出力完了: {out_json_path}")
    print(f"✅ 目視確認用CSV: {out_csv_path}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
