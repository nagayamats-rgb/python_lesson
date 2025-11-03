import csv, json, re, os, time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ======== 初期設定 ========
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INPUT_CSV = "rakuten.csv"
OUTPUT_CSV = "./output/ai_writer/alt_text_refined_final_revived_SEOplus_longform.csv"
ENCODING = "utf-8"

SEMANTIC_DIR = "./output/semantics"
MAX_RETRY = 3
SLEEP_BETWEEN = 3

# ======== ローカルJSON読込 ========
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

persona = load_json(f"{SEMANTIC_DIR}/styled_persona_20251031_0031.json")
lexical = load_json(f"{SEMANTIC_DIR}/lexical_clusters_20251030_223013.json")
market = load_json(f"{SEMANTIC_DIR}/market_vocab_20251030_201906.json")
semantic = load_json(f"{SEMANTIC_DIR}/structured_semantics_20251030_224846.json")
norm = load_json(f"{SEMANTIC_DIR}/normalized_20251031_0039.json")

# ======== ユーティリティ ========
def clean_text(text):
    """句読点・禁則・重複除去"""
    if not text:
        return ""
    text = re.sub(r"[\"'\n\r]", "", text)
    text = re.sub(r"　+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(。){2,}", "。", text)
    text = text.strip()
    return text

def trim_to_range_natural(txt, min_len=100, max_len=120):
    """文末優先で自然な長さに整形"""
    txt = txt.strip()
    if len(txt) <= max_len:
        return txt
    cut = txt[:max_len]
    last_p = max(cut.rfind("。"), cut.rfind("、"))
    if last_p >= min_len:
        return cut[:last_p+1]
    return cut[:max_len]

def summarize_keywords():
    """SEOキーワードを抽出"""
    kws = []
    for source in [market, lexical]:
        if isinstance(source, list):
            for v in source[:15]:
                if isinstance(v, dict):
                    for k in v.values():
                        if isinstance(k, str):
                            kws.append(k)
        elif isinstance(source, dict):
            kws += [v for v in source.values() if isinstance(v, str)]
    kws = list(set(kws))
    return "、".join(kws[:30])

SEO_KEYWORDS = summarize_keywords()

# ======== AI呼び出し ========
def ai_generate_alts(product_name):
    prompt = f"""
あなたはSEOライティングに長けた日本語のプロフェッショナルです。
以下の条件でALTテキストを20件生成してください。

【目的】
・楽天の商品画像に設定するALTテキストとして使用します。
・検索エンジン最適化（SEO）効果を最大化します。

【指示】
・各文は自然で読みやすく、情報量のある日本語で。
・1文で完結し、理想は120〜140字。最低でも100字以上にしてください。
・商品名、主要機能、対応機種、素材、用途を2〜3回自然に含める。
・SEOキーワード候補を自然に散りばめる。
・禁止語：「最安」「No.1」「競合」「他社」「画像」「写真」「見た目」「商品画像」「映える」などは禁止。
・絵文字、特殊記号、タグは禁止。
・句点「。」で文を終える。

【SEOキーワード候補】
{SEO_KEYWORDS}

【商品名】
{product_name}

出力形式:
20個の日本語文をJSON配列として出力。
"""

    for attempt in range(MAX_RETRY):
        try:
            res = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "あなたはSEOに最適化された日本語ライターです。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
    max_completion_tokens=900,
    response_format="text"
)
text = res.choices[0].message.content
alts = re.findall(r"[^。]+。", text)[:20]
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception as e:
            print(f"⚠️ 生成エラー({attempt+1}/{MAX_RETRY}): {e}")
            time.sleep(SLEEP_BETWEEN)
    return []

# ======== メイン処理 ========
def main():
    with open(INPUT_CSV, "r", encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        products = [r["商品名"] for r in reader if r.get("商品名")]

    print(f"🌸 ALT生成開始（SEO強化＋長文モード）")
    print(f"✅ 対象商品数: {len(products)}件")

    results = []
    for nm in tqdm(products, desc="🧠 生成中"):
        alts = ai_generate_alts(nm)
        alts_cleaned = [trim_to_range_natural(clean_text(a), 100, 120) for a in alts]
        row = {"商品名": nm}
        for i, a in enumerate(alts_cleaned[:20]):
            row[f"ALT{i+1}"] = a
        results.append(row)
        time.sleep(1.5)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["商品名"] + [f"ALT{i}" for i in range(1, 21)])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 出力完了: {OUTPUT_CSV}")
    print("✅ 各ALTはSEO語を含む自然文で120〜140字理想、句点整形済。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
