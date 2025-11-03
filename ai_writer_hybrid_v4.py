# ============================================
# 🌸 KOTOHA ENGINE — Hybrid AI Writer v4.1
# 商品単位完全生成 + AI最適化 + 進捗修正版
# ============================================

import os, json, random, re, time, logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ==========
# 設定
# ==========
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = "./output/ai_writer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEMANTICS_PATH = "./output/semantics/structured_semantics_20251030_224846.json"
VOCAB_PATH = "./output/market_vocab_20251030_201906.json"
CLUSTER_PATH = "./output/lexical_clusters_20251030_223013.json"

ALT_COUNT = 20
COPY_RANGE = (40, 60)
ALT_RANGE = (80, 110)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# ==========
# 関数群
# ==========
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} が見つかりません。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    return text.replace("！", "。").replace("!", "。")

def is_valid_length(text, min_len, max_len):
    return min_len <= len(text) <= max_len

def remove_invalid_specs(text, valid_keywords):
    for w in re.findall(r"[A-Za-z0-9]+\w*", text):
        if w not in valid_keywords and re.search(r"\d", w):
            text = text.replace(w, "")
    return text

def decide_ai_usage(vocab_density, category_entropy):
    base = 0.3
    if vocab_density < 0.5:
        base += 0.3
    if category_entropy < 0.4:
        base += 0.1
    return random.random() < min(base, 0.8)

def ai_generate_copy_alt(product_name, keywords, context):
    prompt = f"""
あなたはSEOに強い日本語コピーライターです。
商品名「{product_name}」に基づいて、以下の語彙群を活かし、
自然で魅力的なキャッチコピー（40～60字）とALT文（各80～110字）20本を生成してください。

語彙群: {', '.join(keywords[:15])}
文体: 誠実で知的、ウィットに富む
出力形式:
{{
  "copy": "〜",
  "alt": ["〜","〜",...20件]
}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("copy", ""), data.get("alt", [])
    except Exception as e:
        logging.warning(f"⚠️ AI生成失敗: {e}")
        copy = f"{product_name} — 高品質で信頼のある一品。"
        alt = [f"{product_name} の魅力を伝える高解像度画像"] * ALT_COUNT
        return copy, alt

# ==========
# メイン処理
# ==========
def main():
    start = time.time()
    logging.info("🌸 Hybrid AI Writer v4.1 起動 — 商品単位完全生成モード")

    semantics = load_json(SEMANTICS_PATH)
    vocab = load_json(VOCAB_PATH)
    clusters = load_json(CLUSTER_PATH)

    results = []
    ai_calls, tmpl_uses = 0, 0

    for idx, v in enumerate(tqdm(vocab, desc="🪄 商品生成中", total=len(vocab))):
        # 構造正規化
        if isinstance(v, dict):
            name = v.get("name") or v.get("商品名") or "無題商品"
            keywords = v.get("keywords") or v.get("語彙") or []
        elif isinstance(v, list):
            name = v[0] if v else "無題商品"
            keywords = v[1:] if len(v) > 1 else []
        else:
            name = str(v)
            keywords = []

        context = clusters[idx % len(clusters)]
        valid_words = [w for w in keywords if len(w) > 1]

        vocab_density = len(valid_words) / 50
        category_entropy = random.random()  # ダミー。実際はクラスタ分散率などで計算
        use_ai = decide_ai_usage(vocab_density, category_entropy)

        if use_ai:
            copy, alt = ai_generate_copy_alt(name, keywords, context)
            ai_calls += 1
        else:
            tmpl_uses += 1
            copy = clean_text(f"{name} — {random.choice(['高性能', '新登場', '快適な使用感', '信頼の品質'])}を実現。")
            alt = [
                clean_text(f"{name} {random.choice(['高耐久', '多機能', '軽量設計', 'スタイリッシュ'])}で使いやすいデザイン。")
                for _ in range(ALT_COUNT)
            ]

        copy = clean_text(remove_invalid_specs(copy, valid_words))
        alt = [clean_text(remove_invalid_specs(a, valid_words)) for a in alt]

        if not is_valid_length(copy, *COPY_RANGE):
            copy = copy[:COPY_RANGE[1]]
        alt = [a[:ALT_RANGE[1]] if not is_valid_length(a, *ALT_RANGE) else a for a in alt]

        results.append({"product_id": idx + 1, "name": name, "copy": copy, "alt": alt})

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"{OUTPUT_DIR}/hybrid_writer_full_{ts}.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results)
    avg_alt = sum(len(r["alt"]) for r in results) / total
    logging.info(f"✅ 出力完了: {output_path}")
    logging.info(f"📊 商品数={total} / Copy長平均={sum(len(r['copy']) for r in results)//total} / ALT数平均={avg_alt}")
    logging.info(f"🤖 AI生成={ai_calls}件 / テンプレ展開={tmpl_uses}件")
    logging.info(f"⏱ 実行時間: {time.time() - start:.1f}s")

# ==========
if __name__ == "__main__":
    main()
import atlas_autosave_core
