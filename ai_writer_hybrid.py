# ===============================================
# 🌸 KOTOHA ENGINE — AI Writer (Hybrid Edition)
# Hybrid generation: Cluster + Product-level synthesis
# Author: ChatGPT (KOTOHA ENGINE Dev)
# ===============================================

import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# -----------------------------------------------
# ログ設定
# -----------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# -----------------------------------------------
# 設定・環境読み込み
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "ai_generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("❌ OpenAI APIキーが設定されていません。")
    exit(1)

client = AsyncOpenAI(api_key=api_key)

# -----------------------------------------------
# 既存成果物の読込
# -----------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SEMANTIC_PATH = "./output/semantics/structured_semantics_20251030_224846.json"
VOCAB_PATH = "./output/market_vocab_20251030_201906.json"
CLUSTER_PATH = "./output/lexical_clusters_20251030_223013.json"

structured_semantics = load_json(SEMANTIC_PATH)
market_vocab = load_json(VOCAB_PATH)
lexical_clusters = load_json(CLUSTER_PATH)

logging.info(f"✅ 成果物読込完了: semantics={len(structured_semantics)}, vocab={len(market_vocab)}, clusters={len(lexical_clusters)}")

# -----------------------------------------------
# Hybrid Prompt Generator
# -----------------------------------------------
def build_prompt(cluster, products):
    prompt = [
        {
            "role": "system",
            "content": (
                "あなたは優れた日本語コピーライター兼SEOライターです。"
                "商品情報と語彙データを参考に、購買意欲を高める自然な日本語で"
                "短く明確なコピー文と20件のALTテキストを生成してください。\n\n"
                "⚙️ 制約条件:\n"
                "- キャッチコピーは40〜60文字\n"
                "- ALTは80〜110文字\n"
                "- ALTは1行1文、画像内容への直接言及は禁止（SEOワード中心）\n"
                "- 『！』の多用は禁止\n"
                "- 各ALTは重複禁止・自然な言い換え表現を使用\n"
                "- 出力はJSON形式 {copy: string, alt: [list of strings]}\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"クラスタ代表語: {', '.join(cluster.get('keywords', [])[:10])}\n"
                f"関連商品名の例: {', '.join(p['name'] for p in products[:3])}\n"
                f"関連トピック: {', '.join(p.get('topics', []) for p in products if 'topics' in p)}"
            ),
        },
    ]
    return prompt

# -----------------------------------------------
# OpenAI 呼び出し（バッチ対応）
# -----------------------------------------------
async def generate_for_cluster(cluster_id, cluster_data, product_list):
    prompt = build_prompt(cluster_data, product_list)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.8,
            max_tokens=800,
            timeout=60,
        )
        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
        except Exception:
            logging.warning("⚠️ JSON解析失敗、テンプレ生成にフォールバックします。")
            data = {"copy": "自然な魅力を伝える商品です。", "alt": ["高品質で信頼のアイテムです。"] * 20}
        return {"cluster_id": cluster_id, "output": data}
    except Exception as e:
        logging.error(f"🚫 クラスタ生成失敗: {e}")
        return {"cluster_id": cluster_id, "output": None}

# -----------------------------------------------
# メイン処理
# -----------------------------------------------
async def main():
    start = datetime.now()
    logging.info("🌸 KOTOHA ENGINE — Hybrid AI Writer 起動")

    # 仮: クラスタごとの商品リスト（マッピングは柔軟化可能）
    clusters = market_vocab.get("clusters", market_vocab)  # 両方のフォーマットに対応
    products = [{"name": v.get("name", ""), "topics": v.get("keywords", [])} for v in clusters[:700]]

    tasks = []
    for i, cluster in enumerate(lexical_clusters[:50]):
        subset = products[i*14:(i+1)*14]  # 各クラスタに約14商品割当
        tasks.append(generate_for_cluster(i, cluster, subset))

    results = await tqdm_asyncio.gather(*tasks)
    outfile = os.path.join(OUTPUT_DIR, f"ai_generated_hybrid_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = (datetime.now() - start).seconds
    logging.info(f"✅ Hybrid AI Writer 完了: {len(results)} クラスタ生成, 実行時間: {elapsed}s")
    logging.info(f"💾 出力ファイル: {outfile}")

# -----------------------------------------------
# 実行
# -----------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
import atlas_autosave_core
