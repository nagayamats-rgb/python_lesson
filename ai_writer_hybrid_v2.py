import os, json, asyncio
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from loguru import logger
import aiofiles

# === 初期化 ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OpenAI APIキーが設定されていません。")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === ルートと出力設定 ===
OUTPUT_DIR = "./output/ai_writer"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./output/intermediate", exist_ok=True)

# === ロード関数 ===
async def load_json(filename):
    async with aiofiles.open(filename, mode="r", encoding="utf-8") as f:
        data = await f.read()
    return json.loads(data)

# === JSONローダー（型自動判定付き） ===
async def load_structures():
    semantics = await load_json("./output/semantics/structured_semantics_20251030_224846.json")
    vocab = await load_json("./output/market_vocab_20251030_201906.json")
    clusters = await load_json("./output/lexical_clusters_20251030_223013.json")

    clusters_raw = vocab.get("clusters", vocab)

    # ✅ 構造自動判定（辞書・リスト両対応）
    if isinstance(clusters_raw, list):
        clusters_list = []
        for v in clusters_raw:
            if isinstance(v, dict):
                clusters_list.append({
                    "name": v.get("name", ""),
                    "topics": v.get("keywords", [])
                })
            else:
                clusters_list.append({"name": str(v), "topics": []})
    elif isinstance(clusters_raw, dict):
        clusters_list = []
        for k, v in clusters_raw.items():
            if isinstance(v, dict):
                clusters_list.append({
                    "name": k,
                    "topics": v.get("keywords", [])
                })
            else:
                clusters_list.append({"name": k, "topics": []})
    else:
        raise ValueError(f"❌ clusters の構造が不明: {type(clusters_raw)}")

    logger.info(f"✅ 成果物読込完了: semantics={len(semantics)}, vocab={len(vocab)}, clusters={len(clusters_list)}")
    return semantics, vocab, clusters_list


# === プロンプト生成 ===
def compose_prompt(name, topics):
    topics_str = "、".join(topics[:10])
    prompt = f"""
あなたはプロのコピーライター兼SEOアナリストです。
以下の商品に対して、自然で購買意欲をそそるキャッチコピー（40～60字）を1つ、
およびSEOに効果的なALTテキスト（80～110字）を20個生成してください。

【商品名】{name}
【キーワード】{topics_str}

出力形式は次のJSON形式で：
{{
  "catchcopy": "・・・",
  "alts": ["・・・", "・・・", … (計20件)]
}}
"""
    return prompt.strip()

# === AI生成関数 ===
async def generate_text(item):
    prompt = compose_prompt(item["name"], item["topics"])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは優秀な日本語マーケティングコピーライターです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
        )
        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("⚠️ JSON解析失敗、テンプレ生成にフォールバックします。")
            data = {"catchcopy": "生成失敗", "alts": [f"{item['name']} の画像" for _ in range(20)]}
        return {"name": item["name"], "catchcopy": data["catchcopy"], "alts": data["alts"]}
    except Exception as e:
        logger.error(f"🚫 OpenAI呼び出しエラー: {e}")
        return {"name": item["name"], "catchcopy": "エラー", "alts": [f"{item['name']} の画像" for _ in range(20)]}

# === メイン ===
# === メイン ===
async def main():
    semantics, vocab, clusters = await load_structures()
    logger.info("🌸 KOTOHA ENGINE — Hybrid AI Writer 起動")

    # products生成
    products = clusters[:700]

    # ✅ tqdm_asyncio.gather に変更
    tasks = [generate_text(item) for item in products]
    results = await tqdm_asyncio.gather(*tasks, desc="🪄 生成中", total=len(tasks))

    # 保存
    output_path = os.path.join(
        OUTPUT_DIR,
        f"hybrid_writer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))

    logger.info(f"✅ 出力完了: {output_path} ({len(results)}件)")


if __name__ == "__main__":
    asyncio.run(main())
import atlas_autosave_core
