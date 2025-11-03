import os
import json
import asyncio
import logging
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv("/Users/tsuyoshi/Desktop/python_lesson/.env")
# ============================================================
# 🌸 KOTOHA ENGINE — Hybrid AI Writer v3 (開発者モード)
# ============================================================

# ---- ログ設定 ----
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---- 環境変数 ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.critical("🚫 OPENAI_API_KEY が設定されていません。")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# 🔍 ユーティリティ
# ============================================================

def find_latest_file(base_dir, prefix, ext):
    """指定ディレクトリ以下から最新ファイルを探索（再帰対応）"""
    logging.debug(f"🔎 Searching latest file: {prefix}*{ext} under {base_dir}")
    matched = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(ext):
                matched.append(os.path.join(root, f))
    if not matched:
        logging.warning(f"⚠️ No file found for {prefix}")
        return None
    latest = max(matched, key=os.path.getmtime)
    logging.debug(f"✅ Found latest: {latest}")
    return latest


def safe_load_json(path):
    """安全なJSONロード"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.debug(f"📄 Loaded JSON: {path} ({len(data)} entries)")
        return data
    except Exception as e:
        logging.error(f"🚫 JSON読込失敗: {path} ({e})")
        return []


# ============================================================
# 📦 データロード
# ============================================================

def load_structures():
    """構造データをロード"""
    base_dir = "/Users/tsuyoshi/Desktop/python_lesson"
    logging.info(f"📂 データ読込開始: {base_dir}")

    semantics_file = find_latest_file(base_dir, "structured_semantics_", ".json")
    vocab_file = find_latest_file(base_dir, "market_vocab_", ".json")
    cluster_file = find_latest_file(base_dir, "lexical_clusters_", ".json")

    if not all([semantics_file, vocab_file, cluster_file]):
        raise FileNotFoundError("❌ 必須JSONのいずれかが見つかりません。")

    semantics = safe_load_json(semantics_file)
    vocab = safe_load_json(vocab_file)
    clusters = safe_load_json(cluster_file)

    logging.info(f"✅ 読込完了: semantics={len(semantics)}, vocab={len(vocab)}, clusters={len(clusters)}")
    return semantics, vocab, clusters


# ============================================================
# 🧠 AI生成処理
# ============================================================

async def generate_text(prompt, retries=2):
    """ChatGPT呼び出し（再試行付き）"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは高品質な日本語コピーライターです。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    return parsed
                except json.JSONDecodeError:
                    logging.warning("⚠️ JSON解析失敗、テンプレ生成にフォールバックします。")
            return {"copy": text, "alts": []}
        except Exception as e:
            logging.warning(f"⚠️ APIエラー: {e} (試行 {attempt+1}/{retries})")
            await asyncio.sleep(2)
    return {"copy": "生成失敗", "alts": []}


# ============================================================
# 🧩 クラスタ単位処理
# ============================================================

async def process_cluster(cluster, idx, total):
    """クラスタ単位で生成"""
    topic = cluster.get("name", f"商品{idx}")
    keywords = ", ".join(cluster.get("topics", []))[:200]

    prompt = f"""
次の商品の特徴に基づいてキャッチコピー（40〜60文字）と画像ALTテキスト（各80〜110文字）を生成してください。

【商品】{topic}
【キーワード】{keywords}

出力はJSON形式で：
{{
  "copy": "キャッチコピー文",
  "alts": ["ALT文1", "ALT文2", ...20個]
}}
"""
    logging.debug(f"🧩 [{idx}/{total}] Prompt準備完了: {topic}")
    result = await generate_text(prompt)
    if not result.get("alts"):
        logging.warning(f"⚠️ ALT未生成: {topic}")
    return result


# ============================================================
# 🚀 メイン処理
# ============================================================

async def main():
    try:
        semantics, vocab, clusters = load_structures()
    except FileNotFoundError as e:
        logging.critical(f"🚫 致命的エラー: {e}")
        raise

    logging.info("🌸 KOTOHA ENGINE — Hybrid AI Writer (開発者モード) 起動")

    results = []
    total = len(clusters[:700])

    for idx, cluster in enumerate(tqdm(clusters[:700], desc="🪄 生成中"), start=1):
        res = await process_cluster(cluster, idx, total)
        results.append(res)

    # 統計ログ
    total_alts = sum(len(r.get("alts", [])) for r in results)
    avg_alts = total_alts / len(results) if results else 0
    logging.info(f"📊 ALT生成平均数: {avg_alts:.2f}")
    logging.info(f"📊 総生成文数: {len(results)} クラスタ / {total_alts} ALT")

    # 出力
    out_dir = "./output/ai_writer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/hybrid_writer_full_{datetime.now():%Y%m%d_%H%M}_dev.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"💾 出力完了: {out_path} ({len(results)}件)")
    logging.info("🏁 KOTOHA ENGINE Hybrid Writer 完了")


# ============================================================
# 🔧 エントリーポイント
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
import atlas_autosave_core
