import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ======================================
# 🌸 KOTOHA ENGINE — Lexical Clusterer v1.3 Progress Enhanced
# ======================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --- 初期設定 ---
load_dotenv("/Users/nagayamasoma/Desktop/python_lesson/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not OPENAI_API_KEY:
    logging.error("❌ OpenAI APIキーが設定されていません。")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


def find_latest_file(directory=OUTPUT_DIR, prefix="market_vocab_", ext=".json"):
    """最新の market_vocab ファイルを取得"""
    try:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
        if not files:
            logging.error("❌ market_vocab_*.json が見つかりません。")
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        latest = os.path.join(directory, files[0])
        logging.info(f"📄 最新ファイルを使用: {latest}")
        return latest
    except Exception as e:
        logging.error(f"⚠️ ファイル探索エラー: {e}")
        return None


def get_embedding(text, max_retries=3):
    """OpenAI Embedding API 呼び出し（リトライ付き）"""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding
        except OpenAIError as e:
            logging.warning(f"⚠️ 埋め込み失敗 ({attempt+1}/{max_retries}): {e}")
            time.sleep(2)
        except Exception as e:
            logging.error(f"🚫 予期せぬエラー: {e}")
            time.sleep(2)
    return None


def main(verbose=False):
    start_time = time.time()
    logging.info("🌸 KOTOHA ENGINE — Lexical Clusterer 起動")

    input_file = find_latest_file()
    if not input_file:
        return

    # --- JSON 読み込み ---
    with open(input_file, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    total_items = len(vocab_data)
    logging.info(f"📊 語彙データ読み込み完了: {total_items} 件")

    # --- Embedding 生成 ---
    print("\n🧠 Embedding生成中...\n")
    embeddings = []
    for item in tqdm(vocab_data, total=total_items, desc="🔍 語彙埋め込み進行中", unit="語"):
        text = item.get("term") if isinstance(item, dict) else str(item)
        emb = get_embedding(text)
        if emb:
            embeddings.append({"term": text, "vector": emb})
        else:
            logging.error(f"🚫 埋め込み失敗: {text[:40]}")

    # --- クラスタリング分析（簡易ダミー） ---
    print("\n💡 クラスタリング分析中...\n")
    clusters = [{"cluster_id": i, "terms": [e["term"] for e in embeddings[i::50]]} for i in range(50)]

    # --- 保存 ---
    output_path = os.path.join(OUTPUT_DIR, f"lexical_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n✅ 完了! Lexical Clusterer 実行結果:")
    print(f"🕒 実行時間: {elapsed/60:.2f}分")
    print(f"🧩 クラスタ数: {len(clusters)}")
    print(f"📚 埋め込み語彙数: {len(embeddings)}")
    print(f"💾 出力ファイル: {output_path}\n")

    if verbose:
        logging.info("🔍 詳細モード: 先頭クラスタを出力します:")
        print(json.dumps(clusters[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv
    main(verbose=verbose)
import atlas_autosave_core
