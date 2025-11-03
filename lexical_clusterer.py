# ---------- 初期設定 ----------
import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from pathlib import Path

# ============================================================
# 🌸 KOTOHA ENGINE — lexical_clusterer.py
# ============================================================

# ---------- .env 読み込み ----------
env_path = Path("/Users/nagayamasoma/Desktop/python_lesson/.env.txt")  # ← 絶対パス指定！
if not env_path.exists():
    logging.error(f"❌ .env ファイルが見つかりません: {env_path}")
else:
    load_dotenv(dotenv_path=env_path)
    logging.info(f"✅ .env 読み込み成功: {env_path}")

# ---------- OpenAI初期化 ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("🚫 OPENAI_API_KEY が読み込まれません。")
else:
    client = OpenAI(api_key=api_key)
    logging.info("✅ OpenAI APIキーを確認しました。")


# ---------- ファイル探索ユーティリティ ----------
def find_latest_file(directory="./output", prefix="market_vocab_", ext=".json"):
    """最新の対象ファイルを検出（存在しない場合はフォルダ自動生成）"""
    # --- 出力ディレクトリの存在確認 ---
    if not os.path.exists(directory):
        logging.warning(f"⚠️ ディレクトリが存在しません。作成します: {directory}")
        os.makedirs(directory, exist_ok=True)
        return None  # 新規作成時はまだファイルがないため None を返す

    # --- ファイル検索 ---
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    if not files:
        logging.error(f"❌ {prefix}*.json が見つかりません。")
        return None

    # --- 最終更新日時でソート ---
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    latest = os.path.join(directory, files[0])
    logging.info(f"✅ 最新ファイル検出: {latest}")
    return latest


# ---------- Embeddingユーティリティ ----------
def get_embedding(text):
    """OpenAI Embedding API 呼び出し"""
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"🚫 Embedding 失敗: {e}")
        return None

# ---------- クラスタリング実行 ----------
def cluster_phrases(phrases, n_clusters=7):
    """AI補助クラスタリング"""
    valid_phrases = [p for p in phrases if p and isinstance(p, str)]
    logging.info(f"📊 クラスタリング対象フレーズ数: {len(valid_phrases)}")

    embeddings = []
    for p in valid_phrases:
        emb = get_embedding(p)
        if emb:
            embeddings.append(emb)
        time.sleep(0.1)  # APIレートリミット対策

    if not embeddings:
        logging.error("🚫 有効なEmbeddingが生成されませんでした。")
        return {}

    # 正規化
    X = normalize(np.array(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters = {i: [] for i in range(n_clusters)}
    for phrase, label in zip(valid_phrases, labels):
        clusters[label].append(phrase)
    return clusters

# ---------- クラスタ命名 ----------
CLUSTER_NAMES = ["SPEC", "FEATURE", "USAGE", "USABILITY", "EMOTION", "QUALITY", "MARKET"]

def assign_cluster_names(clusters):
    """単純な順序マッピングでクラスタ名を付与"""
    named = {}
    for i, (key, values) in enumerate(clusters.items()):
        name = CLUSTER_NAMES[i] if i < len(CLUSTER_NAMES) else f"CLUSTER_{i}"
        named[name] = values
    return named

# ---------- メイン処理 ----------
def main():
    logging.info("🌸 KOTOHA ENGINE — lexical_clusterer 起動")

    # 入力ファイル決定
    input_file = find_latest_file() or "market_vocab_20251030_201906.json"
    if not os.path.exists(input_file):
        logging.error(f"❌ 入力ファイルが見つかりません: {input_file}")
        return

    # 出力パス準備
    Path("./output").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = f"./output/lexical_clusters_{ts}.json"
    csv_out = f"./output/lexical_summary_{ts}.csv"

    # 入力読み込み
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 語彙抽出
    vocab = list({w for group in data.values() for w in group if isinstance(w, str)})
    logging.info(f"📦 語彙数（ユニーク）: {len(vocab)}")

    # クラスタリング実行
    clusters = cluster_phrases(vocab, n_clusters=len(CLUSTER_NAMES))
    named_clusters = assign_cluster_names(clusters)

    # 代表語抽出
    summary = []
    for name, words in named_clusters.items():
        rep = words[0] if words else ""
        summary.append({
            "クラスタ": name,
            "代表語": rep,
            "登録語数": len(words)
        })

    # 出力
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(named_clusters, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summary).to_csv(csv_out, index=False, encoding="utf-8-sig")

    logging.info(f"💾 クラスタ出力: {json_out}")
    logging.info(f"💾 サマリ出力: {csv_out}")
    logging.info("✅ lexical_clusterer 完了")

# ---------- 実行 ----------
if __name__ == "__main__":
    main()
import atlas_autosave_core
