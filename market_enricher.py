"""
🌸 KOTOHA ENGINE v1.2 - market_enricher.py
--------------------------------------------
目的:
- query_merger.py の出力（mergedクエリ）を使って市場データを収集
- 楽天 & Yahoo! 商品検索APIを利用して語彙・共起語を抽出
- AIを使わずに "市場辞書" を生成
"""

import os
import re
import json
import time
import csv
import glob
import logging
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter

# ----------------------------
# 🌸 ロガー設定
# ----------------------------
logger = logging.getLogger("KOTOHA_MARKET_ENRICHER")
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/market_enricher_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
logger.setLevel(logging.INFO)

# ----------------------------
# ⚙️ 設定ロード
# ----------------------------
def load_configs():
    load_dotenv(".env.txt")
    rakuten_key = os.getenv("RAKUTEN_APP_ID")
    yahoo_key = os.getenv("YAHOO_APP_ID")

    if not rakuten_key or not yahoo_key:
        raise EnvironmentError("❌ 楽天またはYahooのAPIキーが設定されていません。")

    return {
        "rakuten_url": os.getenv("RAKUTEN_API_BASE_URL", "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601"),
        "yahoo_url": os.getenv("YAHOO_API_BASE_URL", "https://shopping.yahooapis.jp/ShoppingWebService/V3/itemSearch"),
        "rakuten_key": rakuten_key,
        "yahoo_key": yahoo_key,
        "sleep_sec": 1.2,
        "max_hits": 10,
        "output_dir": "./"
    }

# ----------------------------
# 🔍 API呼び出し
# ----------------------------
def fetch_rakuten(keyword, cfg):
    try:
        params = {
            "applicationId": cfg["rakuten_key"],
            "keyword": keyword,
            "hits": cfg["max_hits"],
            "format": "json"
        }
        res = requests.get(cfg["rakuten_url"], params=params, timeout=10)
        if res.status_code != 200:
            return []
        data = res.json().get("Items", [])
        words = []
        for it in data:
            item = it.get("Item", {})
            text = " ".join([
                str(item.get("itemName", "")),
                str(item.get("catchcopy", "")),
                str(item.get("itemCaption", ""))
            ])
            words.extend(extract_keywords(text))
        return words
    except Exception as e:
        logger.warning(f"⚠️ Rakuten取得失敗（{keyword}）: {e}")
        return []

def fetch_yahoo(keyword, cfg):
    try:
        params = {
            "appid": cfg["yahoo_key"],
            "query": keyword,
            "results": cfg["max_hits"]
        }
        res = requests.get(cfg["yahoo_url"], params=params, timeout=10)
        if res.status_code != 200:
            return []
        data = res.json().get("hits", [])
        words = []
        for it in data:
            text = " ".join([
                str(it.get("name", "")),
                str(it.get("headline", "")),
                str(it.get("description", ""))
            ])
            words.extend(extract_keywords(text))
        return words
    except Exception as e:
        logger.warning(f"⚠️ Yahoo取得失敗（{keyword}）: {e}")
        return []

# ----------------------------
# 🧩 キーワード抽出（簡易日本語処理）
# ----------------------------
def extract_keywords(text):
    text = re.sub(r"[!-/:-@[-`{-~]", " ", text)  # 半角記号
    text = re.sub(r"[０-９Ａ-Ｚａ-ｚ]", " ", text)  # 全角英数
    text = re.sub(r"\s+", " ", text)
    # 2文字以上の日本語単語を抽出
    words = re.findall(r"[一-龥ぁ-んァ-ンー]{2,}", text)
    # ノイズ除去
    stopwords = ["送料無料", "ポイント", "セール", "税込", "公式", "人気", "限定"]
    return [w for w in words if w not in stopwords and len(w) < 12]

# ----------------------------
# 📊 集計・スコア化
# ----------------------------
def summarize_vocab(vocab_dict):
    all_words = []
    for product, words in vocab_dict.items():
        all_words.extend(words)
    freq = Counter(all_words)
    df = pd.DataFrame(freq.items(), columns=["word", "count"]).sort_values("count", ascending=False)
    return df

# ----------------------------
# 🚀 メイン
# ----------------------------
def main():
    logger.info("🌸 KOTOHA ENGINE — Market Enricher 起動")
    cfg = load_configs()

    files = sorted(glob.glob("query_batches_merged_*.jsonl"))
    if not files:
        logger.error("❌ query_batches_merged_*.jsonl が見つかりません。query_merger.py を先に実行してください。")
        return

    input_file = files[-1]
    logger.info(f"📄 入力: {input_file}")

    # 語彙辞書 {product_name: [words...]}
    vocab_dict = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pname = data["representative_name"]
            queries = data["queries"]
            all_words = []

            for q in queries:
                words_r = fetch_rakuten(q, cfg)
                words_y = fetch_yahoo(q, cfg)
                all_words.extend(words_r + words_y)
                time.sleep(cfg["sleep_sec"])

            vocab_dict[pname] = all_words
            logger.info(f"📦 {pname}: {len(all_words)}語収集 ({len(set(all_words))}種類)")

    # ------------------ 出力 ------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"market_vocab_{ts}.json"
    csv_path = f"market_vocab_summary_{ts}.csv"
    enriched_path = f"market_enriched_{ts}.csv"

    # 詳細語彙JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    # 頻度集計
    vocab_summary = summarize_vocab(vocab_dict)
    vocab_summary.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # 商品単位で上位語を保存
    rows = []
    for pname, words in vocab_dict.items():
        c = Counter(words)
        top_words = [w for w, _ in c.most_common(20)]
        rows.append({"商品名": pname, "市場語彙_TOP20": "|".join(top_words)})

    pd.DataFrame(rows).to_csv(enriched_path, index=False, encoding="utf-8-sig")

    logger.info(f"💾 詳細辞書: {json_path}")
    logger.info(f"💾 語彙頻度集計: {csv_path}")
    logger.info(f"💾 商品別語彙: {enriched_path}")
    logger.info(f"✅ 完了: {len(vocab_dict)} 商品群の市場語彙を収集しました。")
    logger.info("🧭 次は ai_writer.py で知識＋テンプレートを融合して自然文生成へ。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
