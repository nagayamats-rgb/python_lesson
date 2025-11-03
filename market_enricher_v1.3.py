"""
🌸 KOTOHA ENGINE v1.3 - Market Enricher（統合型）
-------------------------------------------------
楽天市場APIを用いた市場語彙収集モジュール。
前回の辞書状態を検出し、差分があれば更新フェッチを行う。
AIは使用せず、純市場データのみで語彙辞書を形成する。
"""

import os
import re
import json
import time
import glob
import hashlib
import logging
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter

# ==========================================================
# 🌸 ロガー設定
# ==========================================================
logger = logging.getLogger("KOTOHA_MARKET_ENRICHER")
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/market_enricher_{datetime.now().strftime('%Y%m%d_%H%M')}.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
logger.setLevel(logging.INFO)

# ==========================================================
# ⚙️ 設定ロード
# ==========================================================
def load_configs():
    load_dotenv(".env.txt")
    rakuten_key = os.getenv("RAKUTEN_APP_ID")

    if not rakuten_key:
        raise EnvironmentError("❌ RAKUTEN_APP_ID が設定されていません。")

    cfg = {
        "rakuten_url": os.getenv("RAKUTEN_API_BASE_URL", "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601"),
        "rakuten_key": rakuten_key,
        "sleep_sec": 1.2,
        "max_hits": 10
    }
    return cfg

# ==========================================================
# 🧩 ユーティリティ：キーワード抽出
# ==========================================================
def extract_keywords(text):
    text = re.sub(r"[!-/:-@[-`{-~]", " ", text)  # 半角記号
    text = re.sub(r"[０-９Ａ-Ｚａ-ｚ]", " ", text)  # 全角英数
    text = re.sub(r"\s+", " ", text)
    words = re.findall(r"[一-龥ぁ-んァ-ンー]{2,}", text)
    stop = ["送料無料", "ポイント", "公式", "人気", "限定", "税込", "安心"]
    return [w for w in words if w not in stop and len(w) <= 10]

# ==========================================================
# 🔍 楽天APIフェッチ
# ==========================================================
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
            logger.warning(f"⚠️ Rakutenレスポンス異常({res.status_code}) {keyword}")
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
        logger.warning(f"⚠️ Rakuten取得失敗({keyword}): {e}")
        return []

# ==========================================================
# 🧠 差分検知ロジック
# ==========================================================
def hash_vocab_entry(vocab_entry):
    combined = "|".join(sorted(set(vocab_entry)))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def detect_differences(old_vocab, new_queries):
    old_hashes = {k: hash_vocab_entry(v) for k, v in old_vocab.items()}
    diffs = []
    for product in new_queries:
        if product not in old_hashes:
            diffs.append(product)
    return diffs

# ==========================================================
# 🧩 市場語彙収集（全件 or 差分）
# ==========================================================
def collect_vocab(cfg, queries, mode="full", old_vocab=None):
    vocab_dict = {}
    for pname, qlist in queries.items():
        if mode == "diff" and old_vocab and pname in old_vocab:
            continue
        all_words = []
        for q in qlist:
            words = fetch_rakuten(q, cfg)
            all_words.extend(words)
            time.sleep(cfg["sleep_sec"])
        vocab_dict[pname] = all_words
        logger.info(f"📦 {pname}: {len(all_words)}語収集（{len(set(all_words))}種類）")
    return vocab_dict

# ==========================================================
# 📊 集計
# ==========================================================
def summarize_vocab(vocab_dict):
    all_words = []
    for _, words in vocab_dict.items():
        all_words.extend(words)
    freq = Counter(all_words)
    df = pd.DataFrame(freq.items(), columns=["word", "count"]).sort_values("count", ascending=False)
    return df

# ==========================================================
# 🚀 メイン
# ==========================================================
def main():
    logger.info("🌸 KOTOHA ENGINE — Market Enricher v1.3 起動")
    cfg = load_configs()

    # === 入力ファイル検索 ===
    batch_files = sorted(glob.glob("query_batches_merged_*.jsonl"))
    if not batch_files:
        logger.error("❌ query_batches_merged_*.jsonl が見つかりません。")
        return
    input_file = batch_files[-1]
    logger.info(f"📄 入力: {input_file}")

    # === 過去辞書の検出 ===
    latest_json = "market_vocab_latest.json"
    if os.path.exists(latest_json) and os.path.getsize(latest_json) > 0:
        days_since_update = (time.time() - os.path.getmtime(latest_json)) / 86400
        mode = "diff" if days_since_update < 7 else "full"
        with open(latest_json, "r", encoding="utf-8") as f:
            old_vocab = json.load(f)
        logger.info(f"📘 既存辞書検出: {latest_json}（更新 {days_since_update:.1f} 日前）→ {mode.upper()} モードで実行")
    else:
        mode = "full"
        old_vocab = {}
        logger.info("🆕 過去辞書なし: 初回実行モードで開始")

    # === クエリ読み込み ===
    queries = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries[data["representative_name"]] = data["queries"]

    # === 差分検出 ===
    if mode == "diff" and old_vocab:
        diff_products = detect_differences(old_vocab, queries)
        if not diff_products:
            logger.info("✅ 差分なし: 最新状態です。実行スキップ。")
            return
        logger.info(f"🟡 差分検出: {len(diff_products)} 商品群を更新対象に選択。")
        queries = {k: v for k, v in queries.items() if k in diff_products}

    # === 市場語彙収集 ===
    new_vocab = collect_vocab(cfg, queries, mode=mode, old_vocab=old_vocab)

    # === 統合更新 ===
    merged_vocab = old_vocab.copy()
    merged_vocab.update(new_vocab)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = "market_vocab_latest.json"
    diff_path = f"market_vocab_diff_{ts}.json"
    enriched_csv = f"market_enriched_{ts}.csv"

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False, indent=2)

    # === 集計と出力 ===
    df_sum = summarize_vocab(merged_vocab)
    rows = [{"商品名": pname, "市場語彙_TOP20": "|".join([w for w, _ in Counter(words).most_common(20)])}
            for pname, words in merged_vocab.items()]
    pd.DataFrame(rows).to_csv(enriched_csv, index=False, encoding="utf-8-sig")

    logger.info(f"💾 更新済み統合辞書: {latest_path}")
    logger.info(f"💾 差分ログ: {diff_path}")
    logger.info(f"💾 統合市場語彙: {enriched_csv}")
    logger.info("✅ Market Enricher v1.3 実行完了。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
