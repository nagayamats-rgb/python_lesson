"""
🌸 KOTOHA ENGINE v1.1 - query_merger.py
--------------------------------------------
目的:
- query_generator.py の出力を統合（カラバリ/重複商品をマージ）
- 商品単位で一意な検索クエリ群を構築
- 次工程（market_enricher）へのバッチ入力を生成
"""

import os
import re
import csv
import glob
import json
import logging
import pandas as pd
from datetime import datetime

# ----------------------------
# 🌸 ロガー
# ----------------------------
logger = logging.getLogger("KOTOHA_QUERY_MERGER")
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/query_merger_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
logger.setLevel(logging.INFO)

# ----------------------------
# 🔍 類似判定（カラバリ統合のため）
# ----------------------------
def normalize_name(name: str) -> str:
    """ノイズ除去し、比較用の正規化文字列を生成"""
    if not name:
        return ""
    t = str(name)
    t = re.sub(r"[【】\[\]\(\)（）]", " ", t)
    t = re.sub(r"[0-9A-Za-z\-_/|:＋+＊*％%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_variant(base: str, target: str) -> bool:
    """商品名の類似度をチェック（先頭15文字が一致すれば同系統とみなす）"""
    if not base or not target:
        return False
    nb, nt = normalize_name(base), normalize_name(target)
    return nb[:15] == nt[:15] or nb in nt or nt in nb

# ----------------------------
# 📦 統合処理
# ----------------------------
def merge_queries(df):
    merged = []
    seen = set()
    groups = []

    for _, row in df.iterrows():
        name = row.get("商品名", "")
        genre = row.get("ジャンルID", "")
        if not name:
            continue

        # クエリ列
        qs = [str(row.get(f"Q{i}", "")).strip() for i in range(1, 21) if str(row.get(f"Q{i}", "")).strip()]
        if not qs:
            continue

        # 既存グループとのマッチ
        matched = None
        for g in groups:
            if is_variant(g["name"], name) or (genre and g["genre"] == genre):
                matched = g
                break

        if matched:
            matched["queries"].update(qs)
            matched["names"].add(name)
        else:
            groups.append({"name": name, "genre": genre, "queries": set(qs), "names": {name}})

    for g in groups:
        merged.append({
            "代表商品名": list(g["names"])[0],
            "ジャンルID": g["genre"],
            "関連商品数": len(g["names"]),
            **{f"Q{i}": q for i, q in enumerate(sorted(g["queries"]), start=1)}
        })

    return pd.DataFrame(merged)

# ----------------------------
# 🚀 メイン
# ----------------------------
def main():
    logger.info("🌸 KOTOHA ENGINE — Query Merger 起動")

    files = sorted(glob.glob("query_candidates_*.csv"))
    if not files:
        logger.error("❌ query_candidates_*.csv が見つかりません。query_generator.py を先に実行してください。")
        return

    input_file = files[-1]
    logger.info(f"📄 入力: {input_file}")

    df = pd.read_csv(input_file, dtype=str).fillna("")

    # バックアップ
    raw_backup = "query_candidates_raw.csv"
    df.to_csv(raw_backup, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    logger.info(f"💾 バックアップ作成: {raw_backup}")

    # マージ実行
    merged_df = merge_queries(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_csv = f"query_candidates_merged_{ts}.csv"
    merged_jsonl = f"query_batches_merged_{ts}.jsonl"

    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # JSONL形式（APIバッチ用）
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for _, row in merged_df.iterrows():
            qlist = [str(row.get(f"Q{i}", "")).strip() for i in range(1, 51) if str(row.get(f"Q{i}", "")).strip()]
            if not qlist:
                continue
            f.write(json.dumps({
                "representative_name": row.get("代表商品名", ""),
                "genre_id": row.get("ジャンルID", ""),
                "queries": qlist
            }, ensure_ascii=False) + "\n")

    logger.info(f"💾 マージ済みCSV: {merged_csv}")
    logger.info(f"💾 APIバッチJSONL: {merged_jsonl}")
    logger.info(f"✅ 統合完了: {len(merged_df)} 商品群に整理されました。")
    logger.info("🧭 次は market_enricher.py で外部API注入へ進めます。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
