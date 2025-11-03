# -*- coding: utf-8 -*-
"""
🌸 KOTOHA ENGINE — Step1: Product Manifest Builder（安定版）
------------------------------------------------------------
仕様：
- Shift-JIS (cp932) で input.csv を読み込む
- 先頭行がヘッダ
- 「商品名」列を検出し、その下に並ぶ商品名を抽出
- 空白・欠損セルは無視
- 商品件数と上位サンプルをログ表示
- 次フェーズ（ai_selector.py）への橋渡しとして JSON 出力
"""

import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path

# === ログ設定 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === パス設定 ===
INPUT_PATH = Path("./input.csv")
OUTPUT_DIR = Path("./output/manifests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("🌸 KOTOHA ENGINE — Step1: Product Manifest Builder（安定版）起動")

    # === CSV読込 ===
    try:
        df = pd.read_csv(INPUT_PATH, encoding="cp932", dtype=str, header=0)
    except Exception as e:
        logger.error(f"🚫 CSV読込失敗: {e}")
        return

    # === カラム名のクリーニング ===
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    if "商品名" not in df.columns:
        logger.error(f"🚫 『商品名』列が見つかりません。取得ヘッダ: {list(df.columns)[:10]}")
        return

    # === 商品名列の抽出 ===
    products = (
        df["商品名"]
        .dropna()
        .astype(str)
        .str.replace("\u3000", " ", regex=False)  # 全角空白→半角
        .str.strip()
    )
    products = products[products != ""]

    logger.info(f"✅ 総行数: {len(df)}")
    logger.info(f"✅ 商品名あり: {len(products)} 件")
    logger.info(f"🔍 商品名サンプル: {products.head(5).tolist()}")

    # === マニフェスト構築 ===
    manifests = [
        {"index": int(idx) + 1, "商品名": name}
        for idx, name in enumerate(products.tolist())
    ]

    # === 出力 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = OUTPUT_DIR / f"products_manifest_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 出力完了: {output_path}")
    logger.info("🌸 Step1 完了 — 次は Step2: ai_selector.py へ")


if __name__ == "__main__":
    main()
import atlas_autosave_core
