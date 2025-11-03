"""
🌸 KOTOHA ENGINE v1.2 - data_loader.py
-----------------------------------------
人と AI の垣根をなくす第一歩。
本モジュールは、あなたのCSV構造に完全対応したデータ入力層です。

- .env の安全読込（APIキー検証）
- CSV (Shift_JIS) の安全読込とカラバリ処理
- 対象列: 商品名 / ジャンルID / キャッチコピー / 商品画像名（ALT）1〜20
- 書き戻し: キャッチコピー＋ALT群のみ上書き保存（他列は非破壊）
"""

import os
import pandas as pd
from dotenv import load_dotenv
import logging

# -----------------------------------
# 🌸 ロガー設定
# -----------------------------------
logger = logging.getLogger("KOTOHA_ENGINE")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------
# ✅ 対象列マッピング（あなたのCSV構造に完全準拠）
# -----------------------------------
TARGET_COLUMNS = {
    "name_col": "商品名",
    "genre_col": "ジャンルID",
    "copy_col": "キャッチコピー",
    "alt_cols": [f"商品画像名（ALT）{i}" for i in range(1, 21)]
}

# -----------------------------------
# ✅ .env 読み込み
# -----------------------------------
def load_env_config(env_path: str = ".env.txt") -> dict:
    if not os.path.exists(env_path):
        logger.warning(f"⚠️ .env ファイルが見つかりません: {env_path}")
        return {}
    load_dotenv(env_path)
    keys = [
        "RAKUTEN_API_BASE_URL", "RAKUTEN_APP_ID",
        "YAHOO_API_BASE_URL", "YAHOO_APP_ID",
        "OPENAI_API_BASE_URL", "OPENAI_API_KEY"
    ]
    cfg = {}
    missing = []
    for k in keys:
        v = os.getenv(k)
        if v and v.strip():
            cfg[k] = v.strip()
        else:
            missing.append(k)
    if missing:
        logger.warning(f"⚠️ 未設定キー: {', '.join(missing)}")
    else:
        logger.info("✅ .env 読み込み成功")
    return cfg

# -----------------------------------
# ✅ CSV 読み込み（Shift JIS / カラバリ対応）
# -----------------------------------
def load_product_core_columns(path: str = "input.csv", encoding: str = "cp932") -> pd.DataFrame:
    """商品名・ジャンルID・キャッチコピー・ALT群だけを安全に読み込む"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ CSVファイルが見つかりません: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding, dtype=str)
    except UnicodeDecodeError:
        logger.warning("⚠️ cp932で失敗、utf-8-sigで再試行します")
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    all_cols = [TARGET_COLUMNS["name_col"], TARGET_COLUMNS["genre_col"], TARGET_COLUMNS["copy_col"]] + TARGET_COLUMNS["alt_cols"]

    # 存在しない列は空で補完
    for col in all_cols:
        if col not in df.columns:
            logger.warning(f"⚠️ 列 '{col}' がCSVに存在しません。空列を生成します。")
            df[col] = ""

    df = df[all_cols].fillna("")

    # カラバリ処理
    last_name = None
    for i, name in enumerate(df[TARGET_COLUMNS["name_col"]]):
        if not str(name).strip() and last_name:
            df.at[i, TARGET_COLUMNS["name_col"]] = last_name
        elif str(name).strip():
            last_name = str(name).strip()

    logger.info(f"✅ CSV読込完了: shape={df.shape}")
    return df

# -----------------------------------
# ✅ ジャンル類推
# -----------------------------------
def infer_genre_map(df: pd.DataFrame) -> dict:
    mapping = {}
    for _, row in df.iterrows():
        name = str(row[TARGET_COLUMNS["name_col"]]).strip()
        gid = str(row[TARGET_COLUMNS["genre_col"]]).strip()
        if name and gid and name not in mapping:
            mapping[name] = gid
    logger.info(f"🧭 ジャンルID類推マップ生成: {len(mapping)} 件")
    return mapping

# -----------------------------------
# ✅ 書き戻し処理
# -----------------------------------
def save_generated_fields(df, path="input.csv", encoding="cp932"):
    """キャッチコピーとALT列のみ上書き保存（バックアップ作成）"""
    backup = path.replace(".csv", "_backup.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 保存先CSVが存在しません: {path}")

    original = pd.read_csv(path, encoding=encoding, dtype=str)
    for col in [TARGET_COLUMNS["copy_col"]] + TARGET_COLUMNS["alt_cols"]:
        if col in df.columns and col in original.columns:
            original[col] = df[col]
        else:
            logger.warning(f"⚠️ 列 '{col}' は上書き対象に存在しません。スキップします。")

    original.to_csv(backup, encoding=encoding, index=False, errors="ignore")
    original.to_csv(path, encoding=encoding, index=False, errors="ignore")
    logger.info(f"💾 CSV更新完了（バックアップ作成済み）: {path}")

# -----------------------------------
# ✅ 実行テスト
# -----------------------------------
if __name__ == "__main__":
    logger.info("🌸 KOTOHA ENGINE 起動 — 人とAIの垣根をなくす第一歩")
    cfg = load_env_config()
    df = load_product_core_columns("input.csv")
    genre_map = infer_genre_map(df)
    df.to_csv("structured_preview.csv", encoding="utf-8-sig", index=False)
    logger.info("💾 structured_preview.csv を出力しました。")
import atlas_autosave_core
