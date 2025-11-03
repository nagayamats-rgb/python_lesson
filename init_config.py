"""
🌸 KOTOHA ENGINE v1.4 - init_config.py
----------------------------------------
初回セットアップ時に実行。
.kotoha_config.json と モジュール設定フォルダ(config/modules/) を生成し、
KOTOHA ENGINE の基盤を構築する。
"""

import os
import json
import logging
from datetime import datetime

# -----------------------------------
# 🌸 ロガー設定
# -----------------------------------
logger = logging.getLogger("KOTOHA_ENGINE_INIT")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------
# 📁 ディレクトリ構成
# -----------------------------------
BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "config", "modules")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# -----------------------------------
# ⚙️ メイン設定ファイル
# -----------------------------------
MAIN_CONFIG = {
    "PROJECT_NAME": "KOTOHA ENGINE",
    "VERSION": "1.4",
    "OUTPUT_DIR": "./",
    "KEEP_INTERMEDIATE": True,
    "DEFAULT_ENCODING": "cp932",
    "MAX_CONCURRENCY": 6,
    "LOG_LEVEL": "INFO",
    "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "MODULES": {
        "data_loader": "config/modules/data_loader.json",
        "template_composer": "config/modules/template_composer.json",
        "ai_refiner": "config/modules/ai_refiner.json",
        "evaluator": "config/modules/evaluator.json"
    }
}

# -----------------------------------
# 🧩 各モジュール個別設定
# -----------------------------------
MODULE_CONFIGS = {
    "data_loader": {
        "description": "入力CSVの整形と構造化",
        "output_file": "structured_preview.csv",
        "backup_file": "input_backup.csv",
        "encoding_priority": ["cp932", "utf-8-sig"]
    },
    "template_composer": {
        "description": "構造データから文テンプレートを生成",
        "output_file": "output_templates.csv",
        "copy_length": {"min": 30, "max": 60},
        "alt_count": 20
    },
    "ai_refiner": {
        "description": "OpenAI APIを用いた文の自然化・最適化",
        "output_file": "output_final.csv",
        "use_batch": True,
        "max_retries": 3
    },
    "evaluator": {
        "description": "出力文の品質分析・スコアリング",
        "report_file": "evaluation_report.csv",
        "metrics": ["readability", "seo_density", "uniqueness"]
    }
}

# -----------------------------------
# 🚀 初期化関数
# -----------------------------------
def initialize_kotoha_engine():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    main_config_path = os.path.join(BASE_DIR, "kotoha_config.json")
    with open(main_config_path, "w", encoding="utf-8") as f:
        json.dump(MAIN_CONFIG, f, indent=2, ensure_ascii=False)

    for name, cfg in MODULE_CONFIGS.items():
        path = os.path.join(CONFIG_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    logger.info("✅ kotoha_config.json を生成しました。")
    logger.info("✅ モジュール設定ファイルを作成しました。")
    logger.info("📁 構成フォルダ: ./config/modules/")
    logger.info("🌸 KOTOHA ENGINE 設定初期化が完了しました。")

# -----------------------------------
# 🧭 実行
# -----------------------------------
if __name__ == "__main__":
    logger.info("🌸 KOTOHA ENGINE 設定初期化を開始します。")
    initialize_kotoha_engine()
import atlas_autosave_core
