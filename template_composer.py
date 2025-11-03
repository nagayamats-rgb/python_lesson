"""
🌸 KOTOHA ENGINE v1.5 - template_composer.py
---------------------------------------------
第二レイヤー：テンプレート生成層
設定ファイル（kotoha_config.json / config/modules/template_composer.json）
を自動参照して柔軟にテンプレート出力を行う。

目的：
- data_loader出力を元に、文構造テンプレートを生成
- キャッチコピー（30〜60文字）とALT文（20件）を出力
- 出力・設定・フォルダ構成をconfigで管理
"""

import os
import re
import random
import json
import logging
import pandas as pd
from datetime import datetime

# -----------------------------------
# 🌸 ロガー設定
# -----------------------------------
logger = logging.getLogger("KOTOHA_ENGINE_TEMPLATE")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------
# ⚙️ 設定読み込みユーティリティ
# -----------------------------------
def load_global_config():
    """kotoha_config.json を読み込む"""
    path = "kotoha_config.json"
    if not os.path.exists(path):
        logger.error("❌ kotoha_config.json が存在しません。init_config.py を実行してください。")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_module_config(module_name: str):
    """config/modules/{module}.json を読み込む"""
    path = os.path.join("config", "modules", f"{module_name}.json")
    if not os.path.exists(path):
        logger.error(f"❌ モジュール設定ファイルが見つかりません: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------------
# 🧭 テンプレート生成クラス
# -----------------------------------
class TemplateComposer:
    def __init__(self, df: pd.DataFrame, genre_map: dict, config: dict):
        self.df = df
        self.genre_map = genre_map
        self.config = config
        self.copy_min = config.get("copy_length", {}).get("min", 30)
        self.copy_max = config.get("copy_length", {}).get("max", 60)
        self.alt_count = config.get("alt_count", 20)
        logger.info(f"🧩 TemplateComposer 起動: コピー{self.copy_min}-{self.copy_max}文字 / ALT{self.alt_count}件")

    def generate_copy_templates(self, row):
        """キャッチコピー生成"""
        name = str(row.get("商品名", "")).strip()
        genre = self.genre_map.get(name, "")
        base_kw = self._extract_keywords(name)
        benefit_kw = self._extract_benefits()

        templates = [
            f"{base_kw}で、毎日をもっと{benefit_kw}に。",
            f"{genre}の新定番。{base_kw}が叶える{benefit_kw}な暮らし。",
            f"あなたの時間を{benefit_kw}に変える{base_kw}。",
            f"{benefit_kw}さが違う。{base_kw}、話題の{genre}トレンド。",
            f"暮らしを進化させる{base_kw}、{benefit_kw}な人に。"
        ]

        chosen = random.choice(templates)
        # 長すぎる場合は丸め、短すぎる場合は拡張
        if len(chosen) > self.copy_max:
            chosen = chosen[:self.copy_max]
        elif len(chosen) < self.copy_min:
            chosen = chosen + "。" * ((self.copy_min - len(chosen)) // 2)
        return chosen

    def generate_alt_templates(self, row):
        """ALT文生成"""
        name = str(row.get("商品名", "")).strip()
        genre = self.genre_map.get(name, "")
        base_kw = self._extract_keywords(name)
        benefit_kw = self._extract_benefits()

        patterns = [
            f"{base_kw}を使った{genre}の使用シーン",
            f"{benefit_kw}を重視した{base_kw}のディテール写真",
            f"人気の{base_kw}、{genre}カテゴリの注目アイテム",
            f"高品質な{base_kw}の素材感を伝えるイメージ",
            f"{base_kw}を利用したライフスタイル例",
            f"シンプルな{base_kw}の外観写真",
            f"{benefit_kw}を感じる{base_kw}のデザイン性",
            f"{genre}向けの{base_kw}、実用的な使用イメージ",
            f"自然光で撮影した{base_kw}のリアルな質感",
            f"{base_kw}の魅力を引き出すアングルショット"
        ]

        alts = [random.choice(patterns) for _ in range(self.alt_count)]
        return alts

    # ------------------------
    # 内部ユーティリティ
    # ------------------------
    def _extract_keywords(self, text):
        text = re.sub(r"[【】\[\]\(\)（）0-9A-Za-z]", "", text)
        words = re.findall(r"[一-龥ぁ-んァ-ンー]{2,}", text)
        return words[0] if words else text[:10]

    def _extract_benefits(self):
        words = ["便利", "快適", "上質", "美しい", "心地よい", "安心", "長持ち", "軽やか", "スマート", "柔らかい"]
        return random.choice(words)

# -----------------------------------
# 🚀 メイン処理
# -----------------------------------
def run_template_composer():
    global_cfg = load_global_config()
    module_cfg = load_module_config("template_composer")
    output_dir = global_cfg.get("OUTPUT_DIR", "./")

    # 入力確認
    input_path = os.path.join(output_dir, "structured_preview.csv")
    if not os.path.exists(input_path):
        logger.error(f"❌ 入力ファイルが見つかりません: {input_path}")
        return

    df = pd.read_csv(input_path, dtype=str).fillna("")
    genre_map = {r["商品名"]: r["ジャンルID"] for _, r in df.iterrows() if str(r["ジャンルID"]).strip()}

    composer = TemplateComposer(df, genre_map, module_cfg)

    for idx, row in df.iterrows():
        df.at[idx, "キャッチコピー"] = composer.generate_copy_templates(row)
        alts = composer.generate_alt_templates(row)
        for i in range(1, module_cfg.get("alt_count", 20) + 1):
            col = f"商品画像名（ALT）{i}"
            if col in df.columns:
                df.at[idx, col] = alts[i-1] if i <= len(alts) else ""

    # 出力ファイル生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"output_templates_{timestamp}.csv")
    df.to_csv(output_file, encoding="utf-8-sig", index=False)

    logger.info(f"💾 テンプレート出力完了: {output_file}")
    logger.info(f"✅ 総出力件数: {len(df)} 件")

# -----------------------------------
# ✅ エントリーポイント
# -----------------------------------
if __name__ == "__main__":
    logger.info("🌸 KOTOHA ENGINE TemplateComposer v1.5 起動 — 設定連携型テンプレート生成")
    run_template_composer()
import atlas_autosave_core
