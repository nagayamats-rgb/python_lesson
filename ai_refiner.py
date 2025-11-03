"""
🌸 KOTOHA ENGINE v1.6 - ai_refiner.py
---------------------------------------------
第三レイヤー：AI生成層
テンプレート出力 (output_templates_*.csv) を読み込み、
OpenAI APIを用いて自然で魅力的なキャッチコピーとALT文を生成する。

目的：
- KOTOHA ENGINE の「職人型AI」段階への進化
- 各テンプレート文をリライトし、感性・自然性・SEOを強化
- APIキーや出力設定は .env / kotoha_config.json / modules/ai_refiner.json から自動参照
"""

import os
import re
import json
import glob
import logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import openai

# -----------------------------------
# 🌸 ロガー設定
# -----------------------------------
logger = logging.getLogger("KOTOHA_ENGINE_AIREFINER")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------
# ⚙️ 設定ロード
# -----------------------------------
def load_configs():
    load_dotenv(".env.txt")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open("kotoha_config.json", "r", encoding="utf-8") as f:
        global_cfg = json.load(f)
    with open("config/modules/ai_refiner.json", "r", encoding="utf-8") as f:
        module_cfg = json.load(f)

    output_dir = global_cfg.get("OUTPUT_DIR", "./")
    return openai.api_key, output_dir, module_cfg

# -----------------------------------
# 🧠 AIによる自然文最適化関数
# -----------------------------------
def refine_text(prompt, model="gpt-4o-mini", temperature=0.8):
    """
    OpenAI API を呼び出して自然文リファインを行う。
    - キャッチコピー: 魅力と簡潔性を強化
    - ALT: SEO的に自然な文脈を維持
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは日本語マーケティングコピーの専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=120
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"🚫 OpenAI呼び出しエラー: {e}")
        return prompt  # 安全設計：元文を返す

# -----------------------------------
# 🚀 メイン処理
# -----------------------------------
def run_ai_refiner():
    api_key, output_dir, module_cfg = load_configs()
    if not api_key:
        logger.error("❌ OpenAI APIキーが設定されていません (.env.txt を確認してください)")
        return

    # 最新テンプレートファイルを取得
    files = sorted(glob.glob(os.path.join(output_dir, "output_templates_*.csv")))
    if not files:
        logger.error("❌ テンプレートファイルが見つかりません。template_composer.py を先に実行してください。")
        return

    latest_file = files[-1]
    logger.info(f"📄 入力テンプレート: {latest_file}")

    df = pd.read_csv(latest_file, dtype=str).fillna("")

    # 各行処理
    refined_rows = []
    for idx, row in df.iterrows():
        name = row.get("商品名", "")
        copy_raw = row.get("キャッチコピー", "")
        alt_cols = [c for c in df.columns if "ALT" in c]

        # キャッチコピー最適化
        copy_prompt = f"次のテンプレートを自然で魅力的な日本語キャッチコピーに整えてください（30〜60文字）：\n「{copy_raw}」"
        refined_copy = refine_text(copy_prompt)

        # ALT最適化
        refined_alts = []
        for c in alt_cols:
            alt_prompt = f"次のALT文をSEO的に自然で読みやすく整えてください（〜60文字）：\n「{row[c]}」"
            refined_alts.append(refine_text(alt_prompt, temperature=0.6))

        row["キャッチコピー"] = refined_copy
        for i, c in enumerate(alt_cols):
            row[c] = refined_alts[i]

        refined_rows.append(row)

    # 出力ファイル
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"output_final_{timestamp}.csv")
    pd.DataFrame(refined_rows).to_csv(output_file, encoding="utf-8-sig", index=False)

    logger.info(f"💾 最終出力完了: {output_file}")
    logger.info(f"✅ 生成件数: {len(refined_rows)} 件")
    logger.info("🌸 KOTOHA ENGINE が職人型フェーズに到達しました。")

# -----------------------------------
# ✅ メイン実行
# -----------------------------------
if __name__ == "__main__":
    logger.info("🌸 KOTOHA ENGINE ai_refiner 起動 — 感性で磨く自然文生成層")
    run_ai_refiner()
import atlas_autosave_core
