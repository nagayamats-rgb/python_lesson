# ============================================
# 🌸 writer_splitter_perfect_v3_3.py
# 全件AI生成＋知見要約＋楽天/Yahoo/ALT分割出力
# GPT-4o安定モード（writer_hybrid_v5_8_fixed3 構造統合）
# ============================================

import os
import csv
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ==== 基本設定 ====
INPUT_CSV = "./input.csv"
OUTPUT_DIR = "./output/ai_writer"
MAX_TOKENS = 1500
RETRY_WAIT = 5
RETRIES = 3

# ==== GPT設定 ====
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==== ファイル名生成 ====
ts = time.strftime("%Y%m%d_%H%M", time.localtime())
RAKUTEN_CSV = f"{OUTPUT_DIR}/rakuten_copy_{ts}.csv"
YAHOO_CSV = f"{OUTPUT_DIR}/yahoo_copy_{ts}.csv"
ALT_CSV = f"{OUTPUT_DIR}/alt_text_{ts}.csv"
JSONL_FILE = f"{OUTPUT_DIR}/split_full_{ts}.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 汎用関数 ====
def load_product_names(csv_path):
    """Shift-JIS CSVから商品名を抽出"""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="cp932")
    name_col = [c for c in df.columns if "商品名" in c]
    if not name_col:
        raise ValueError("商品名カラムが見つかりません")
    names = df[name_col[0]].dropna().unique().tolist()
    return [n.strip() for n in names if str(n).strip()]


def call_openai_json(messages, retries=RETRIES):
    """
    GPT-4o対応の安全呼び出し。
    空応答・JSON不整合・通信エラー時は再試行。
    """
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1,
            )
            content = res.choices[0].message.content.strip()
            if not content:
                print(f"⚠️ 空応答（{attempt+1}/{retries}）→再試行中…")
                time.sleep(RETRY_WAIT)
                continue

            # JSON抽出
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                if "{" in content and "}" in content:
                    part = content[content.find("{"):content.rfind("}")+1]
                    data = json.loads(part)
                else:
                    print(f"⚠️ JSON変換失敗（{attempt+1}/{retries}）→再試行")
                    time.sleep(RETRY_WAIT)
                    continue
            return data

        except Exception as e:
            print(f"⚠️ OpenAIエラー: {e}（{attempt+1}/{retries}）")
            time.sleep(RETRY_WAIT)

    print("❌ 応答取得失敗（すべての再試行が失敗）")
    return None


# ==== AI生成関数 ====
def ai_generate(product_name, knowledge):
    """
    商品名＋知見要約から楽天/Yahoo/ALTを生成。
    """
    sys_prompt = (
        "あなたはEコマースSEO最適化ライターです。"
        "楽天市場とYahooショッピングの両方に対応する販促コピーを生成してください。"
        "以下の構成を参考に、自然で人間的な文を出力してください：\n"
        "・商品スペック\n"
        "・コアコンピタンス（他社との差別化）\n"
        "・どんな人が\n"
        "・どんなシーンで使うと\n"
        "・どんな課題を解決し、どんな利便性を提供するか\n"
        "楽天/Yahoo/ALTそれぞれでトーンや文字数を調整し、SEOに有効な語を自然に含めてください。"
        "ALTには画像描写を入れず、機能・用途・便益を中心に80〜110文字で20件生成してください。"
    )

    user_prompt = f"""
商品名：{product_name}
知見要約：{knowledge}

出力形式は必ずJSONで：
{{
  "rakuten": "全角60〜80文字（最大87文字）の自然文コピー",
  "yahoo": "全角25〜30文字の自然文コピー",
  "alts": ["ALT1","ALT2",...,"ALT20"]
}}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = call_openai_json(messages)
    if not result:
        return "", "", []
    return result.get("rakuten", ""), result.get("yahoo", ""), result.get("alts", [])


# ==== メイン ====
def main():
    print(f"🌸 writer_splitter_perfect_v3.3 実行開始（GPT-4o安定モード）")

    # 商品名読み込み
    names = load_product_names(INPUT_CSV)
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")

    # JSONL・CSV出力初期化
    rakuten_rows, yahoo_rows, alt_rows = [], [], []
    jsonl_f = open(JSONL_FILE, "w", encoding="utf-8")

    for nm in tqdm(names, desc="🧠 商品別AI生成中"):
        # ローカル知見要約（簡易ダミー）
        knowledge = (
            "マーケット傾向・購買意図・競合差別化要素を総合し、"
            "主要キーワードを自然に織り交ぜた販促知見。"
        )

        rak, yah, alts = ai_generate(nm, knowledge)
        json.dump(
            {"product_name": nm, "rakuten": rak, "yahoo": yah, "alts": alts},
            jsonl_f,
            ensure_ascii=False,
        )
        jsonl_f.write("\n")

        rakuten_rows.append([nm, rak])
        yahoo_rows.append([nm, yah])
        alt_rows.append([nm] + alts + [""] * (20 - len(alts)))

    jsonl_f.close()

    # CSV出力
    with open(RAKUTEN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["商品名", "楽天キャッチコピー"]] + rakuten_rows)

    with open(YAHOO_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["商品名", "Yahooキャッチコピー"]] + yahoo_rows)

    alt_header = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]
    with open(ALT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([alt_header] + alt_rows)

    print(f"✅ 出力完了:\n   - 楽天: {RAKUTEN_CSV}\n   - Yahoo: {YAHOO_CSV}\n   - ALT20: {ALT_CSV}\n   - JSONL: {JSONL_FILE}")
    print("✅ 共通ALT20は『alt_text_*.csv』に全商品ぶんを横持ちで書き出します。")


# ==== 実行 ====
if __name__ == "__main__":
    main()
import atlas_autosave_core
