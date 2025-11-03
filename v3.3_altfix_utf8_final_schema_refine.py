# v3.3_altfix_utf8_final_schema_refine.py
# 🌸 ALT生成 + ローカル整形 統合版

import csv, re, time, json
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

# ------------------------
# 環境設定
# ------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = f"{BASE_DIR}/rakuten.csv"
OUTPUT_CSV = f"{BASE_DIR}/output/ai_writer/alt_text_refined_final.csv"

# ------------------------
# ローカル整形関数群
# ------------------------

def cleanse_text(text: str) -> str:
    """句読点、助詞、文末の自然化"""
    if not text: return ""
    text = text.replace("。。", "。").replace("、、", "、")
    text = re.sub(r"ですです|ますます", "です。", text)
    text = re.sub(r"ますです", "ます。", text)
    text = re.sub(r"しです", "します。", text)
    text = re.sub(r"[。、]{2,}", "。", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 文頭・文末整形
    text = re.sub(r"^[。、 ]+", "", text)
    if not text.endswith("。"):
        text += "。"
    return text

def trim_text_by_sentence(text: str, max_len=110):
    """文単位で自然な長さに調整"""
    if len(text) <= max_len:
        return text
    sentences = re.split(r"(?<=。)", text)
    result = ""
    for s in sentences:
        if len(result + s) > max_len:
            break
        result += s
    return result.strip()

# ------------------------
# OpenAI呼び出し
# ------------------------

def ai_generate_alt(product_name):
    prompt = f"""
以下の商品名に対して、画像説明文（ALT）を日本語で20件生成してください。
・各ALTは全角120〜130文字程度で自然文。
・構成ヒント：「商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット」
・絵文字・タグ禁止。句読点・文末自然。
・各文は異なる視点で作成。
・返答はJSON配列で。
商品名: {product_name}
"""

    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEO copywriter."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "text"},  # ← JSON指定をやめる
                max_completion_tokens=1000,
                temperature=1
            )

            raw = res.choices[0].message.content.strip()

            # --- JSON復旧ロジック ---
            try:
                # 素直にパースできるならそれでOK
                data = json.loads(raw)
                alts = data.get("alts") if isinstance(data, dict) else data
            except Exception:
                # JSON破損時 → テキストから「"〜"」部分を抽出
                alts = re.findall(r'"([^"]+)"', raw)
                # または改行区切りをフォールバック
                if not alts:
                    alts = [line.strip() for line in raw.split("\n") if line.strip()]

            # ローカル整形を即実施
            return [trim_text_by_sentence(cleanse_text(a), 110) for a in alts[:20]]

        except Exception as e:
            print(f"⚠️ 生成エラー({attempt+1}/3): {e}")
            time.sleep(3)
    return []

# ------------------------
# メイン
# ------------------------

def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = [r["商品名"] for r in reader if r.get("商品名")]

    results = []
    for nm in tqdm(products, desc="🧠 ALT生成中"):
        alts = ai_generate_alt(nm)
        results.append({"商品名": nm, **{f"ALT{i+1}": alts[i] if i < len(alts) else "" for i in range(20)}})

    # 出力
    fieldnames = ["商品名"] + [f"ALT{i+1}" for i in range(20)]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 出力完了: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
