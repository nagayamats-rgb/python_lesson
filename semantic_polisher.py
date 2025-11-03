import os
import json
import csv
import logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ============================================================
# 🌸 KOTOHA ENGINE — Semantic Polisher v1.0
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("❌ OpenAI APIキーが設定されていません。")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_DIR = "./output/ai_generated"
OUTPUT_DIR = "./output/polished"
LOG_DIR = "./logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def find_latest_ai_output():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("ai_generated_") and f.endswith(".json")
    ]
    if not files:
        logging.error("❌ ai_generated_*.json が見つかりません。")
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(INPUT_DIR, f)), reverse=True)
    latest = os.path.join(INPUT_DIR, files[0])
    logging.info(f"📄 使用ファイル: {latest}")
    return latest


def refine_texts(cluster):
    """コピーとALT群を自然に再整形"""
    catch = cluster["catch_copy"]
    alts = cluster["alt_texts"]
    joined = "\n".join([f"- {a}" for a in alts[:10]])

    prompt = f"""
次の日本語コピーとALTテキスト群を、人が読んで自然で購買意欲を喚起する表現に磨いてください。
句読点の位置、リズム、語感を調整し、文を短く保ち、過度な修飾語を排除します。

入力:
キャッチコピー:
{catch}

ALT群（抜粋）:
{joined}

出力形式:
{{
  "catch_copy": "修正済みキャッチコピー",
  "alt_texts": ["修正ALT1", "修正ALT2", ...]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは熟練の日本語コピーエディターです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()

        # JSONパースを試みる
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
    except (OpenAIError, json.JSONDecodeError) as e:
        logging.warning(f"⚠️ 磨き処理失敗: {e}")

    # フォールバック（軽整形）
    refined_alts = [a.replace("。", "").strip() for a in alts]
    return {"catch_copy": catch.strip(" 。"), "alt_texts": refined_alts}


def main():
    logging.info("🌸 KOTOHA ENGINE — Semantic Polisher 起動")
    input_file = find_latest_ai_output()
    if not input_file:
        return

    with open(input_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    polished_output = []
    raw_log = os.path.join(LOG_DIR, f"semantic_polisher_raw_{timestamp}.txt")

    for cluster in tqdm(clusters, desc="🪞 磨き処理中", unit="cluster"):
        refined = refine_texts(cluster)
        polished_output.append({
            "cluster_id": cluster["cluster_id"],
            "keywords": cluster["keywords"],
            "catch_copy": refined["catch_copy"],
            "alt_texts": refined["alt_texts"],
        })

        with open(raw_log, "a", encoding="utf-8") as logf:
            logf.write(json.dumps(polished_output[-1], ensure_ascii=False))
            logf.write("\n")

    # 出力
    json_out = os.path.join(OUTPUT_DIR, f"polished_{timestamp}.json")
    csv_out = os.path.join(OUTPUT_DIR, f"polished_{timestamp}.csv")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(polished_output, f, ensure_ascii=False, indent=2)

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "keywords", "catch_copy", "alt_texts"])
        for o in polished_output:
            writer.writerow([
                o["cluster_id"],
                ", ".join(o["keywords"]),
                o["catch_copy"],
                "; ".join(o["alt_texts"]),
            ])

    logging.info(f"✅ 磨き完了: {len(polished_output)}件")
    logging.info(f"💾 出力: {json_out}")
    logging.info(f"💾 CSV出力: {csv_out}")
    print("\n🎨 KOTOHA ENGINE Semantic Polisher 完了 — 言葉が磨かれました。\n")


if __name__ == "__main__":
    main()
import atlas_autosave_core
