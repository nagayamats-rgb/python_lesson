import os
import json
import csv
import re
import time
import logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ============================================================
# 🌸 KOTOHA ENGINE — AI Writer v2.1 JSON-Safe Edition
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# ======== 基本設定 ========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("❌ OpenAI APIキーが設定されていません。")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_DIR = "./output/semantics"
OUTPUT_DIR = "./output/ai_generated"
LOG_DIR = "./logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 🔍 JSON 抽出・修復ユーティリティ
# ============================================================

def extract_json_block(text: str) -> str | None:
    """```json フェンスや余分な文が混入しても JSON ブロックだけ抜く"""
    if not text:
        return None
    # ```json フェンス優先
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    # フェンスがない場合 { ... } の最外殻を抽出
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def sanitize_json_str(s: str) -> str:
    """壊れたJSONを修復: スマートクォート、全角記号、末尾カンマなど"""
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    s = s.replace("：", ":").replace("，", ",").replace("．", ".")
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", s)
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def parse_json_safely(raw_text: str, save_stub_path: str | None = None) -> dict | None:
    """テキストから安全にJSONへパース。失敗時はNone"""
    if save_stub_path:
        with open(save_stub_path, "a", encoding="utf-8") as f:
            f.write("\n\n--- RAW RESPONSE ---\n")
            f.write(raw_text)

    block = extract_json_block(raw_text)
    if not block:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    fixed = sanitize_json_str(block)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None

# ============================================================
# 🔮 OpenAI 呼び出し
# ============================================================

def ai_generate(prompt: str, max_tokens: int = 600) -> str:
    """OpenAI Chat API呼び出し (JSON厳格モード + リトライ付き)"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたは熟練のSEOコピーライター兼プロダクトエディターです。"
                            "絶対に有効なJSONのみを返し、説明・前置き・装飾・コメントは禁止です。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            logging.warning(f"⚠️ OpenAI呼び出しエラー({attempt+1}/3): {e}")
            time.sleep(2)
    return ""


def local_fallback(cluster):
    """AI呼び出し失敗時のフォールバック生成"""
    kw = ", ".join(cluster["keywords"][:3])
    return {
        "catch_copy": f"{kw} — 高品質・高機能の人気アクセサリ。",
        "alt_texts": [f"{kw} 用アクセサリ {i}" for i in range(1, 21)],
    }

# ============================================================
# 🧠 メイン処理
# ============================================================

def find_latest_semantics():
    """最新の structured_semantics_*.json を検出"""
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("structured_semantics_") and f.endswith(".json")
    ]
    if not files:
        logging.error("❌ structured_semantics_*.json が見つかりません。")
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(INPUT_DIR, f)), reverse=True)
    latest = os.path.join(INPUT_DIR, files[0])
    logging.info(f"📄 使用ファイル: {latest}")
    return latest


def main():
    start_time = time.time()
    logging.info("🌸 KOTOHA ENGINE — AI Writer 起動")

    input_file = find_latest_semantics()
    if not input_file:
        return

    with open(input_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    outputs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_path = os.path.join(LOG_DIR, f"ai_writer_raw_{timestamp}.txt")

    for cluster in tqdm(clusters, desc="🪄 生成中", unit="cluster"):
        keywords = ", ".join(cluster["keywords"][:5])

        prompt = f"""
以下のキーワード群から、JSONのみを出力してください。
前置き・説明・補足・コードフェンス以外の文字は一切出力しないこと。

要件:
- "catch_copy": 日本語 / 最大60文字 / 最小30文字 / 絵文字・顔文字なし / 訴求力重視
- "alt_texts": 長さ20の文字列配列 / 各ALTは自然な日本語フレーズ / SEOと感情バランス
- 出力は有効なUTF-8 JSONで返すこと。

キーワード: {keywords}

出力形式サンプル:
{{
  "catch_copy": "ここに60文字以内のキャッチコピー",
  "alt_texts": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...",
                "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]
}}
"""

        result = ai_generate(prompt)
        parsed = parse_json_safely(result, save_stub_path=raw_log_path)

        if not parsed or "catch_copy" not in parsed or "alt_texts" not in parsed:
            logging.warning("⚠️ JSON解析失敗、テンプレ生成にフォールバックします。")
            parsed = local_fallback(cluster)

        outputs.append({
            "cluster_id": cluster["cluster_id"],
            "keywords": cluster["keywords"],
            **parsed,
        })

    # ===== 出力 =====
    json_out = os.path.join(OUTPUT_DIR, f"ai_generated_{timestamp}.json")
    csv_out = os.path.join(OUTPUT_DIR, f"ai_generated_{timestamp}.csv")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "keywords", "catch_copy", "alt_texts"])
        for o in outputs:
            writer.writerow([o["cluster_id"], ", ".join(o["keywords"]), o["catch_copy"], "; ".join(o["alt_texts"])])

    elapsed = time.time() - start_time
    logging.info(f"✅ 完了! AI Writer 実行結果: {len(outputs)}クラスタ生成 / {elapsed:.1f}秒")
    logging.info(f"💾 JSON出力: {json_out}")
    logging.info(f"💾 CSV出力: {csv_out}")
    print("\n🎉 KOTOHA ENGINE AI Writer 完了 — 美しいコピーの誕生です。\n")


if __name__ == "__main__":
    main()
import atlas_autosave_core
