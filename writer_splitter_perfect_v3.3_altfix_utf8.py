import csv, json, re, os, time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# --- 環境変数ロード ---
load_dotenv()

# --- 固定設定 ---
INPUT_FILE = "rakuten.csv"
OUTPUT_DIR = "./output/ai_writer"
MODEL = "gpt-4o-mini"
MAX_TOKENS = 1000
ALT_COUNT = 20

# --- クライアント ---
client = OpenAI()

# --- 禁則語・基本指針 ---
FORBIDDEN_WORDS = ["競合", "他社", "優位性", "写真", "画像", "クリック", "こちら"]
STRUCTURE_HINT = (
    "自然な日本語で1文80〜130文字程度。"
    "構成ヒント：『商品スペック→コアコンピタンス→どんな人→どんなシーン→ベネフィット』を自然に含めてください。"
    "画像描写やクリック誘導は禁止。句読点・助詞で自然に切れる文体。"
)

def refine_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"。{2,}", "。", text)
    if len(text) > 120:
        sentences = text.split("。")
        result, total = [], 0
        for s in sentences:
            if not s:
                continue
            if total + len(s) + 1 <= 110:
                result.append(s)
                total += len(s) + 1
            else:
                break
        text = "。".join(result) + "。"
    return text

def parse_json_response(raw: str) -> list:
    """AI応答を柔軟にパース"""
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\[.*\]", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                d = json.loads(match.group(0))
                if isinstance(d, dict):
                    return list(d.values())[0]
            except Exception:
                pass
    return []

def ai_generate_alt(product_name: str) -> list[str]:
    """1商品につきALT20件を生成"""
    prompt = (
        f"以下の商品について、SEO最適化されたALTテキストを{ALT_COUNT}件生成してください。\n"
        f"各ALTは全角100〜130文字で自然な日本語にしてください。\n"
        f"禁止語: {', '.join(FORBIDDEN_WORDS)}\n"
        f"{STRUCTURE_HINT}\n"
        f"商品名: {product_name}\n\n"
        "出力形式: JSON配列（例: [\"ALT1\", \"ALT2\", ...]）"
    )

    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=MAX_TOKENS,
                temperature=1.0,
                response_format={"type": "json_object"},  # ✅ 正式指定
            )
            data = res.choices[0].message.content
            parsed = parse_json_response(data)
            cleaned = [refine_text(a) for a in parsed if isinstance(a, str)]
            if len(cleaned) >= ALT_COUNT // 2:
                return cleaned[:ALT_COUNT]
        except Exception as e:
            print(f"⚠️ 生成エラー({attempt+1}/3): {e}")
            time.sleep(3)
    return [""] * ALT_COUNT

def main():
    print("🌸 writer_splitter_perfect_v3.3_altfix_utf8_final 実行開始（安定版）")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = [r["商品名"] for r in reader if r.get("商品名")]

    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")
    results = []

    for nm in tqdm(products, desc="🧠 ALT生成中"):
        alts = ai_generate_alt(nm)
        results.append({"商品名": nm, **{f"ALT{i+1}": alts[i] if i < len(alts) else "" for i in range(ALT_COUNT)}})

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUTPUT_DIR, f"alt_text_{ts}.csv")
    out_json = os.path.join(OUTPUT_DIR, f"alt_text_{ts}.jsonl")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["商品名"] + [f"ALT{i+1}" for i in range(ALT_COUNT)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(out_json, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 出力完了: {out_csv}\n✅ ログ: {out_json}\n🌸 全ALT生成終了（UTF-8／再試行保護付）")

if __name__ == "__main__":
    main()
import atlas_autosave_core
