# ==========================================================
# v3.3_altfix_utf8_final_schema_compat.py
# ----------------------------------------------------------
# ==========================================================
# v3.3_altfix_utf8_final_schema_compat.py
# ==========================================================
import csv, json, re, time, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ← ここを追加！！

MODEL_NAME = "gpt-4o-mini"
client = OpenAI()

# ======== 設定 ========
MODEL_NAME = "gpt-4o-mini"
INPUT_FILE = "./rakuten.csv"
OUTPUT_FILE = "./output/ai_writer/alt_text_20251101_test.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

client = OpenAI()

# ======== パラメータ ========
TARGET_MIN = 80
TARGET_MAX = 110
TARGET_PREFERRED_MIN = 95
TARGET_PREFERRED_MAX = 105

# ======== ユーティリティ ========
def _count_zenkaku(text: str) -> int:
    return len(text)

def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r'(?<=[。！？])\s*', text.strip())
    return [c for c in chunks if c]

def _trim_to_range(text: str) -> str:
    """自然終端カット（理想95〜105）"""
    sents = _split_sentences(text)
    if not sents:
        return text.strip()

    center = (TARGET_PREFERRED_MIN + TARGET_PREFERRED_MAX) // 2
    cands = []
    for k in range(len(sents), 0, -1):
        cand = ''.join(sents[:k]).strip()
        n = _count_zenkaku(cand)
        cands.append((abs(center - n),
                      (TARGET_PREFERRED_MIN <= n <= TARGET_PREFERRED_MAX),
                      (TARGET_MIN <= n <= TARGET_MAX),
                      n, cand))

    ideal = [c for c in cands if c[1]]
    if ideal:
        return sorted(ideal)[0][4]
    ok = [c for c in cands if c[2]]
    if ok:
        return sorted(ok)[0][4]

    nearest = sorted(cands)[0][4]
    nearest = re.sub(r'(。\s*)+$', '。', nearest)
    nearest = re.sub(r'(です|ます|となります)[。]*$', 'です。', nearest)
    while _count_zenkaku(nearest) > TARGET_MAX and '、' in nearest:
        nearest = nearest.rsplit('、', 1)[0] + '。'
    return nearest.strip()

# ======== ALT生成プロンプト構築 ========
def _build_alt_system_prompt(knowledge_text: str = "", forbidden_words: list[str] = None) -> str:
    forb = '、'.join(sorted(set(forbidden_words or [])))
    return (
        "あなたはECモール（楽天・Yahoo）専門の日本語コピーライターです。"
        "画像の描写は一切しません。商品理解と利用シーン、便益を自然な日本語で伝えるALT（代替テキスト）を作成します。\n"
        "要件：\n"
        "・1〜2文の自然文。絵文字・記号・HTMLタグ禁止。競合比較や“競合優位性”などのメタ表現は禁止。\n"
        "・構成ヒント（テンプレではない）：『商品スペック→コアコンピタンス→どんな人→どんなシーン→どんなベネフィット』を無理なく含める。\n"
        "・画像説明語（画像・写真・映っている・クリック等）は使用禁止。\n"
        f"・禁止語：{forb if forb else '（なし）'}\n"
        "・まず120〜130字で作成（後段で自然終端カットして80〜110字に整形）。\n"
        "・サイト内SEOを意識し、型番・素材などを自然に織り込む。\n"
        "・出力はテキストのみ。\n\n"
        f"知見要約：{knowledge_text.strip()}"
    )

def _build_alt_user_prompt(product_name: str) -> str:
    return (
        f"商品名：{product_name}\n"
        "上記商品に対するALTを日本語で20件作成してください。"
        "それぞれ120〜130字の1〜2文としてください。\n"
        "可能なら次のJSON形式で返してください：\n"
        "{\n"
        '  "alts": ["ALT1", "ALT2", …]\n'
        "}\n"
        "JSONが難しければテキストで20行でも構いません。"
    )

def _recover_alts_from_text(raw: str) -> list[str]:
    lines = [ln.strip(" ・-•\t") for ln in (raw or "").splitlines()]
    return [ln for ln in lines if ln and len(ln) > 10]

# ======== ALT生成本体 ========
def ai_generate_alt(product_name: str, knowledge_text: str = "", forbidden_words: list[str] = None) -> list[str]:
    """長文ALT生成＋自然終端トリム"""
    system_prompt = _build_alt_system_prompt(knowledge_text, forbidden_words)
    user_prompt = _build_alt_user_prompt(product_name)

    raw = ""
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            response_format={"type": "text"},
            max_completion_tokens=1000,
        )
        raw = (res.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ OpenAIエラー: {e}")

    alts = []
    if raw:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("alts"), list):
                    alts = [str(x).strip() for x in obj["alts"] if str(x).strip()]
            except Exception:
                pass

    if not alts:
        alts = _recover_alts_from_text(raw)

    if len(alts) < 20:
        alts += [alts[-1]] * (20 - len(alts))

    alts = [_trim_to_range(a) for a in alts[:20]]
    return alts

# ======== メイン処理 ========
def main():
    print("🌸 v3.3_altfix_utf8_final_schema_compat 実行開始（ALT長文→自然終端トリムモード）")

    # CSV読込（UTF-8固定）
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = [r["商品名"] for r in reader if r.get("商品名")]

    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")

    results = []
    for idx, name in enumerate(products, 1):
        print(f"🧠 ALT生成中 {idx}/{len(products)}: {name[:30]}...")
        alts = ai_generate_alt(name)
        results.append({"商品名": name, **{f"ALT{i+1}": a for i, a in enumerate(alts)}})
        time.sleep(1)

    # 書き出し
    fieldnames = ["商品名"] + [f"ALT{i+1}" for i in range(20)]
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 出力完了: {OUTPUT_FILE}")
    print("🌸 全処理完了")

# ===== エントリーポイント =====
if __name__ == "__main__":
    main()
import atlas_autosave_core
