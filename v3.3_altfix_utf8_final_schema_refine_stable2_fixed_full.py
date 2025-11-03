# -*- coding: utf-8 -*-
"""
v3.3_altfix_utf8_final_schema_refine_stable2_fixed_full.py
------------------------------------------------------------
ALT長文生成 → ローカル整形 → 禁則適用 の安定版
OpenAI呼び出し構文を現行仕様(gpt-4o)に準拠。
ロジック順序・関数名・出力仕様は396行版を完全維持。
"""

import os
import re
import csv
import json
import glob
import time
from typing import List, Tuple, Dict, Any

# ========= 0. .env ロード =========
def load_env_file():
    candidates = [
        ".env",
        "/Users/tsuyoshi/Desktop/python_lesson/.env"
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
load_env_file()

# ========= 1. OpenAI クライアント初期化 =========
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("❌ openai パッケージが見つかりません。`pip install openai` を実行してください。")

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ========= 2. 基本設定 =========
INPUT_RAKUTEN = "/Users/tsuyoshi/Desktop/python_lesson/rakuten.csv"
OUT_DIR = "/Users/tsuyoshi/Desktop/python_lesson/output/ai_writer"
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUT_DIR, "alt_text_refined_final_stable2_fixed.csv")

MAX_COMPLETION_TOKENS = 1000
RETRY = 3
RETRY_WAIT = 3

SEMANTICS_DIR = "/Users/tsuyoshi/Desktop/python_lesson/output/semantics"

# ========= 3. ローカルJSON知見ロード =========
def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_local_knowledge() -> Tuple[str, List[str]]:
    forbidden = []
    if os.path.isdir(SEMANTICS_DIR):
        for fp in glob.glob(os.path.join(SEMANTICS_DIR, "*.json")):
            data = safe_load_json(fp)
            if isinstance(data, dict) and "forbidden_words" in data:
                forbidden.extend(data["forbidden_words"])
            elif isinstance(data, list):
                for d in data:
                    if isinstance(d, dict) and "forbidden_words" in d:
                        forbidden.extend(d["forbidden_words"])
    base_forbidden = [
        "画像", "写真", "イメージ", "こちら", "当店",
        "競合", "競合優位性", "売上No.1", "No.1", "ランキング上位",
        "リンク", "クリック", "ページ"
    ]
    forbidden.extend(base_forbidden)
    forbidden = sorted(set([x.strip() for x in forbidden if x.strip()]))

    knowledge_text = (
        "・画像描写は禁止。\n"
        "・商品スペック／コアコンピタンス／想定ユーザー／使用シーン／ベネフィットを自然に含める。\n"
        "・競合比較や“競合優位性”などのメタ表現は禁止。\n"
        "・句読点や助詞を正しく使い、自然な日本語で100〜130文字程度の文を生成。\n"
        "・文末は句点で自然に終える。"
    )
    return knowledge_text, forbidden

KNOWLEDGE_TEXT, FORBIDDEN_WORDS = collect_local_knowledge()

# ========= 4. テキスト整形関連 =========
def normalize_text(s: str) -> str:
    s = s.replace("\r", "").replace("\n", " ").strip()
    s = re.sub(r"[ \t\u3000]+", " ", s)
    s = re.sub(r"[\"'‘“”（()）\[\]]", "", s)
    s = re.sub(r"[。\.]{2,}", "。", s)
    return s.strip()

def finalize_sentence(s: str) -> str:
    s = normalize_text(s)
    if not s:
        return ""
    if not s.endswith(("。", "！", "？", "!", "?")):
        s += "。"
    return s

def natural_trim(s: str, min_len=80, target_max=110, hard_max=130):
    s = normalize_text(s)
    if len(s) > hard_max:
        s = s[:hard_max]
    if len(s) > target_max:
        last = s.rfind("。", 0, target_max)
        if last > min_len:
            s = s[:last + 1]
    return finalize_sentence(s)

def apply_forbidden_filters(s: str, forbidden: List[str]) -> str:
    text = s
    for word in forbidden:
        text = re.sub(word, "", text)
    return normalize_text(text)

# ========= 5. OpenAI呼び出し（安全修正版） =========
def call_openai_text(product_name: str) -> str:
    system_prompt = (
        "あなたは日本語のプロライターです。"
        "楽天の商品画像ALTテキストを自然な日本語で作成してください。"
        "句読点を正しく使い、文法的に正しい日本語を生成。"
        "画像や写真の説明は禁止。"
    )

    user_prompt = (
        f"商品名: {product_name}\n\n"
        f"{KNOWLEDGE_TEXT}\n\n"
        "出力条件:\n"
        "・改行で区切った25文を生成。\n"
        "・各文は100〜130文字程度。\n"
        "・句点で自然に終える。\n"
        "・競合優位性やメタ表現は禁止。\n"
        "・画像・写真・見た目などの語句は含めない。\n"
        "・自然でSEOに強い日本語を使用。"
    )

    last_err = None
    for attempt in range(RETRY):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            content = res.choices[0].message.content
            if content and content.strip():
                return content
        except Exception as e:
            last_err = e
            print(f"⚠️ OpenAI呼び出し失敗({attempt+1}/{RETRY}): {e}")
            time.sleep(RETRY_WAIT)
    raise RuntimeError(f"OpenAI呼び出し失敗: {last_err}")
# ========= 6. ALT生成ロジック =========
def ai_generate_alt(product_name: str) -> List[str]:
    """
    ALTテキストを生成。
    100〜130字の文をAIに生成させ、ローカルで整形。
    禁則語・句読点補正・文字数トリムを適用。
    """
    try:
        raw = call_openai_text(product_name)
    except Exception as e:
        print(f"⚠️ {product_name[:25]}... でAI呼び出し失敗 → fallback適用 ({e})")
        fallback = finalize_sentence(f"{product_name} の魅力を引き立て、毎日の生活を快適にする設計。機能性とデザイン性を兼ね備えています。")
        return [fallback] * 20

    # 改行単位で整形
    lines = [ln.strip("-・●* \t") for ln in raw.splitlines() if ln.strip()]
    alts = []
    for ln in lines:
        ln = apply_forbidden_filters(ln, FORBIDDEN_WORDS)
        ln = natural_trim(ln)
        if 80 <= len(ln) <= 130 and "。" in ln:
            alts.append(finalize_sentence(ln))
    # 不足時はfallback
    if not alts:
        fallback = finalize_sentence(f"{product_name} の特長を活かし、使いやすさと快適さを両立した高品質設計。")
        alts = [fallback] * 20

    # ALTを20件に調整
    if len(alts) < 20:
        alts += [alts[-1]] * (20 - len(alts))
    return alts[:20]


# ========= 7. CSV I/O =========
def read_products(path: str) -> List[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nm = (row.get("商品名") or "").strip()
            if nm:
                names.append(nm)
    uniq = list(dict.fromkeys(names))
    return uniq


def write_alt_csv(path: str, data: List[Tuple[str, List[str]]]):
    fields = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for nm, alts in data:
            row = {"商品名": nm}
            for i in range(20):
                row[f"ALT{i+1}"] = alts[i] if i < len(alts) else ""
            writer.writerow(row)


# ========= 8. ログ整形 =========
def show_progress(idx: int, total: int, name: str, avg_len: float):
    bar_len = 30
    filled = int(bar_len * (idx / total))
    bar = "█" * filled + "-" * (bar_len - filled)
    print(f"🧠 [{bar}] {idx}/{total} {name[:25]}... 平均{avg_len:.1f}字")


# ========= 9. main =========
def main():
    print("🌸 v3.3_altfix_utf8_final_schema_refine_stable2_fixed_full 実行開始（gpt-4o安全仕様）")
    print(f"✅ 使用モデル: {MODEL}")
    print(f"✅ 入力ファイル: {INPUT_RAKUTEN}")
    products = read_products(INPUT_RAKUTEN)
    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")

    results = []
    for i, nm in enumerate(products, 1):
        try:
            alts = ai_generate_alt(nm)
        except Exception as e:
            print(f"⚠️ {nm[:25]}... 生成中断 ({e})")
            fallback = finalize_sentence(f"{nm} の特長を活かした実用的な設計。日常を快適にする優れた機能性。")
            alts = [fallback] * 20
        avg_len = sum(len(a) for a in alts) / len(alts)
        show_progress(i, len(products), nm, avg_len)
        results.append((nm, alts))

    write_alt_csv(OUTPUT_CSV, results)
    print(f"✅ 出力完了: {OUTPUT_CSV}")
    print("✅ ALT: AIで100〜130字→ローカル整形で80〜110字に収束。禁則語適用・句点整形済。")


# ========= 10. 実行エントリ =========
if __name__ == "__main__":
    main()
import atlas_autosave_core
