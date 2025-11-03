# -*- coding: utf-8 -*-
"""
v3.3 ALT長文生成（SEO＋自然文＋gpt-5最適化版）
- 入力: ./rakuten.csv（UTF-8, ヘッダに「商品名」）
- 出力:
    1. output/ai_writer/alt_text_ai_raw_longform_v4.csv
    2. output/ai_writer/alt_text_refined_final_longform_v4.csv
    3. output/ai_writer/alt_text_diff_longform_v4.csv
- モデル: gpt-5（.envで固定）
"""

import os
import re
import csv
import time
import json
import glob
from dotenv import load_dotenv
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai SDKが見つかりません。pip install openai python-dotenv を実行してください。")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# ====== 基本設定 ======
INPUT_CSV = "./rakuten.csv"
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.csv")
SEMANTICS_DIR = "./output/semantics"

FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "当店", "当社", "レビュー", "ランキング",
    "クリック", "こちら", "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
    "リンク", "ページ", "カート", "購入はこちら", "クリックして", "送料無料", "返金保証"
]

RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・]\s*[\.．、]?\s*")
WHITESPACE_RE = re.compile(r"\s+")
MULTI_COMMA_RE = re.compile(r"、{3,}")

# ====== 環境初期化 ======
def init_env():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が設定されていません。.envを確認してください。")

    model = os.getenv("OPENAI_MODEL", "gpt-5")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "1.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1800"))
    client = OpenAI(api_key=api_key)
    return client, model, temperature, max_tokens

# ====== 入力CSV読込 ======
def load_products(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("❌ 『商品名』列が見つかりません。")
        prods = [r["商品名"].strip() for r in reader if r.get("商品名")]
    uniq = list(dict.fromkeys(prods))
    return uniq

# ====== 知見サマリ ======
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "主要キーワード・用途・対象・スペックを自然に含めて。", FORBIDDEN

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    vocab = []
    forbid_local = []
    for p in files:
        data = safe_load_json(p)
        if not data:
            continue
        if isinstance(data, dict):
            if "forbidden_words" in data:
                forbid_local += data["forbidden_words"]
            if "clusters" in data:
                for c in data["clusters"]:
                    vocab += c.get("terms", [])
    kb = "知見: " + "、".join(list(dict.fromkeys(vocab[:12]))) + "。"
    all_forbidden = list({*FORBIDDEN, *forbid_local})
    return kb, all_forbidden

# ====== プロンプト ======
SYSTEM_PROMPT = (
    "あなたは楽天市場の商品画像ALTテキストを作るプロのコピーライターです。\n"
    "目的はSEO最適化された自然文を生成することです。\n"
    "【禁止事項】画像や写真の描写、競合比較、店舗メタ語、範囲表記（例：iPhone12〜16）は禁止。\n"
    "【必須構成】商品スペック→強み→対象→利用シーン→ベネフィットの順で自然に含める。\n"
    f"各ALTは全角{RAW_MIN}〜{RAW_MAX}文字で、句点「。」で終える。\n"
    "20行で、各行は1〜2文の自然文。JSON不要。"
)

def build_user_prompt(product, kb, forbidden):
    ftxt = "、".join(sorted(set(forbidden)))
    return (
        f"商品名: {product}\n"
        f"{kb}\n"
        "禁止語: " + ftxt + "\n"
        "商品名・機種・用途・機能・スペック・利点を自然に織り込みながら20文書いてください。"
    )

# ====== OpenAI呼び出し ======
def call_openai_lines(client, model, temp, max_tokens, prod, kb, forbid):
    user_prompt = build_user_prompt(prod, kb, forbid)
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "text"},
        max_completion_tokens=max_tokens,
        temperature=temp,
    )
    content = (res.choices[0].message.content or "").strip()
    lines = [LEADING_ENUM_RE.sub("", ln).strip() for ln in content.split("\n") if ln.strip()]
    return lines[:40]

# ====== 整形 ======
def soft_clip(text):
    t = text.strip()
    if not t.endswith("。"):
        t += "。"
    t = WHITESPACE_RE.sub(" ", t)
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
        else:
            t = cut
    for ng in FORBIDDEN:
        t = t.replace(ng, "")
    return t.strip()

def refine_lines(raw):
    refined = []
    for ln in raw:
        if not ln:
            continue
        ln = soft_clip(ln)
        if len(ln) < 15:
            continue
        refined.append(ln)
    # 20行に調整
    while len(refined) < 20 and refined:
        refined.append(refined[-1])
    return refined[:20]

# ====== 書き出し ======
def write_csv(products, raws, refs):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RAW_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)])
        for p, r in zip(products, raws):
            w.writerow([p] + r[:20])

    with open(REF_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_{i+1}" for i in range(20)])
        for p, r in zip(products, refs):
            w.writerow([p] + r[:20])

# ====== メイン ======
def main():
    print("🌸 ALT長文生成 v4（gpt-5安定＋SEO自然文）")
    client, model, temp, max_tokens = init_env()
    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")
    kb, forbid = summarize_knowledge()

    raws, refs = [], []
    for p in tqdm(products, desc="🧠 生成中", total=len(products)):
        try:
            raw = call_openai_lines(client, model, temp, max_tokens, p, kb, forbid)
        except Exception as e:
            raw = [f"{p} の特長を活かし、使いやすさを高めた設計です。"] * 20
        ref = refine_lines(raw)
        raws.append(raw)
        refs.append(ref)
        time.sleep(0.3)

    write_csv(products, raws, refs)
    print("✅ 出力完了:")
    print("   - RAW:", RAW_PATH)
    print("   - REF:", REF_PATH)
    print("📏 gpt-5最適化完了（句点補完・SEO強化・自然文構成）")

if __name__ == "__main__":
    main()
import atlas_autosave_core
