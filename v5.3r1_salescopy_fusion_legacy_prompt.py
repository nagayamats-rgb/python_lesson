# -*- coding: utf-8 -*-
"""
v5.3r1_salescopy_fusion_legacy_prompt.py
- 楽天ALT生成（v3.3自然文プロンプト × v5.3安定構造）
- 句点終止・禁則・体言止め可・文末引用符禁止
"""

import os
import re
import csv
import glob
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# =========================
# 定数
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_CSV = "/Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv"
OUT_DIR   = os.path.join(BASE_DIR, "output", "ai_writer")

RAW_PATH  = os.path.join(OUT_DIR, "alt_text_ai_raw_salescopy_v5_3r1.csv")
REF_PATH  = os.path.join(OUT_DIR, "alt_text_refined_salescopy_v5_3r1.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_salescopy_v5_3r1.csv")

SEMANTICS_DIR = os.path.join(BASE_DIR, "output", "semantics")
FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "イメージ図",
    "当店", "当社", "レビュー", "ランキング", "クリック", "こちら", "リンク", "購入はこちら",
    "最安", "No.1", "ナンバーワン", "売上No1", "業界最高", "競合", "競合優位性", "返金保証"
]
RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
SPACE_RE = re.compile(r"\s+")
MULTI_COMMA_RE = re.compile(r"、{3,}")

# =========================
# 環境ロード
# =========================
def init_client():
    load_dotenv(override=True)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    temp = float(os.getenv("OPENAI_TEMPERATURE", "0.9"))
    max_t = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
    client = OpenAI(api_key=key)
    return client, model, temp, max_t

# =========================
# 入力
# =========================
def load_products(path):
    if not os.path.exists(path):
        raise SystemExit(f"❌ 入力CSVが見つかりません: {path}")
    products = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                products.append(nm)
    seen, uniq = set(), []
    for n in products:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq

# =========================
# 知見（ゆる要約）
# =========================
def safe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "知見: スペック・機能・用途・対象を自然に含め、句点終止・画像語禁止。", []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    words = []
    forb = []
    for p in files:
        data = safe_load_json(p)
        if not data: continue
        try:
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        words.extend([x for x in v if isinstance(x, str)])
                    elif isinstance(v, str):
                        words.append(v)
            elif isinstance(data, list):
                words.extend([x for x in data if isinstance(x, str)])
        except Exception:
            pass
    words = list(dict.fromkeys(words))
    return "知見: " + "、".join(words[:20]) + "。", list(set(FORBIDDEN + forb))

# =========================
# SYSTEM / USER プロンプト（v3.3復元＋文末引用符禁止）
# =========================
SYSTEM_PROMPT = (
    "あなたは楽天のSEOに強い日本語コピーライターです。"
    "目的は商品画像のALTテキストを自然な日本語で20本生成することです。"
    "以下の条件を厳守してください：\n"
    "・画像や写真の描写語を使わない。\n"
    "・レビュー・ランキング・リンク等のメタ語を使わない。\n"
    "・誇大表現や競合比較は禁止。\n"
    "・各文は全角約100〜130文字、1〜2文で自然に。句点「。」で終える。\n"
    "・箇条書きや番号や「ALT:」などは付けない。\n"
    "・商品名、対応機種、機能、用途、対象を自然に織り込む。\n"
    "・出力は20行テキストのみ。\n"
    "・文末に「\"」や「'」を付けないこと。"
)

def build_user_prompt(product, kb_text, forb):
    forbid_txt = "、".join(sorted(forb))
    return (
        f"商品名: {product}\n"
        f"{kb_text}\n"
        f"禁止語: {forbid_txt}\n"
        "各行は自然な日本語で1〜2文構成。句点「。」で終える。\n"
        "20行で出力。"
    )

# =========================
# OpenAI 呼び出し
# =========================
def call_openai_lines(client, model, temp, max_t, sys, usr, retry=3, wait=5):
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
                response_format={"type": "text"},
                temperature=temp,
                max_completion_tokens=max_t
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                continue
            lines = [LEADING_ENUM_RE.sub("", l).strip("・-—●　").strip()
                     for l in content.split("\n") if l.strip()]
            return lines[:80]
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAIエラー: {last_err}")

# =========================
# 整形
# =========================
def soft_clip_sentence(text):
    if not text: return ""
    t = text.strip()
    if not t.endswith("。"): t += "。"
    t = SPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    for ng in FORBIDDEN:
        if ng in t: t = t.replace(ng, "")
    if len(t) > 120:
        cut = t[:120]; p = cut.rfind("。")
        t = cut[:p+1] if p != -1 else cut
    return t.strip()

def refine_lines(raw, product):
    norm = [soft_clip_sentence(x) for x in raw if len(x.strip()) > 10]
    uniq = []
    for n in norm:
        if n not in uniq:
            uniq.append(n)
    while len(uniq) < 20:
        uniq.append(f"{product}の特長を活かした設計で、日常を快適にします。")
    return uniq[:20]

# =========================
# 書き出し
# =========================
def ensure_outdir(): os.makedirs(OUT_DIR, exist_ok=True)

def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

# =========================
# メイン
# =========================
def main():
    print("🌸 ALT生成 v5.3r1（3.3プロンプト×5.3構造）")
    client, model, temp, max_t = init_client()
    ensure_outdir()

    kb_text, forb = summarize_knowledge()
    products = load_products(INPUT_CSV)
    print(f"✅ 商品数: {len(products)}件")

    all_raw, all_ref = [], []

    for p in tqdm(products, desc="🧠 生成中", total=len(products)):
        usr = build_user_prompt(p, kb_text, forb)
        try:
            raw = call_openai_lines(client, model, temp, max_t, SYSTEM_PROMPT, usr)
        except Exception:
            raw = [f"{p}は使いやすさと耐久性を兼ね備えた設計です。"] * 20
        ref = refine_lines(raw, p)
        all_raw.append(raw[:20])
        all_ref.append(ref)
        time.sleep(0.2)

    write_csv(RAW_PATH, ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)],
              [[p]+r for p, r in zip(products, all_raw)])
    write_csv(REF_PATH, ["商品名"] + [f"ALT_{i+1}" for i in range(20)],
              [[p]+r for p, r in zip(products, all_ref)])
    write_csv(DIFF_PATH,
              ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_refined_{i+1}" for i in range(20)],
              [[p]+r+f for p, r, f in zip(products, all_raw, all_ref)])

    def avg_len(blocks): 
        lens = [len(x) for lines in blocks for x in lines if x]
        return sum(lens)/max(1,len(lens))

    print("✅ 出力完了")
    print(f"📏 平均文字数: raw={avg_len(all_raw):.1f}, refined={avg_len(all_ref):.1f}")
    print(f"💾 出力先:\n - {RAW_PATH}\n - {REF_PATH}\n - {DIFF_PATH}")
    print("🔒 プロンプト: v3.3自然文 × 文末引用符禁止")

if __name__ == "__main__":
    main()
import atlas_autosave_core
