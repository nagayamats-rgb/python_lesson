# -*- coding: utf-8 -*-
"""
ALT長文生成 v4.6（自然文＋安定生成＋リトライ保護）
------------------------------------------------------------
- 対象: 楽天 ALT 専用
- 入力: ./rakuten.csv （UTF-8, ヘッダに「商品名」）
- 出力:
    1. output/ai_writer/alt_text_ai_raw_longform_v4.6.csv
    2. output/ai_writer/alt_text_refined_final_longform_v4.6.csv
    3. output/ai_writer/alt_text_diff_longform_v4.6.csv
- 特徴:
    ✅ gpt-5対応（.env固定）
    ✅ 名詞羅列禁止・自然文強制
    ✅ OpenAI空応答対策
    ✅ 安定スロットリング＋リトライ保護
"""

import os
import re
import csv
import glob
import json
import time
from dotenv import load_dotenv
from collections import defaultdict

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDKが見つかりません。`pip install openai python-dotenv` を実行してください。")

# ====== 定数 ======
INPUT_CSV = "./rakuten.csv"
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.6.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.6.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.6.csv")
SEMANTICS_DIR = "./output/semantics"

FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "当店", "当社", "レビュー", "ランキング",
    "クリック", "こちら", "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン",
    "リンク", "ページ", "カート", "購入はこちら", "送料無料（確約）", "返金保証"
]

RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

# ====== 正規表現 ======
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・●]\s*[\.．、]?\s*")
MULTI_COMMA_RE = re.compile(r"、{3,}")
WS_RE = re.compile(r"\s+")

# ====== 環境設定 ======
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")
    model = os.getenv("OPENAI_MODEL", "gpt-5").strip()
    temp = float(os.getenv("OPENAI_TEMPERATURE", "1.2").strip())
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000").strip())
    client = OpenAI(api_key=api_key)
    return client, model, temp, max_tokens

# ====== 入力 ======
def load_products(path):
    if not os.path.exists(path):
        raise SystemExit(f"❌ 入力CSVが見つかりません: {path}")
    products = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("❌ CSVに『商品名』列が存在しません。")
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                products.append(nm)
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

# ====== 知見要約 ======
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "主要スペック・用途・対象・機能・便益を自然に含める。", FORBIDDEN[:]
    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    clusters, market, concept, tmpl, forbid_local = [], [], [], [], []
    for p in files:
        data = safe_load_json(p)
        if not data:
            continue
        name = os.path.basename(p).lower()
        if isinstance(data, list):
            data_list = data
            data_dict = {}
        elif isinstance(data, dict):
            data_list = []
            data_dict = data
        else:
            continue
        # clusters
        if "lexical" in name or "cluster" in name:
            arr = data_dict.get("clusters") if data_dict else data_list
            for c in arr or []:
                if isinstance(c, dict):
                    clusters += [t for t in c.get("terms", []) if isinstance(t, str)]
                elif isinstance(c, str):
                    clusters.append(c)
        # market
        if "market" in name:
            arr = data_dict.get("vocabulary") if data_dict else data_list
            for v in arr or []:
                if isinstance(v, dict) and isinstance(v.get("vocabulary"), str):
                    market.append(v["vocabulary"])
                elif isinstance(v, str):
                    market.append(v)
        # semantic
        if "semantic" in name:
            for k in ["concepts", "targets", "use_cases", "scenes"]:
                vals = data_dict.get(k) if data_dict else []
                if isinstance(vals, list):
                    concept += [v for v in vals if isinstance(v, str)]
        # template
        if "template" in name:
            tmpl += [v for v in data_dict.get("templates", []) if isinstance(v, str)]
        # forbid
        if "forbid" in name:
            forbid_local += [v for v in data_dict.get("forbidden_words", []) if isinstance(v, str)]

    all_forbid = list({*FORBIDDEN, *forbid_local})
    def cap(xs, n): return "、".join(list(dict.fromkeys([x for x in xs if isinstance(x, str)]))[:n])
    text = f"語彙:{cap(clusters,6)} / 市場語:{cap(market,6)} / 構造:{cap(concept,5)} / 骨子:{cap(tmpl,3)}。"
    text += "自然な日本語で1〜2文、句点終止で書く。"
    return text, all_forbid

# ====== プロンプト ======
SYSTEM_PROMPT = (
    "あなたは日本語コピーライターです。"
    "楽天の商品画像ALTを自然な日本語で作成します。"
    "必ず1〜2文の自然文で句点「。」で終えること。\n"
    "名詞の羅列は禁止（例: iPhone14、iPhone15、ケーブル、充電）。"
    "文末は『〜できる』『〜です』『〜します』『〜に便利です』など丁寧語で終える。\n"
    "画像や写真、レビュー、ランキング、当店などのメタ語は禁止。\n"
)

def build_user_prompt(product, knowledge_text, forbid_words):
    forbid_txt = "、".join(sorted(set(forbid_words)))
    return (
        f"商品名: {product}\n"
        f"{knowledge_text}\n"
        f"禁止語: {forbid_txt}\n"
        "20行の自然文ALTを出力してください。各行は句点終止の1〜2文。"
    )

# ====== OpenAI呼び出し（安定版） ======
def call_openai_20_lines(client, model, temp, max_tokens, product, kb_text, forbid_words, retry=3, wait=6):
    user_prompt = build_user_prompt(product, kb_text, forbid_words)
    for attempt in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=temp,
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("空応答を検出")
            lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            return lines[:80]
        except Exception as e:
            print(f"⚠️ OpenAIエラー({attempt+1}/{retry}): {e}")
            time.sleep(wait)
    # fallback
    return [f"{product}は高品質な設計で、快適に使用できます。"] * 20

# ====== ローカル整形 ======
def looks_listy(s):
    if not s: return True
    if LEADING_ENUM_RE.search(s): return True
    if "。" not in s and "、" in s: return True
    return False

def rewrite_listy_to_sentence(s):
    t = LEADING_ENUM_RE.sub("", s)
    bits = [b.strip(" 。、・-—") for b in t.split("、") if len(b.strip()) > 1]
    if not bits: return s.strip() + "。"
    head = "、".join(bits[:2])
    tail = "、".join(bits[2:]) if len(bits) > 2 else ""
    cand = f"{head}を備え、{tail}でも使いやすい設計です。" if tail else f"{head}に対応し、日常を快適にします。"
    for ng in FORBIDDEN:
        cand = cand.replace(ng, "")
    if not cand.endswith("。"):
        cand += "。"
    return cand

def normalize_lines(lines):
    out = []
    for ln in lines:
        if not ln: continue
        ln = ln.strip()
        if looks_listy(ln):
            ln = rewrite_listy_to_sentence(ln)
        if not ln.endswith("。"):
            ln += "。"
        out.append(ln)
    return out

def soft_clip_sentence(t):
    t = WS_RE.sub(" ", t)
    if not t.endswith("。"):
        t += "。"
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
    for ng in FORBIDDEN:
        t = t.replace(ng, "")
    return t.strip()

def refine_20_lines(raw_lines):
    raw_lines = normalize_lines(raw_lines)
    refined = [soft_clip_sentence(x) for x in raw_lines if len(x) >= 20]
    uniq = list(dict.fromkeys(refined))
    while len(uniq) < 20:
        uniq.append(f"{uniq[-1]}より使いやすいデザインです。")
    return uniq[:20]

# ====== 出力 ======
def ensure_outdir(): os.makedirs(OUT_DIR, exist_ok=True)
def write_csv(path, products, data, prefix):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"{prefix}_{i+1}" for i in range(20)])
        for p, lines in zip(products, data):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

# ====== メイン ======
def main():
    print("🌸 ALT長文生成 v4.6（自然文＋安定生成＋リトライ）")
    client, model, temp, max_tokens = init_env_and_client()
    ensure_outdir()

    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    kb_text, forb = summarize_knowledge()
    all_raw, all_ref = [], []

    for p in tqdm(products, desc="🧠 生成中", total=len(products)):
        raw = call_openai_20_lines(client, model, temp, max_tokens, p, kb_text, forb)
        refined = refine_20_lines(raw)
        all_raw.append(raw[:20])
        all_ref.append(refined)
        time.sleep(0.8)

    write_csv(RAW_PATH, products, all_raw, "ALT_raw")
    write_csv(REF_PATH, products, all_ref, "ALT")

    # diff 出力
    with open(DIFF_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_ref_{i+1}" for i in range(20)])
        for p, raw, ref in zip(products, all_raw, all_ref):
            w.writerow([p] + raw[:20] + ref[:20])

    def avg_len(block):
        lens = [len(x) for lines in block for x in lines if x]
        return sum(lens) / len(lens) if lens else 0

    print("✅ 出力完了:")
    print(f"   - AI生出力: {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 平均文字数 raw={avg_len(all_raw):.1f}, refined={avg_len(all_ref):.1f}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
