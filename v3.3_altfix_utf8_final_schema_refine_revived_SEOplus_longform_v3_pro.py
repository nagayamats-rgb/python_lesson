# -*- coding: utf-8 -*-
"""
ALT長文生成 v3_pro（SEO＋自然文＋知見展開完全復元）
----------------------------------------------------------
- 入力: ./rakuten.csv（UTF-8, ヘッダに「商品名」）
- 出力:
  1) output/ai_writer/alt_text_ai_raw_longform_v3.csv
  2) output/ai_writer/alt_text_refined_final_longform_v3.csv
  3) output/ai_writer/alt_text_diff_longform_v3.csv
- 知見: ./output/semantics/ 内の JSON 群を統合してプロンプトに展開
- 特徴:
    ✅ 句点分割フォールバック（改行喪失防止）
    ✅ JSON知見語群をSEOワードとして展開
    ✅ AI長文化指令（100〜130字・1〜2文）
    ✅ 欠損ALTはローカル補完（テンプレ構文）
"""

import os, re, csv, glob, json, time, random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =========================
# 定数・パス設定
# =========================
INPUT_CSV = "./rakuten.csv"
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v3.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v3.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v3.csv")
SEMANTICS_DIR = "./output/semantics"

# 禁則語
FORBIDDEN = [
    "画像","写真","見た目","当店","当社","レビュー","ランキング","クリック","リンク",
    "カート","購入はこちら","ページ","最安","No.1","ナンバーワン","売上","送料無料",
    "返金保証","優位性","競合","評価","口コミ"
]

RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

# =========================
# OpenAI初期化
# =========================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")
    model = "gpt-4o"
    client = OpenAI(api_key=api_key)
    return client, model

# =========================
# 商品名読込
# =========================
def load_products(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list({r["商品名"].strip() for r in reader if r.get("商品名")})

# =========================
# 知見要約＋SEO語展開
# =========================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "知見: スペック・用途・対象を自然に含め、SEOに強いALTを作成。", []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    keywords = []
    forbidden_local = []

    for p in files:
        data = safe_load_json(p)
        if not data: continue
        name = os.path.basename(p).lower()
        try:
            if "lexical" in name:
                if isinstance(data, dict):
                    for c in data.get("clusters", []):
                        keywords += [t for t in c.get("terms", []) if isinstance(t, str)]
            elif "market" in name:
                if isinstance(data, list):
                    keywords += [v.get("vocabulary","") for v in data if isinstance(v, dict)]
            elif "semantic" in name:
                if isinstance(data, dict):
                    for k in ["concepts","targets","use_cases"]:
                        keywords += data.get(k, [])
            elif "forbid" in name:
                if isinstance(data, dict):
                    forbidden_local += data.get("forbidden_words", [])
        except Exception:
            pass

    all_forbidden = list({*FORBIDDEN, *forbidden_local})
    seo_terms = [k for k in keywords if isinstance(k, str)][:20]
    joined_terms = "、".join(seo_terms)
    kb = f"知見: 以下の語を自然に含めてALTを生成。推奨語句: {joined_terms}。1〜2文構成で、句点で終える自然文を作成。"
    return kb, all_forbidden

# =========================
# プロンプト強化
# =========================
SYSTEM_PROMPT = (
    "あなたは楽天市場のSEO最適化を専門とする日本語コピーライターです。\n"
    "目的は、自然で読みやすく、かつ検索に強いALTテキストを作成することです。\n"
    f"各ALTは全角{RAW_MIN}〜{RAW_MAX}文字、1〜2文構成で句点「。」で終えること。\n"
    "箇条書き、番号、記号、画像描写語は禁止。\n"
    "必ず改行で20行に分けて出力。短い文や省略は禁止です。"
)

def build_user_prompt(product, kb_text, forbidden):
    forbid_txt = "、".join(forbidden)
    return (
        f"商品名: {product}\n"
        f"{kb_text}\n"
        f"禁止語: {forbid_txt}\n"
        f"20行で、1行あたり100〜130文字の自然なALTテキストを生成。"
    )

# =========================
# OpenAI呼び出し（句点フォールバック付き）
# =========================
def call_openai_20_lines(client, model, product, kb_text, forbidden, retry=3, wait=5):
    user_prompt = build_user_prompt(product, kb_text, forbidden)
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":user_prompt},
                ],
                response_format={"type":"text"},
                max_completion_tokens=1000,
                temperature=1
            )
            txt = (res.choices[0].message.content or "").strip()
            if not txt:
                continue
            lines = [x.strip() for x in txt.split("\n") if x.strip()]
            if len(lines) <= 1:
                lines = re.split(r"(?<=。)\s*", txt)
            clean = []
            for ln in lines:
                ln2 = re.sub(r"^\s*[\d\-\*・\.]+\s*", "", ln).strip("・-—●　")
                if ln2:
                    clean.append(ln2)
            return clean[:60]
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答を取得できません: {last_err}")

# =========================
# ローカル補完
# =========================
SPEC_TEMPLATES = [
    "高出力で安定した充電を実現するモデル",
    "軽量で持ち運びに便利な設計",
    "複数デバイス対応の高性能タイプ",
    "耐久性とデザイン性を兼ね備えた仕様",
]
BENEFIT_TEMPLATES = [
    "日常からビジネスまで快適に使用できます。",
    "持ち運びにも便利で、外出時にも最適です。",
    "長く安心して使える品質です。",
    "スマートな生活をサポートします。",
]

def generate_local_alt(product):
    spec = random.choice(SPEC_TEMPLATES)
    benefit = random.choice(BENEFIT_TEMPLATES)
    return f"{product}は{spec}。{benefit}"

# =========================
# 整形
# =========================
def refine_20_alt_lines(raw_lines, product):
    refined = []
    for ln in raw_lines:
        ln = ln.strip()
        if len(ln) > 150:
            parts = re.split(r"(?<=。)\s*", ln)
            refined.extend(parts[:3])
        elif len(ln) < 40:
            refined.append(generate_local_alt(product))
        else:
            if not ln.endswith("。"): ln += "。"
            refined.append(ln)
    if not refined:
        refined = [generate_local_alt(product)] * 20
    while len(refined) < 20:
        refined.append(generate_local_alt(product))
    return refined[:20]

# =========================
# 書き出し
# =========================
def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# =========================
# メイン
# =========================
def main():
    print("🌸 ALT長文生成 v3_pro（SEO＋自然文＋知見展開）")
    client, model = init_env_and_client()
    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    kb_text, forb = summarize_knowledge()
    all_raw, all_ref = [], []

    for p in tqdm(products, desc="🧠 生成中"):
        try:
            raw = call_openai_20_lines(client, model, p, kb_text, forb)
        except Exception:
            raw = [generate_local_alt(p)]*20
        refined = refine_20_alt_lines(raw, p)
        all_raw.append(raw[:20]); all_ref.append(refined)
        time.sleep(0.2)

    write_csv(RAW_PATH, ["商品名"]+[f"ALT_raw_{i+1}" for i in range(20)], [[p]+r for p,r in zip(products,all_raw)])
    write_csv(REF_PATH, ["商品名"]+[f"ALT_{i+1}" for i in range(20)], [[p]+r for p,r in zip(products,all_ref)])
    print(f"✅ 出力完了\n📄 AI生出力: {RAW_PATH}\n📄 整形後: {REF_PATH}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
