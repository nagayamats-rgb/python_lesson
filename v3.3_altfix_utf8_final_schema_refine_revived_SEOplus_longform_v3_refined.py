# -*- coding: utf-8 -*-
"""
ALT長文生成 v3（SEO＋自然文＋かんなめ補完）
-------------------------------------------------
- 入力: ./rakuten.csv（UTF-8, ヘッダ「商品名」）
- 出力:
    1) output/ai_writer/alt_text_ai_raw_longform_v3.csv
    2) output/ai_writer/alt_text_refined_final_longform_v3.csv
    3) output/ai_writer/alt_text_diff_longform_v3.csv
- 知見: ./output/semantics/ 内の JSON 群を要約活用
- ローカル補完層: local_refiner（かんなめ）
"""

import os, re, csv, glob, json, time, random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =========================
# 定数・入出力パス
# =========================
INPUT_CSV = "./rakuten.csv"
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v3.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v3.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v3.csv")
SEMANTICS_DIR = "./output/semantics"

# 禁則語
FORBIDDEN = [
    "画像","写真","見た目","当店","当社","ランキング","レビュー","クリック","リンク","カート",
    "購入はこちら","ページ","最安","No.1","ナンバーワン","売上","送料無料","返金保証","優位性","競合","評価","口コミ"
]

# 文字数範囲
RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

# =========================
# OpenAI初期化
# =========================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
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
# 知見サマリ
# =========================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "知見: キーワード・用途・対象・スペックを自然に含めて2文以内で。", []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    clusters, market, semantics, forbidden_local = [], [], [], []
    for p in files:
        data = safe_load_json(p)
        if not data:
            continue
        name = os.path.basename(p).lower()
        try:
            if "lexical" in name and isinstance(data, dict):
                clusters.extend(sum([c.get("terms", []) for c in data.get("clusters", [])], []))
            elif "market" in name and isinstance(data, list):
                market.extend([v.get("vocabulary", "") for v in data if isinstance(v, dict)])
            elif "semantic" in name and isinstance(data, dict):
                for k in ["concepts", "targets", "use_cases"]:
                    semantics.extend(data.get(k, []))
            elif "forbid" in name and isinstance(data, dict):
                forbidden_local.extend(data.get("forbidden_words", []))
        except Exception:
            pass
    all_forbidden = list({*FORBIDDEN, *forbidden_local})
    kb = f"知見: {', '.join(list(set(clusters + market + semantics))[:15])}。自然でSEOに強い文章を生成。"
    return kb, all_forbidden

# =========================
# AIプロンプト
# =========================
SYSTEM_PROMPT = (
    "あなたは楽天市場の商品画像ALTテキストを作成するプロの日本語コピーライターです。"
    "目的は、SEOに強く自然な文章を生成することです。"
    "禁止: 画像や写真の描写語、店舗メタ語、競合比較表現。\n"
    f"各ALTは全角{RAW_MIN}〜{RAW_MAX}文字、句点「。」で終える自然文にしてください。\n"
    "JSONや番号は使わず、20行のテキストのみを出力。"
)

def build_user_prompt(product, kb_text, forbidden):
    forbid_txt = "、".join(forbidden)
    hint = "構成ヒント: スペック→強み→対象→シーン→便益。"
    return f"商品名: {product}\n{kb_text}\n{hint}\n禁止語: {forbid_txt}"

# =========================
# OpenAI呼び出し
# =========================
def call_openai_20_lines(client, model, product, kb_text, forbidden, retry=3, wait=5):
    user_prompt = build_user_prompt(product, kb_text, forbidden)
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content":user_prompt}],
                response_format={"type":"text"},
                max_completion_tokens=1000,
                temperature=1
            )
            txt = (res.choices[0].message.content or "").strip()
            if txt:
                lines = [re.sub(r"^\s*[\d\-\*・\.]+\s*", "", x).strip() for x in txt.split("\n") if x.strip()]
                return lines[:60]
        except Exception as e:
            time.sleep(wait)
    return []

# =========================
# かんなめ（local_refiner）統合
# =========================
FORBIDDEN_LOCAL = FORBIDDEN
SPEC_TEMPLATES = [
    "高出力で安定した充電を実現するモデル",
    "コンパクトながら高性能な設計",
    "複数デバイスに対応した多機能仕様",
    "軽量かつ持ち運びやすいデザイン",
    "安全保護機能を備えた高品質モデル",
    "USB-C対応で汎用性に優れたタイプ",
    "マグネットで簡単に装着できる設計",
    "スタンド機能付きでデスク作業にも便利",
    "耐久性の高い素材を採用した設計",
    "最新の高速通信規格に対応したモデル",
]
CONNECTORS = ["で", "により", "を備え", "を活かして", "を搭載し"]
BENEFIT_TEMPLATES = [
    "ビジネスから日常使いまで快適にサポートします。",
    "外出先でもストレスなく使用できます。",
    "長時間の使用にも安定したパフォーマンスを発揮します。",
    "スマートな暮らしをサポートする便利アイテムです。",
    "オフィスや家庭でも幅広く活躍します。",
    "どなたでも直感的に使いやすい設計です。",
    "使うたびに快適さを感じられる仕上がりです。",
    "急な充電にも素早く対応できます。",
    "持ち運びにも便利で出張や旅行にも最適です。",
    "高品質な素材で長く安心して使えます。",
]

def generate_local_alt(product):
    spec = random.choice(SPEC_TEMPLATES)
    connector = random.choice(CONNECTORS)
    benefit = random.choice(BENEFIT_TEMPLATES)
    text = f"{product}は{spec}{connector}{benefit}"
    for ng in FORBIDDEN_LOCAL:
        text = text.replace(ng, "")
    if not text.endswith("。"):
        text += "。"
    return text.strip()

def refine_alt_text(line, product):
    if not line or len(line.strip()) < 40:
        return generate_local_alt(product)
    line = re.sub(r"\s+", " ", line.strip())
    if not line.endswith("。"):
        line += "。"
    for ng in FORBIDDEN_LOCAL:
        line = line.replace(ng, "")
    if random.random() < 0.35:
        for a, b in [("です。","になります。"),("します。","できます。"),("できます。","しやすいです。")]:
            if line.endswith(a):
                line = line[:-len(a)] + b
                break
    return line

def refine_20_alt_lines(raw_lines, product):
    refined = [refine_alt_text(x, product) for x in raw_lines if x.strip()]
    while len(refined) < 20:
        refined.append(generate_local_alt(product))
    return refined[:20]

# =========================
# 出力
# =========================
def write_csv(path, header, rows):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# =========================
# メイン処理
# =========================
def main():
    print("🌸 ALT長文生成 v3＋かんなめ補完 開始")
    client, model = init_env_and_client()
    products = load_products(INPUT_CSV)
    kb_text, forb = summarize_knowledge()
    all_raw, all_ref = [], []

    for p in tqdm(products, desc="🧠 生成中"):
        raw = call_openai_20_lines(client, model, p, kb_text, forb)
        if not raw:
            raw = [generate_local_alt(p)]*20
        refined = refine_20_alt_lines(raw, p)
        all_raw.append(raw[:20]); all_ref.append(refined)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(RAW_PATH, ["商品名"]+[f"ALT_raw_{i+1}" for i in range(20)],
              [[p]+r for p,r in zip(products,all_raw)])
    write_csv(REF_PATH, ["商品名"]+[f"ALT_{i+1}" for i in range(20)],
              [[p]+r for p,r in zip(products,all_ref)])
    print("✅ 出力完了")
    print(f"📄 AI生出力: {RAW_PATH}\n📄 整形後: {REF_PATH}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
