# -*- coding: utf-8 -*-
"""
semantic_extractor_rebuilder_v1_1_unified_fixed.py
KOTOHA: ローカル知見（semantics）再構築ユニファイド版（環境固定対応）

■ 目的
- /sauce/rakuten.csv と /sauce/yahoo.csv の商品名から検索クエリを生成
- 楽天商品検索API(20220601)へ 7〜15ページ範囲でアクセスしてテキスト収集
- 収集テキストから以下の知見JSON群を再構築して ./output/semantics へ出力
  - lexical_clusters_YYYYmmdd_HHMMSS.json
  - market_vocab_YYYYmmdd_HHMMSS.json
  - structured_semantics_YYYYmmdd_HHMMSS.json
  - styled_persona_YYYYmmdd_HHMMSS.json（既存があれば継承、なければ汎用Personaを生成）
  - normalized_YYYYmmdd_HHMMSS.json（禁則語など）
  - template_composer.json（汎用テンプレ骨子）
  - knowledge_fused_structured.json（ライターが読みやすい統合知見）
  - knowledge_fused_text.txt（人間可読の要約）

■ .env 固定（プロジェクト全体で準拠）
RAKUTEN_API_BASE_URL=https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601
RAKUTEN_APP_ID=...
YAHOO_API_BASE_URL=https://shopping.yahooapis.jp/ShoppingWebService/V3/itemSearch
YAHOO_APP_ID=...
OPENAI_API_KEY=...
OPENAI_MODEL="gpt-5"
OPENAI_MODE="chat"
OPENAI_TEMPERATURE="0.9"
OPENAI_MAX_TOKENS="2000"
OPENAI_TOP_P="0.9"
OPENAI_PRESENCE_PENALTY="0.4"
OPENAI_FREQUENCY_PENALTY="0.3"
USE_KOTOHA_PERSONA="ON"

※ 本スクリプトは OpenAI を呼びません（上流知見構築のみ）。
※ MOCK_MODE="ON" で実APIを叩かず手元テキストから擬似処理。

"""

import os
import re
import csv
import sys
import json
import time
import glob
import math
import textwrap
import datetime
from collections import Counter, defaultdict

# ========= 依存の自己解決 =========
def ensure_packages(pkgs):
    import importlib
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)
    if missing:
        print(f"📦 Installing: {', '.join(missing)} ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("✅ Install done.")

ensure_packages([
    "python-dotenv", "requests", "tqdm", "scikit-learn", "numpy"
])

from dotenv import load_dotenv
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ========= パス & 定数 =========
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
SAUCE_DIR = os.path.join(BASE_DIR, "sauce")
OUT_DIR = os.path.join(BASE_DIR, "output", "semantics")
os.makedirs(OUT_DIR, exist_ok=True)

# 入力CSV固定（プロジェクト全体ルール）
RAKUTEN_CSV = os.path.join(SAUCE_DIR, "rakuten.csv")   # UTF-8, ヘッダに「商品名」
YAHOO_CSV   = os.path.join(SAUCE_DIR, "yahoo.csv")     # UTF-8, ヘッダに「name」

# 収集対象ページ（楽天API）
RAKUTEN_PAGE_START = 7
RAKUTEN_PAGE_END   = 15
RAKUTEN_HITS       = 30  # 1ページあたり件数（上限はAPI仕様に従う）

# 収集のレート制御
API_DELAY_SEC = 0.5  # 過度な負荷を避ける軽スロットリング

# ========= ユーティリティ =========
def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_read_csv_rows(path, encoding="utf-8"):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # 変な制御文字など除去
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return s

def split_keywords(name: str):
    """
    商品名を素朴にトークン化して重要そうなキーワードを返す
    - 記号除去
    - 半角/全角数字・英字・カタカナ/ひらがな/漢字を単純抽出
    """
    s = normalize_text(name)
    s = re.sub(r"[【】\[\]\(\)（）/｜|／・\-—_:+：;；~〜!！?？,，。.\u3000]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    return toks[:10]  # 多すぎるとノイズになるので上位10語に制限

def generate_queries(product_name: str):
    """
    クエリ生成：商品名から 2〜3 パターンほど
    - 完全名
    - 主要キーワード上位3〜5語をスペース連結
    - 型番らしき英数を優先
    """
    toks = split_keywords(product_name)
    full = normalize_text(product_name)
    # 型番候補（英数字連続）
    model_like = [t for t in toks if re.search(r"[A-Za-z0-9]{3,}", t)]
    short = " ".join(toks[:5]) if toks else full
    queries = [full]
    if model_like:
        queries.append(" ".join(unique_preserve(model_like[:3])))
    if short and short not in queries:
        queries.append(short)
    return unique_preserve([q for q in queries if q])

def load_persona_if_any():
    """
    既存 persona を拾う。なければデフォルトを返す
    """
    # 既存の styled_persona_* を探索
    cand = sorted(glob.glob(os.path.join(OUT_DIR, "styled_persona_*.json")), reverse=True)
    if cand:
        try:
            with open(cand[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # デフォルト
    return {
        "tone": {
            "style": "明快・誠実・過剰表現を避ける",
            "register": "常体と敬体を自然に使い分ける",
            "constraints": [
                "絵文字・特殊記号・画像描写語は使わない",
                "競合比較や誇大表現は避ける",
                "ユーザーの理解を助ける語彙を選ぶ"
            ]
        },
        "persona": {
            "role": "SEOに強い日本語コピーライター",
            "focus": ["自然文", "商品理解", "サイト内SEO最適化"]
        }
    }

# ========= .env 読み込み =========
load_dotenv(override=True)
MOCK_MODE = os.getenv("MOCK_MODE", "").strip().upper() == "ON"

RAKUTEN_API_BASE_URL = os.getenv("RAKUTEN_API_BASE_URL", "").strip()
RAKUTEN_APP_ID       = os.getenv("RAKUTEN_APP_ID", "").strip()
YAHOO_API_BASE_URL   = os.getenv("YAHOO_API_BASE_URL", "").strip()
YAHOO_APP_ID         = os.getenv("YAHOO_APP_ID", "").strip()

USE_KOTOHA_PERSONA   = os.getenv("USE_KOTOHA_PERSONA", "ON").strip().upper()

if not RAKUTEN_API_BASE_URL or not RAKUTEN_APP_ID:
    if not MOCK_MODE:
        print("❌ .env の楽天API設定が不足しています。RAKUTEN_API_BASE_URL / RAKUTEN_APP_ID を確認してください。")
        sys.exit(1)

# ========= クローラ（楽天API） =========
def rakuten_search(keyword: str, page: int, hits: int = RAKUTEN_HITS):
    """
    楽天商品検索API 20220601
    https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601
    """
    params = {
        "format": "json",
        "applicationId": RAKUTEN_APP_ID,
        "keyword": keyword,
        "page": page,
        "hits": hits,
        # 並び順など必要に応じて
        # "sort": "+reviewCount"
    }
    try:
        r = requests.get(RAKUTEN_API_BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def extract_texts_from_rakuten_response(data: dict):
    """
    楽天APIレスポンスからテキスト群を抽出
    - itemName, itemCaption, catchcopy(あれば), shopName など
    """
    texts = []
    if not isinstance(data, dict):
        return texts
    items = data.get("Items") or []
    for it in items:
        # 20220601は {"Item": {...}} 構造
        node = it.get("Item") if isinstance(it, dict) else None
        if not isinstance(node, dict):
            continue
        fields = [
            node.get("itemName"),
            node.get("itemCaption"),
            node.get("catchcopy"),
            node.get("shopName"),
        ]
        for t in fields:
            t = normalize_text(t)
            if t and len(t) > 3:
                texts.append(t)
    return texts

# ========= 収集オーケストレーション =========
def load_product_names():
    r_rows = safe_read_csv_rows(RAKUTEN_CSV, "utf-8")
    y_rows = safe_read_csv_rows(YAHOO_CSV,   "utf-8")
    rak_names = [normalize_text(r.get("商品名") or "") for r in r_rows if (r.get("商品名") or "").strip()]
    yah_names = [normalize_text(r.get("name") or "") for r in y_rows if (r.get("name") or "").strip()]
    prods = unique_preserve(rak_names + yah_names)
    return [p for p in prods if p]

def collect_corpus_from_api(product_names):
    """
    各商品名 -> 2〜3個のクエリ -> 楽天API 7〜15ページ巡回 -> テキスト収集
    MOCK_MODE=ON の場合は既存 semantics のテキストや商品名から合成
    """
    corpus = []
    if MOCK_MODE:
        print("🧪 MOCK_MODE=ON: APIコールをスキップし、ローカルで簡易コーパスを合成します。")
        for nm in product_names:
            toks = split_keywords(nm)
            base = f"{nm} 高品質 使いやすい 互換性 充電 便利 軽量 耐久性 日常 利用シーン 多様"
            corpus.append(normalize_text(base + " " + " ".join(toks)))
        return corpus

    for nm in tqdm(product_names, desc="🔎 楽天API収集", total=len(product_names)):
        queries = generate_queries(nm)
        for q in queries:
            for page in range(RAKUTEN_PAGE_START, RAKUTEN_PAGE_END + 1):
                data = rakuten_search(q, page, RAKUTEN_HITS)
                if "error" in data:
                    # 軽く待機して次へ
                    time.sleep(API_DELAY_SEC * 2)
                    continue
                texts = extract_texts_from_rakuten_response(data)
                corpus.extend(texts)
                time.sleep(API_DELAY_SEC)
    return corpus

# ========= 特徴抽出（TF-IDFベース） =========
def build_lexical_clusters(corpus, top_k=300):
    """
    コーパスから重要語を抽出して単純なクラスタ表現を作る
    - 日本語の形態素解析は使わず、N-gramで近似（2-gram/3-gram）
    - TF-IDF上位を「語彙候補」として clusters=[{"terms":[...]}] を1クラスタ返却
    """
    if not corpus:
        return {"clusters": []}
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        min_df=2,
        max_features=2000
    )
    X = vectorizer.fit_transform(corpus)
    feats = vectorizer.get_feature_names_out()
    # tf-idf の列毎最大値で重要度を近似
    import numpy as np
    scores = np.asarray(X.max(axis=0)).ravel()
    idx = scores.argsort()[::-1][:top_k]
    terms = [feats[i] for i in idx]
    # 余計なスペース・記号を掃除
    terms = [normalize_text(t) for t in terms if len(t.strip()) >= 2]
    return {"clusters": [{"terms": terms}]}

def build_market_vocab(corpus, top_k=150):
    """
    市場語彙：頻出の単語/フレーズを抽出（空白分割＆n-gram）
    """
    tokens = []
    for s in corpus:
        s2 = re.sub(r"[【】\[\]\(\)（）/｜|／・\-—_:+：;；~〜!！?？,，。.\u3000]", " ", s)
        parts = [p for p in s2.split() if len(p) >= 2]
        tokens.extend(parts)
    freq = Counter(tokens)
    vocab = [w for w, _ in freq.most_common(top_k)]
    return vocab

def build_structured_semantics(corpus):
    """
    構造化知見（超軽量ルールベース）
    - features: 性能/機能に関わる語彙
    - scenes: 利用シーン
    - targets: 対象/対応
    - benefits: ベネフィット
    """
    feats_cue = ["耐久", "軽量", "薄型", "高速", "急速", "強化", "保護", "防滴", "防水", "防塵", "静音", "低遅延", "安定"]
    scenes_cue = ["通勤", "通学", "出張", "旅行", "在宅", "ビジネス", "スポーツ", "屋外", "オフィス", "自宅", "寝室", "デスク周り"]
    targets_cue = ["iPhone", "Android", "iPad", "Galaxy", "Apple Watch", "MacBook", "ノートPC", "Switch", "PS5"]
    benefits_cue = ["時短", "快適", "省スペース", "省電力", "持ち運びやすい", "扱いやすい", "安心", "長持ち", "コスパ"]

    txt = " ".join(corpus[:5000])  # 過大な長さを避ける
    def pick(cues):
        hits = [w for w in cues if w in txt]
        return unique_preserve(hits)

    return {
        "concepts": ["商品スペック", "コアコンピタンス", "対象ユーザー", "利用シーン", "ベネフィット"],
        "features": pick(feats_cue),
        "scenes": pick(scenes_cue),
        "targets": pick(targets_cue),
        "benefits": pick(benefits_cue),
    }

def default_normalized():
    return {
        "forbidden_words": [
            "画像", "写真", "見た目", "上の画像", "下の写真", "当店", "当社", "レビュー", "ランキング",
            "クリック", "こちら", "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
            "リンク", "ページ", "カート", "購入はこちら", "返金保証", "送料無料（確約）",
        ]
    }

def default_template():
    return {
        "hints": [
            "スペック→強み→対象→シーン→便益",
            "互換性・対応機種の明記",
            "サイズ感・重量・素材などの客観情報",
            "“何がどう便利か”を1文で伝える"
        ]
    }

# ========= 統合知見（ライター向け） =========
def fuse_knowledge(lexical, market, structured):
    """
    ライターに渡しやすい “構造化＆テキスト” の2系統を作る
    """
    payload = {
        "templates": default_template().get("hints", []),
        "market": market or [],
        "features": structured.get("features", []),
        "scenes": structured.get("scenes", []),
        "targets": structured.get("targets", []),
        "benefits": structured.get("benefits", []),
    }
    # テキスト版
    lines = []
    if payload["templates"]:
        lines.append("骨子: " + " / ".join(payload["templates"][:3]))
    if payload["market"]:
        lines.append("市場語: " + "、".join(payload["market"][:12]))
    if payload["features"]:
        lines.append("機能: " + "、".join(payload["features"][:10]))
    if payload["targets"]:
        lines.append("対応: " + "、".join(payload["targets"][:10]))
    if payload["scenes"]:
        lines.append("シーン: " + "、".join(payload["scenes"][:10]))
    if payload["benefits"]:
        lines.append("便益: " + "、".join(payload["benefits"][:10]))
    text = "。".join(lines) + ("。" if lines else "")
    return payload, text

# ========= メイン =========
def main():
    print("🧩 semantic_extractor_rebuilder_v1.1（unified fixed）起動")
    # 1) 入力ロード
    products = load_product_names()
    print(f"📦 対象商品数: {len(products)}")

    # 2) コーパス収集
    corpus = collect_corpus_from_api(products)
    print(f"🧾 収集テキスト数: {len(corpus)}")

    # 3) 特徴抽出
    lexical = build_lexical_clusters(corpus, top_k=300)
    market  = build_market_vocab(corpus, top_k=150)
    struct  = build_structured_semantics(corpus)
    persona = load_persona_if_any()
    normalz = default_normalized()
    tmpl    = default_template()

    # 4) 統合知見
    fused_structured, fused_text = fuse_knowledge(lexical, market, struct)

    # 5) 出力
    ts = now_ts()
    path_lexical = os.path.join(OUT_DIR, f"lexical_clusters_{ts}.json")
    path_market  = os.path.join(OUT_DIR, f"market_vocab_{ts}.json")
    path_struct  = os.path.join(OUT_DIR, f"structured_semantics_{ts}.json")
    path_persona = os.path.join(OUT_DIR, f"styled_persona_{ts}.json")
    path_norm    = os.path.join(OUT_DIR, f"normalized_{ts}.json")
    path_tmpl    = os.path.join(OUT_DIR, "template_composer.json")  # テンプレは安定ファイルで持つ
    path_fused_s = os.path.join(OUT_DIR, "knowledge_fused_structured.json")
    path_fused_t = os.path.join(OUT_DIR, "knowledge_fused_text.txt")

    with open(path_lexical, "w", encoding="utf-8") as f:
        json.dump(lexical, f, ensure_ascii=False, indent=2)
    with open(path_market, "w", encoding="utf-8") as f:
        json.dump(market, f, ensure_ascii=False, indent=2)
    with open(path_struct, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)
    with open(path_persona, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)
    with open(path_norm, "w", encoding="utf-8") as f:
        json.dump(normalz, f, ensure_ascii=False, indent=2)
    # template_composer は常に上書き（安定ファイル）
    with open(path_tmpl, "w", encoding="utf-8") as f:
        json.dump(tmpl, f, ensure_ascii=False, indent=2)
    with open(path_fused_s, "w", encoding="utf-8") as f:
        json.dump(fused_structured, f, ensure_ascii=False, indent=2)
    with open(path_fused_t, "w", encoding="utf-8") as f:
        f.write(fused_text or "")

    print(f"✅ {os.path.basename(path_lexical)}")
    print(f"✅ {os.path.basename(path_market)}")
    print(f"✅ {os.path.basename(path_struct)}")
    print(f"✅ {os.path.basename(path_persona)}")
    print(f"✅ {os.path.basename(path_norm)}")
    print(f"✅ {os.path.basename(path_tmpl)}")
    print(f"✅ {os.path.basename(path_fused_s)}")
    print(f"✅ {os.path.basename(path_fused_t)}")
    print("🎯 完了: /output/semantics に知見JSONを再構築しました。")
    print("   → ライター側からは knowledge_fused_structured.json / knowledge_fused_text.txt を読み込むのが最短です。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
