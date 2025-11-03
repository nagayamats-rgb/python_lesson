# -*- coding: utf-8 -*-
"""
semantic_crawler_v2_traffic_conversion_dual.py
- 目的:
  1) rakuten.csv / yahoo.csv の商品名から想起クエリを生成
  2) 楽天API / Yahoo API（必要ならHTMLフォールバック）で 7〜15ページ相当を収集
  3) 形態素解析 + TF-IDF で頻出語抽出
  4) Traffic語 / Conversion語 に二層分類してJSON出力
- 入力:
  /Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv  （UTF-8, ヘッダ: 商品名）
  /Users/tsuyoshi/Desktop/python_lesson/sauce/yahoo.csv    （UTF-8, ヘッダ: 商品名 or name）
- 設定: .env（プロジェクト全体で固定）
  RAKUTEN_API_BASE_URL
  RAKUTEN_APP_ID
  YAHOO_API_BASE_URL
  YAHOO_APP_ID
  OPENAI_*（未使用可）
- 出力:
  ./output/semantics/raw_crawl_{timestamp}.json  … 素材＋集計
  ./output/semantics/traffic_conversion_{timestamp}.json … 二層語彙セット（ライター取り込み向け）
"""

import os
import re
import csv
import json
import time
import math
import random
import pathlib
from datetime import datetime
from collections import Counter, defaultdict

# -------------------------
# 依存モジュールの自己解決
# -------------------------
def _ensure(mod, pip_name=None):
    try:
        __import__(mod)
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or mod, "-q"], check=False)

_ensure("requests")
_ensure("beautifulsoup4", "beautifulsoup4")
_ensure("tqdm")
_ensure("python-dotenv", "python-dotenv")
_ensure("janome")
_ensure("scikit-learn", "scikit-learn")

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# 定数・パス
# -------------------------
BASE = pathlib.Path(__file__).resolve().parent
SAUCE = BASE / "sauce"
INPUT_RAKUTEN = SAUCE / "rakuten.csv"
INPUT_YAHOO   = SAUCE / "yahoo.csv"
OUT_DIR = BASE / "output" / "semantics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# .env読込
load_dotenv(override=True)
RAKUTEN_API = os.getenv("RAKUTEN_API_BASE_URL", "").strip()
RAKUTEN_APP = os.getenv("RAKUTEN_APP_ID", "").strip()
YAHOO_API   = os.getenv("YAHOO_API_BASE_URL", "").strip()
YAHOO_APP   = os.getenv("YAHOO_APP_ID", "").strip()

# 収集レンジ（検索7〜15ページ相当）
RAKUTEN_PAGES = list(range(7, 16))   # page=7..15
YAHOO_PAGES   = list(range(7, 16))   # start=(page-1)*10+1, results=10

# -------------------------
# 前処理・ユーティリティ
# -------------------------
JP_DIGIT = "０１２３４５６７８９"
EN_DIGIT = "0123456789"
D2Z = str.maketrans(EN_DIGIT, JP_DIGIT)
Z2D = str.maketrans(JP_DIGIT, EN_DIGIT)

STOPWORDS = set("""
これ それ あれ ここ そこ あそこ 私 あなた する なる いる ある ため
そして また ので が と に へ で を は も の より や から まで にて
です ます でした でしたら ません でしたが できる でき
""".split())

# ルールシード（初期分類の軸）
TRAFFIC_SEEDS = set("""
iPhone iPad Android Galaxy Xperia Pixel Apple Watch AirPods MagSafe
ケース フィルム ガラス 充電 ケーブル アダプタ ドック ワイヤレス PD QC3.0
USB Type-C Lightning microUSB 15W 20W 30W 60W 100W 3A 5A 10Gbps
防水 防塵 耐衝撃 耐久 抗菌 薄型 軽量 マグネット 折りたたみ スタンド
""".split())

CONVERSION_SEEDS = set("""
人気 売れ筋 高評価 安心 公式 返品保証 ギフト プレゼント おすすめ 便利
ビジネス 用途 幅広い シーン 旅行 出張 在宅 ワーク 学生
快適 使いやすい 長持ち 省スペース コンパクト 静音 高品質
""".split())

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_products(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    products = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        key = "商品名" if "商品名" in headers else ("name" if "name" in headers else None)
        if not key:
            return []
        for r in reader:
            nm = normalize_text(r.get(key, ""))
            if nm:
                products.append(nm)
    # 重複除去（順序維持）
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

def generate_queries(product: str) -> list:
    """
    商品名から想起クエリを数本生成（過剰にならない範囲）。
    - モデル名/容量/色っぽいトークンを抽出しつつ原型を残す
    """
    base = product
    # 型番/数字/容量/インチ/色の候補抽出（緩め）
    toks = re.findall(r"[A-Za-z0-9\-+]+|[０-９]+|[0-9]+(W|w|A|a|mm|MM|gb|GB|G|g|inch|インチ)?", product)
    toks = [t.translate(Z2D) for t in toks if t]
    keybits = [k for k in toks if len(k) >= 2][:4]
    # クエリ候補
    qs = [base]
    if keybits:
        qs.append(base + " " + " ".join(keybits))
    if len(base) > 12:
        qs.append(base.split()[0])
    # 固定の広義語を混ぜる（露出拾い）
    qs.append(base + " 充電")
    qs.append(base + " ケース")
    # ユニーク化
    out, seen = [], set()
    for q in qs:
        qn = normalize_text(q)
        if qn and qn not in seen:
            out.append(qn)
            seen.add(qn)
    return out[:4]

# -------------------------
# 楽天 / Yahoo API 呼び出し
# -------------------------
def rakuten_search(query: str, page: int, retry=2, timeout=20):
    params = {
        "keyword": query,
        "applicationId": RAKUTEN_APP,
        "page": page,
        "hits": 30,
        "format": "json",
        "sort": "+affiliateRate"  # 露出寄り
    }
    url = RAKUTEN_API
    for _ in range(retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1.5)
    return None

def yahoo_search(query: str, page: int, retry=2, timeout=20):
    # v3 itemSearch: start（1-based）, results
    start = (page - 1) * 10 + 1
    params = {
        "appid": YAHOO_APP,
        "query": query,
        "results": 10,
        "start": start,
        "sort": "-score"  # 関連度
    }
    url = YAHOO_API
    for _ in range(retry):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1.5)
    return None

def fetch_html_text(url: str, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # タイトル + 説明 + 見出し
        parts = []
        title = soup.find("title")
        if title: parts.append(title.get_text(" ", strip=True))
        for sel in ["meta[name='description']", "meta[property='og:description']"]:
            m = soup.select_one(sel)
            if m and m.get("content"):
                parts.append(m["content"])
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt and len(txt) > 5:
                parts.append(txt)
        return " ".join(parts)
    except Exception:
        return ""

# -------------------------
# 形態素解析・語彙抽出
# -------------------------
_tokenizer = Tokenizer()

def tokenize(text: str):
    words = []
    for t in _tokenizer.tokenize(text):
        base = t.base_form if t.base_form != "*" else t.surface
        base = base.strip()
        if not base or base in STOPWORDS:
            continue
        # 記号・英字のみ等を抑制
        if re.fullmatch(r"[\W_]+", base):
            continue
        words.append(base)
    return words

def tfidf_top_terms(sentences, top_k=200):
    if not sentences:
        return []
    vec = TfidfVectorizer(tokenizer=tokenize, max_df=0.9, min_df=2)
    try:
        X = vec.fit_transform(sentences)
    except Exception:
        return []
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = list(zip(terms, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:top_k]]

def classify_terms(terms):
    """シード＋語尾/品詞っぽいヒューリスティックでTraffic/Conversionに分配"""
    traffic, conversion = [], []
    for w in terms:
        w0 = w
        # シード一致
        if w0 in TRAFFIC_SEEDS:
            traffic.append(w0)
            continue
        if w0 in CONVERSION_SEEDS:
            conversion.append(w0)
            continue
        # 語尾ヒューリスティック
        if re.search(r"(対応|充電|保護|互換|取付|固定|搭載|内蔵|防水|耐衝撃|軽量|薄型)$", w0):
            traffic.append(w0)
            continue
        if re.search(r"(人気|安心|快適|最適|便利|高品質|高評価|おすすめ)$", w0):
            conversion.append(w0)
            continue
        # 英数字が多い/単位風→Traffic寄せ
        if re.search(r"[0-9A-Za-z]{2,}", w0):
            traffic.append(w0)
            continue
        # デフォルトはTraffic寄せ（露出狙い優先）
        traffic.append(w0)
    # 重複除去
    traffic = list(dict.fromkeys(traffic))
    conversion = [w for w in dict.fromkeys(conversion) if w not in set(traffic)]
    return traffic, conversion

# -------------------------
# メイン処理
# -------------------------
def main():
    # 入力読込
    rakuten_products = read_products(INPUT_RAKUTEN)
    yahoo_products   = read_products(INPUT_YAHOO)
    products = rakuten_products + [p for p in yahoo_products if p not in set(rakuten_products)]

    if not RAKUTEN_API or not RAKUTEN_APP or not YAHOO_API or not YAHOO_APP:
        raise SystemExit("❌ .envのAPI設定が不足しています。RAKUTEN_* / YAHOO_* を確認してください。")

    print(f"🔎 収集対象商品数: {len(products)}")

    # 収集本体
    raw_docs = []   # [{"source","query","title","text","url"}...]
    for prod in tqdm(products, desc="🧲 クローリング/検索API", total=len(products)):
        queries = generate_queries(prod)
        for q in queries:
            # 楽天API 7..15ページ
            for page in RAKUTEN_PAGES:
                data = rakuten_search(q, page)
                if data and isinstance(data, dict):
                    items = (data.get("Items") or [])
                    for it in items:
                        ritem = it.get("Item", {})
                        title = normalize_text(ritem.get("itemName", ""))
                        cap   = normalize_text(ritem.get("itemCaption", ""))
                        url   = normalize_text(ritem.get("itemUrl", ""))
                        text  = " ".join([title, cap])
                        if text:
                            raw_docs.append({
                                "source": "rakuten",
                                "query": q,
                                "title": title,
                                "text": text,
                                "url": url
                            })
                        # フォールバックでHTML拡張（軽め）
                        if url and random.random() < 0.08:
                            html_txt = fetch_html_text(url)
                            if html_txt and len(html_txt) > 120:
                                raw_docs.append({
                                    "source": "rakuten_html",
                                    "query": q,
                                    "title": title,
                                    "text": normalize_text(html_txt)[:4000],
                                    "url": url
                                })
            # Yahoo API 7..15ページ
            for page in YAHOO_PAGES:
                data = yahoo_search(q, page)
                if data and isinstance(data, dict):
                    hits = data.get("hits") or data.get("items") or []
                    # v3は "hits" キー配列要素に title / description / url が入る
                    for h in hits:
                        title = normalize_text(h.get("name") or h.get("title") or "")
                        desc  = normalize_text(h.get("description") or h.get("caption") or "")
                        url   = normalize_text(h.get("url") or h.get("link") or "")
                        text  = " ".join([title, desc])
                        if text:
                            raw_docs.append({
                                "source": "yahoo",
                                "query": q,
                                "title": title,
                                "text": text,
                                "url": url
                            })
                        if url and random.random() < 0.08:
                            html_txt = fetch_html_text(url)
                            if html_txt and len(html_txt) > 120:
                                raw_docs.append({
                                    "source": "yahoo_html",
                                    "query": q,
                                    "title": title,
                                    "text": normalize_text(html_txt)[:4000],
                                    "url": url
                                })
        # スロットリング（API礼儀）
        time.sleep(0.15)

    # 文単位の素材（短文に割る）
    sentences = []
    for d in raw_docs:
        for seg in re.split(r"[。!?！？」\n]+", d["text"]):
            seg = normalize_text(seg)
            if 10 <= len(seg) <= 180:
                sentences.append(seg)

    # TF-IDFで上位語彙
    top_terms = tfidf_top_terms(sentences, top_k=400)
    traffic_terms, conversion_terms = classify_terms(top_terms)

    # まとめ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUT_DIR / f"raw_crawl_{ts}.json"
    out_path = OUT_DIR / f"traffic_conversion_{ts}.json"

    payload_raw = {
        "meta": {
            "timestamp": ts,
            "products": len(products),
            "docs": len(raw_docs),
            "sentences": len(sentences),
            "rakuten_api": bool(RAKUTEN_API),
            "yahoo_api": bool(YAHOO_API)
        },
        "documents": raw_docs[:5000],  # サイズ保護（必要なら増やす）
    }
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(payload_raw, f, ensure_ascii=False, indent=2)

    payload_tc = {
        "meta": payload_raw["meta"],
        "traffic_terms": traffic_terms[:200],
        "conversion_terms": conversion_terms[:120],
        "sample_sentences": sentences[:1000],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload_tc, f, ensure_ascii=False, indent=2)

    print(f"✅ 素材出力: {raw_path}")
    print(f"✅ 二層語彙: {out_path}")
    print("🎯 ライター側は traffic_conversion_*.json の terms + sample_sentences を使うと効果的です。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
