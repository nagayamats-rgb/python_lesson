# -*- coding: utf-8 -*-
"""
semantic_extractor_rebuilder_v1_1_unified.py
------------------------------------------------
目的:
  1) /sauce/rakuten.csv（列: 商品名）と /sauce/yahoo.csv（列: name）を一括読込
  2) 商品名→関連クエリ生成（OpenAI任意。無ければローカル規則で安定動作）
  3) 楽天商品検索API(20220601) を page=7..15, hits=30 でクロール
     → itemName / catchcopy / itemCaption のテキスト収集
  4) 正規化→語彙/構造抽出→クラスタリング→知見JSON群を出力
  5) ライターが使いやすい最終“融合知見” JSON もあわせて出力

入出力:
  - 入力CSV:
      /Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv   （列名: 商品名）
      /Users/tsuyoshi/Desktop/python_lesson/sauce/yahoo.csv     （列名: name）
  - .env:
      RAKUTEN_APP_ID=xxxxxxxx
      OPENAI_API_KEY=xxxxxx（任意）
      OPENAI_MODEL=gpt-4o など（任意。無ければ gpt-4o を既定）
  - 出力先:
      /Users/tsuyoshi/Desktop/python_lesson/output/semantics
    出力ファイル例:
      lexical_clusters_YYYYmmdd_HHMMSS.json
      market_vocab_YYYYmmdd_HHMMSS.json
      structured_semantics_YYYYmmdd_HHMMSS.json
      styled_persona_YYYYmmdd_HHMMSS.json
      template_composer.json
      normalized_YYYYmmdd_HHMMSS.json
      knowledge_fused_structured.json     ← ライター直結の最終融合知見
      knowledge_fused_text.txt            ← 文章プロンプト向けの知見文

依存:
  自動インストール: python-dotenv, requests, fugashi, unidic-lite, numpy, scikit-learn
  OpenAIは任意（存在時のみクエリ生成で補助）
"""

# ========= 0) 依存の自動インストール =========
import sys, subprocess, importlib

def _ensure(pkg, pip_name=None, version=None):
    try:
        importlib.import_module(pkg)
    except Exception:
        name = pip_name or pkg
        if version:
            name = f"{name}=={version}"
        print(f"📦 Installing {name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])

_ensure("dotenv", "python-dotenv")
_ensure("requests")
_ensure("fugashi")
_ensure("unidic_lite")
_ensure("numpy")
_ensure("sklearn", "scikit-learn")

# OpenAIは任意
_HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    _HAS_OPENAI = False

# ========= 1) インポート & 定数 =========
import os, re, csv, json, time, html, random, unicodedata
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

BASE_DIR  = "/Users/tsuyoshi/Desktop/python_lesson"
IN_RAKU   = f"{BASE_DIR}/sauce/rakuten.csv"
IN_YAHOO  = f"{BASE_DIR}/sauce/yahoo.csv"
OUT_DIR   = f"{BASE_DIR}/output/semantics"

RAKUTEN_API = "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601"

FORBIDDEN_DEFAULT = [
    "画像","写真","見た目","図","上の画像","下の写真",
    "当店","当社","レビュー","ランキング","クリック","こちら",
    "競合","優位性","業界最高","最安","No.1","ナンバーワン","売上No1",
    "リンク","ページ","購入はこちら","送料無料（確約）","返金保証",
]

# “構造”のヒント軸（緩い辞書）
FEATURE_HINTS = ["耐衝撃","防水","防塵","軽量","薄型","高耐久","磁力","急速充電","高速","低遅延","省電力","低発熱","静音","安定","強化ガラス","シリコン","TPU","アルミ","ステンレス","抗菌","防汚"]
SCENE_HINTS   = ["通勤","通学","旅行","出張","キャンプ","ビジネス","自宅","オフィス","車内","キッチン","アウトドア","スポーツ","勉強","会議","ジム"]
TARGET_HINTS  = ["学生","社会人","子ども","高齢者","ゲーマー","クリエイター","ビジネスパーソン","主婦","家族"]
BENEFIT_HINTS = ["快適","便利","安心","効率化","省力化","時短","整理整頓","保護","美観","携帯性","耐久性","安定性","安全性"]

DEVICE_REGEX = re.compile(
    r"(iPhone\s?(?:[0-9]{1,2}|SE|SE2|SE\s?2|SE\s?第\d世代)|"
    r"iPad(?:\s?(?:Pro|Air|mini|第\d世代))?|"
    r"Apple\s?Watch(?:\s?Series\s?\d+|SE2?|Ultra)?|"
    r"Galaxy\s?[A-Z]?\d+\w*|Xperia\s?\w+|Pixel\s?\d+\w*|"
    r"任天堂Switch|Switch|PS5|PS4|AirPods(?:\s?Pro|Max)?)",
    re.IGNORECASE
)

SPECS_REGEXES = [
    re.compile(r"\b(?:USB[- ]?C|Type[- ]?C|Lightning|Micro[- ]?USB)\b", re.IGNORECASE),
    re.compile(r"\b(?:Bluetooth\s?(?:4\.2|5(?:\.0|\.1|\.2|\.3)?)|Wi[- ]?Fi(?:6|6E|7)?)\b", re.IGNORECASE),
    re.compile(r"\b\d{3,5}\s?mAh\b", re.IGNORECASE),
    re.compile(r"\b(?:15W|20W|30W|45W|65W|100W|140W)\b", re.IGNORECASE),
    re.compile(r"\b(?:mm|cm|inch|インチ)\b"),
]

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", "。", s, flags=re.IGNORECASE)
    s = re.sub(r"<.*?>", " ", s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def read_csv_column(path, col):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if col not in (r.fieldnames or []):
            return out
        for row in r:
            v = (row.get(col) or "").strip()
            if v:
                out.append(v)
    return out

def uniq_keep(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ========= 2) OpenAI 任意 =========
def init_openai_client():
    if not _HAS_OPENAI:
        return None, None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, None
    model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
    client = OpenAI(api_key=key)
    return client, model

def build_queries_local(name: str, k=8):
    base = re.sub(r"[\"'、,。()\[\]【】/\\]+", " ", name)
    base = re.sub(r"\s+", " ", base).strip()
    mods = [" 透明"," 耐衝撃"," 薄型"," 軽量"," 急速充電"," マグネット"," 互換"," 純正風"," 高品質"," おしゃれ"," ビジネス"," 学生"]
    seeds = [base] + [base+m for m in mods]
    random.shuffle(seeds)
    return uniq_keep(seeds)[:k]

def build_queries_openai(name: str, client, model: str):
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"あなたは日本のEC検索に強いSEOプランナーです。"},
                {"role":"user","content": f"商品名: {name}\nこの商品への流入に効く関連検索クエリを10個、日本語で。出力は改行区切りのみ。"}
            ],
            response_format={"type":"text"},
            max_completion_tokens=300,
            temperature=0.4
        )
        text = (res.choices[0].message.content or "").strip()
        qs = [q.strip("・-—● ") for q in text.split("\n") if q.strip()]
        return qs[:10] if qs else build_queries_local(name)
    except Exception:
        return build_queries_local(name)

# ========= 3) 楽天API収集 =========
def fetch_rakuten_texts(app_id: str, query: str, start_page=7, end_page=15, hits=30, sleep=0.35):
    out = []
    for p in range(start_page, end_page+1):
        params = {
            "applicationId": app_id,
            "format": "json",
            "keyword": query,
            "page": p,
            "hits": hits
        }
        try:
            r = requests.get(RAKUTEN_API, params=params, timeout=20)
            if r.status_code != 200:
                time.sleep(sleep); continue
            data = r.json()
            items = data.get("Items") or []
            for it in items:
                d = it.get("Item") or {}
                for k in ("itemName","catchcopy","itemCaption"):
                    t = normalize_text(d.get(k) or "")
                    if t:
                        out.append(t)
            time.sleep(sleep)
        except Exception:
            time.sleep(sleep)
            continue
    return out

# ========= 4) 形態素・抽出 =========
from fugashi import Tagger
_TAGGER = Tagger()

def tokenize(text: str):
    return [w.surface for w in _TAGGER(text)]

def split_sentences_jp(s: str):
    s = s.replace("！","。").replace("？","。")
    parts = [p.strip() for p in s.split("。") if p.strip()]
    return parts

def pick_market_vocab(texts):
    vocab = set()
    for t in texts:
        for m in DEVICE_REGEX.findall(t):
            vocab.add(unicodedata.normalize("NFKC", m))
        for rx in SPECS_REGEXES:
            for m in rx.findall(t):
                vocab.add(unicodedata.normalize("NFKC", m))
        for m in re.findall(r"\b[A-Z0-9\-]{3,}\b", t, flags=re.IGNORECASE):
            if not m.isdigit():
                vocab.add(unicodedata.normalize("NFKC", m))
    return sorted(vocab)

def extract_semantics(texts):
    features, scenes, targets, benefits = Counter(), Counter(), Counter(), Counter()
    for t in texts:
        for sent in split_sentences_jp(t):
            s = unicodedata.normalize("NFKC", sent)
            for kw in FEATURE_HINTS:
                if kw in s: features[kw]+=1
            for kw in SCENE_HINTS:
                if kw in s: scenes[kw]+=1
            for kw in TARGET_HINTS:
                if kw in s: targets[kw]+=1
            for kw in BENEFIT_HINTS:
                if kw in s: benefits[kw]+=1
    def top(c, n): return [k for k,_ in c.most_common(n)]
    return {
        "features": top(features, 40),
        "scenes":   top(scenes,   30),
        "targets":  top(targets,  30),
        "benefits": top(benefits, 40),
    }

def build_lexical_clusters(texts, n_clusters=8, top_terms=15):
    docs = [t for t in texts if t]
    if not docs:
        return {"clusters":[]}
    try:
        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=2, max_df=0.8)
        X = vec.fit_transform(docs)
        n = min(n_clusters, max(2, X.shape[0]//10))
        km = KMeans(n_clusters=n, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        terms = np.array(vec.get_feature_names_out())
        centers = km.cluster_centers_.argsort()[:, ::-1]
        clusters=[]
        for i in range(n):
            idx = centers[i,:top_terms]
            words = [w for w in terms[idx] if len(w)>=2]
            clusters.append({"terms": words})
        return {"clusters": clusters}
    except Exception:
        # フォールバック：頻出語で等分割
        toks=[]
        for t in docs: toks.extend(tokenize(t))
        freq = [w for w,_ in Counter(toks).most_common(n_clusters*top_terms)]
        chunks = [freq[i:i+top_terms] for i in range(0,len(freq),top_terms)]
        return {"clusters":[{"terms":[w for w in ch if len(w)>=2]} for ch in chunks[:n_clusters]]}

# ========= 5) Persona / Template / Normalized =========
def default_persona():
    return {
        "tone": {
            "style": "自然体で知的、過剰な装飾を避ける",
            "register": "丁寧体で簡潔、2文以内を基本",
            "rhythm": "読みやすさ重視、接続詞の乱用は避ける"
        }
    }

def default_templates():
    return {
        "hints":[
            "スペック→強み→対象→シーン→便益",
            "素材/仕様→保護/快適→対象→使用文脈→解決",
            "互換/対応→操作性→安全/安心→導入効果"
        ]
    }

def normalized_forbidden():
    return {"forbidden_words": FORBIDDEN_DEFAULT}

# ========= 6) 融合知見（ライター向け最終JSON/Text） =========
def fuse_for_writer(lexical, market_vocab, semantics, persona, templates, normalized):
    # 軽く上限を設けて過学習回避
    clusters  = (lexical or {}).get("clusters", [])
    top_terms = []
    for c in clusters:
        ts = c.get("terms") or []
        top_terms.extend([t for t in ts if isinstance(t,str)])
    top_terms = list(dict.fromkeys(top_terms))[:80]

    market = (market_vocab or [])[:80]
    sem    = {
        "features": (semantics.get("features") or [])[:25],
        "scenes":   (semantics.get("scenes")   or [])[:20],
        "targets":  (semantics.get("targets")  or [])[:20],
        "benefits": (semantics.get("benefits") or [])[:25],
    }
    fused = {
        "lexical_terms": top_terms,
        "market_vocab":  market,
        "semantics":     sem,
        "persona":       persona.get("tone", {}),
        "templates":     templates.get("hints", []),
        "forbidden":     (normalized or {}).get("forbidden_words", []),
    }
    return fused

def build_prompt_sentences(fused, aim_min=9, aim_max=14):
    # ライターの system/user にそのまま渡せる“自然文”の知見文セット
    def pick(xs, n): return [x for x in xs[:n] if isinstance(x,str)]
    feats = pick(fused["semantics"].get("features",[]), 6)
    scenes= pick(fused["semantics"].get("scenes",[]),   6)
    targs = pick(fused["semantics"].get("targets",[]),  6)
    bens  = pick(fused["semantics"].get("benefits",[]), 6)
    vocab = pick(fused.get("market_vocab",[]),          10)

    sents=[]
    # なるべく“自然文”で短めに。句点で終える。
    if feats: sents.append(f"頻出する機能・仕様は「{ '、'.join(feats[:4]) }」など。")
    if vocab: sents.append(f"関連する型番・規格・語彙には「{ '、'.join(vocab[:6]) }」が見られます。")
    if scenes: sents.append(f"使用シーンは「{ '、'.join(scenes[:4]) }」が多く想定されます。")
    if targs: sents.append(f"対象ユーザーは「{ '、'.join(targs[:4]) }」が中心です。")
    if bens:  sents.append(f"訴求される便益は「{ '、'.join(bens[:4]) }」が目立ちます。")
    sents.append("画像描写語や店舗メタ語は使用しません。")
    sents.append("文は自然な日本語で、過度な羅列やテンプレ表現を避けます。")
    sents.append("対応機種・規格・シーン・ユーザーを“自然に”織り込みます。")
    sents.append("体言止めは必要に応じて許可、句読点は日本語のリズムを優先します。")

    # 範囲に収める（不足なら補足、超過なら切り詰め）
    if len(sents) < aim_min:
        sents.append("読みやすさを最優先し、2文以内を基本とします。")
    return sents[:aim_max]

# ========= 7) 書き出し =========
def write_json(obj, fname):
    ensure_outdir()
    p = os.path.join(OUT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ {fname}")
    return p

def write_text(lines, fname):
    ensure_outdir()
    p = os.path.join(OUT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"✅ {fname}")
    return p

# ========= 8) メイン =========
def main():
    print("🧩 semantic_extractor_rebuilder_v1.1 起動")

    load_dotenv(override=True)
    app_id = os.getenv("RAKUTEN_APP_ID", "").strip()
    if not app_id:
        raise SystemExit("❌ .env に RAKUTEN_APP_ID が見つかりません。")

    client, model = init_openai_client()

    # 1) 入力
    rakuten_names = read_csv_column(IN_RAKU,  "商品名")
    yahoo_names   = read_csv_column(IN_YAHOO, "name")
    products = uniq_keep(rakuten_names + yahoo_names)
    print(f"📦 対象商品数: {len(products)}")

    # 2) 収集
    corpus = []
    for nm in products:
        qs = build_queries_openai(nm, client, model) if client else build_queries_local(nm)
        qs = uniq_keep(qs)
        for q in qs:
            corpus.extend(fetch_rakuten_texts(app_id, q, start_page=7, end_page=15, hits=30, sleep=0.35))
        time.sleep(0.2)

    corpus = [normalize_text(t) for t in corpus if t]
    corpus = uniq_keep(corpus)
    print(f"🧾 収集テキスト数: {len(corpus)}")

    # 3) 抽出
    market_vocab = pick_market_vocab(corpus)
    semantics    = extract_semantics(corpus)
    lexical      = build_lexical_clusters(corpus, n_clusters=8, top_terms=18)

    # 4) 付帯（persona / template / normalized）
    persona   = default_persona()
    template  = default_templates()
    normalized= normalized_forbidden()

    # 既存 normalized があればマージ
    exist_norm = os.path.join(OUT_DIR, "normalized_20251031_0039.json")
    if os.path.exists(exist_norm):
        try:
            with open(exist_norm, "r", encoding="utf-8") as f:
                ex = json.load(f)
            fw = list({*(ex.get("forbidden_words") or []), *normalized["forbidden_words"]})
            normalized["forbidden_words"] = fw
        except Exception:
            pass

    # 5) 書き出し（時刻タグ）
    tag = now_tag()
    write_json(lexical,                  f"lexical_clusters_{tag}.json")
    write_json(market_vocab,             f"market_vocab_{tag}.json")
    write_json(semantics,                f"structured_semantics_{tag}.json")
    write_json(persona,                  f"styled_persona_{tag}.json")
    write_json(template,                 f"template_composer.json")   # 上書きOK
    write_json(normalized,               f"normalized_{tag}.json")

    # 6) 最終“融合知見”（ライターに直結）
    fused = fuse_for_writer(lexical, market_vocab, semantics, persona, template, normalized)
    write_json(fused, "knowledge_fused_structured.json")
    prompt_lines = build_prompt_sentences(fused, aim_min=9, aim_max=14)
    write_text(prompt_lines, "knowledge_fused_text.txt")

    print("🎯 完了: /output/semantics に知見JSONを再構築しました。")
    print("   → ライター側からは knowledge_fused_structured.json / knowledge_fused_text.txt を読み込むのが最短です。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
