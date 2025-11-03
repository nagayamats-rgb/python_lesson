# -*- coding: utf-8 -*-
"""
alt_writer_v5_semantic_router.py
================================
楽天ALT専用（20本/商品） — Semantic Router Ready（“要/かんなめ”内蔵）

入力:
  /Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv
    - UTF-8、ヘッダに「商品名」必須

知見フォルダ:
  /Users/tsuyoshi/Desktop/python_lesson/output/semantics
    - lexical_clusters_*.json
    - market_vocab_*.json
    - structured_semantics_*.json
    - styled_persona_*.json
    - normalized_*.json
    - template_composer*.json
  ※存在しない/形式不揃いOK（自動吸収＆フォールバック）

出力:
  /Users/tsuyoshi/Desktop/python_lesson/output/ai_writer/alt_text_ai_raw_router_v5.csv
  /Users/tsuyoshi/Desktop/python_lesson/output/ai_writer/alt_text_refined_router_v5.csv
  /Users/tsuyoshi/Desktop/python_lesson/output/ai_writer/alt_text_diff_router_v5.csv

仕様ハイライト:
- AI生産: まず1〜2文の自然文で 100〜130字 / 行 を目標に20行生成（句点終止）
- ローカル整形: 80〜110字に自然カット、禁則適用、重複除去、箇条書き剥がし
- 空行/極短行: その場で補完（“自然な楽天ALTテンプレ最小構文”で自動埋め）
- “要 / かんなめ”: 生成のブレを抑える中核規範（口調・構文・禁則）を常時注入
- Semantic Router: 商品名⇄知見語彙の類似度で“その商品に効く語彙/構文”を抽出投入
- OpenAI: .env の OPENAI_MODEL / OPENAI_MODE / OPENAI_TEMPERATURE / OPENAI_MAX_TOKENS を尊重
  * 未設定時は model="gpt-5.1-mini" があれば使用、なければ "gpt-4o" へ自動フォールバック
  * response_format={"type":"text"} / max_completion_tokens=env or 1000 / 温度は env or 1.0

注意:
- ALTは楽天専用。Yahoo専用の販促語/装飾語は避ける。
- “画像・写真・当店・ランキング・クリック〜”等のメタ語は禁止（ローカル側で強制除去）
"""

import os
import re
import csv
import glob
import json
import time
import math
from collections import defaultdict, Counter
from pathlib import Path
from dotenv import load_dotenv

# 進捗バー
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# OpenAIクライアント
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# ─────────────────────────────────────────────────────────
# 0) 固定パス（要）
# ─────────────────────────────────────────────────────────
BASE = Path("/Users/tsuyoshi/Desktop/python_lesson")
INPUT_CSV = BASE / "sauce" / "rakuten.csv"
SEMANTICS_DIR = BASE / "output" / "semantics"
OUT_DIR = BASE / "output" / "ai_writer"

RAW_PATH  = OUT_DIR / "alt_text_ai_raw_router_v5.csv"
REF_PATH  = OUT_DIR / "alt_text_refined_router_v5.csv"
DIFF_PATH = OUT_DIR / "alt_text_diff_router_v5.csv"

# ─────────────────────────────────────────────────────────
# 1) 定数（禁則・正規化・目標長など）
# ─────────────────────────────────────────────────────────
# 画像描写/メタ/比較表現 禁止語（“要/かんなめ”規範）
FORBIDDEN_GLOBAL = [
    "画像", "写真", "見た目", "上の画像", "下の写真",
    "当店", "当社", "ショップ", "レビュー", "ランキング",
    "リンク", "こちら", "クリック", "カート", "購入はこちら",
    "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
    "返金保証", "送料無料（確約）",
]

# 箇条書き・列挙の頭を剥がす
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-・\*\u2022]\s*[\.．、]?\s*")
MULTI_COMMA_RE  = re.compile(r"、{3,}")
WS_RE           = re.compile(r"\s+")

# AI出力→ローカル整形の目標レンジ
RAW_MIN, RAW_MAX     = 100, 130
FINAL_MIN, FINAL_MAX =  80, 110

# “要 / かんなめ” — コア規範（プロンプト常駐）
KANNAME_COVENANT = (
    "【要/かんなめ】\n"
    "・ALTは“画像の説明”ではなく、商品の特徴・用途・対象・便益を自然文で要約すること。\n"
    "・楽天サイト内SEOを意識し、対応機種/型番・機能・素材・規格・使用シーン・ユーザー像を無理なく織り込む。\n"
    "・句読点の過剰や箇条書きは避け、1〜2文の流れる日本語にする。\n"
    "・禁止語（画像/写真/当店/ランキング/クリック 等）は使わない。競合比較も避ける。\n"
    "・宣伝調の過度な誇張は避け、具体を重視する。"
)

# ─────────────────────────────────────────────────────────
# 2) 環境 & OpenAI 初期化（.env 固定/フォールバック）
# ─────────────────────────────────────────────────────────
def init_env_and_client():
    load_dotenv(override=True)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
    # モデル/モード/温度/トークン: .env を尊重しつつ安全フォールバック
    model = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-5.1-mini"
    if model.lower() in {"gpt-5-nano", "gpt-5.1-nano"}:
        # nanoは出力が短文化しがち → mini系を推奨
        model = "gpt-5.1-mini"
    # 万一gpt-5系が未開通なら 4o に切替（実行時エラーを避ける）
    fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o").strip()
    mode = (os.getenv("OPENAI_MODE") or "chat").strip()
    temperature = float(os.getenv("OPENAI_TEMPERATURE") or "1.0")
    max_tokens  = int(os.getenv("OPENAI_MAX_TOKENS") or "1000")

    client = OpenAI(api_key=api_key)
    return client, model, fallback_model, mode, temperature, max_tokens

# ─────────────────────────────────────────────────────────
# 3) 入力（商品名）
# ─────────────────────────────────────────────────────────
def load_products(path: Path):
    if not path.exists():
        raise SystemExit(f"入力CSVが見つかりません: {path}")
    products = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("入力CSVに『商品名』ヘッダが見つかりません。")
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                products.append(nm)
    # 順序を保った重複除去
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            uniq.append(nm); seen.add(nm)
    return uniq

# ─────────────────────────────────────────────────────────
# 4) 知見の読込 & Semantic Router
# ─────────────────────────────────────────────────────────
def safe_load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_semantic_files():
    if not SEMANTICS_DIR.is_dir():
        return {}
    def latest(pattern):
        files = list(SEMANTICS_DIR.glob(pattern))
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

    return {
        "lexical": latest("lexical_clusters_*.json"),
        "market": latest("market_vocab_*.json"),
        "semantic": latest("structured_semantics_*.json"),
        "persona": latest("styled_persona_*.json"),
        "normalized": latest("normalized_*.json"),
        "template": latest("template_composer*.json"),
    }

def flatten_terms(data):
    out = []
    if isinstance(data, dict):
        for v in data.values():
            out.extend(flatten_terms(v))
    elif isinstance(data, list):
        for v in data:
            out.extend(flatten_terms(v))
    elif isinstance(data, str):
        out.append(data)
    return out

def tokenize(s: str):
    # 簡易トークン化：全角→半角の一部、非文字除去、空白split
    s2 = re.sub(r"[^\w\dぁ-んァ-ン一-龥\-＋+/\.％%㎜mmcmCMxX ]+", " ", s)
    s2 = WS_RE.sub(" ", s2).strip()
    return [t for t in s2.split(" ") if t]

def jaccard(a: set, b: set):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def semantic_router(product: str, all_buckets: dict, top_k=24):
    """
    商品名のトークン集合と知見語彙の集合の Jaccard 類似で粗くスコア → 上位抽出
    ※精緻でなくてOK。安定・高速・再現性重視。
    """
    p_tokens = set(tokenize(product))
    scored = []
    # それぞれのバケツから語彙を収集
    for bucket_name, terms in all_buckets.items():
        for t in terms:
            t_tokens = set(tokenize(t))
            score = jaccard(p_tokens, t_tokens)
            if score > 0:
                scored.append((t, score))
    # 上位抽出（重複語はスコア最大のみ）
    score_map = {}
    for t, sc in scored:
        if t not in score_map or sc > score_map[t]:
            score_map[t] = sc
    top_terms = [t for t, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return top_terms

def load_knowledge_for_router():
    files = list_semantic_files()
    buckets = defaultdict(list)
    forb_local = set()

    # lexical clusters
    for p in files.get("lexical", []):
        data = safe_load_json(p)
        if not data: continue
        if isinstance(data, dict) and "clusters" in data:
            for c in (data.get("clusters") or []):
                terms = c.get("terms") or []
                buckets["lexical"].extend([t for t in terms if isinstance(t, str)])
        else:
            buckets["lexical"].extend([t for t in flatten_terms(data)])

    # market vocab
    for p in files.get("market", []):
        data = safe_load_json(p)
        if not data: continue
        if isinstance(data, list):
            for v in data:
                if isinstance(v, dict) and isinstance(v.get("vocabulary"), str):
                    buckets["market"].append(v["vocabulary"])
                elif isinstance(v, str):
                    buckets["market"].append(v)
        elif isinstance(data, dict):
            arr = data.get("vocabulary") or data.get("vocab") or []
            if isinstance(arr, list):
                buckets["market"].extend([x for x in arr if isinstance(x, str)])

    # structured semantics
    for p in files.get("semantic", []):
        data = safe_load_json(p)
        if not data: continue
        # 想定: {"concepts":[...], "scenes":[...], "targets":[...], "use_cases":[...], "features":[...], "benefits":[...]}
        if isinstance(data, dict):
            for k in ("concepts","scenes","targets","use_cases","features","benefits","semantics"):
                arr = data.get(k) or []
                buckets["semantic"].extend([x for x in arr if isinstance(x, str)])
        else:
            buckets["semantic"].extend([t for t in flatten_terms(data)])

    # persona（口調/レジスター）
    for p in files.get("persona", []):
        data = safe_load_json(p)
        if not data: continue
        if isinstance(data, dict):
            tone = data.get("tone") or {}
            if isinstance(tone, dict):
                for v in tone.values():
                    if isinstance(v, str): buckets["persona"].append(v)
        buckets["persona"].extend([t for t in flatten_terms(data)])

    # normalized（禁則）
    for p in files.get("normalized", []):
        data = safe_load_json(p)
        if not data: continue
        if isinstance(data, dict):
            fw = data.get("forbidden_words") or []
            for w in fw:
                if isinstance(w, str): forb_local.add(w)

    # template composer（骨子/型ヒント）
    for p in files.get("template", []):
        data = safe_load_json(p)
        if not data: continue
        if isinstance(data, dict):
            hints = data.get("hints") or data.get("templates") or []
            buckets["template"].extend([h for h in hints if isinstance(h, str)])
        else:
            buckets["template"].extend([t for t in flatten_terms(data)])

    return buckets, sorted(set(list(FORBIDDEN_GLOBAL) + list(forb_local)))

# ─────────────────────────────────────────────────────────
# 5) プロンプト
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "あなたはECサイト（楽天）のALTテキスト専門コピーライターです。\n"
    + KANNAME_COVENANT +
    "\n以下のルールを厳守して20行の自然文を生成してください:\n"
    f"・各行は1〜2文、全角およそ{RAW_MIN}〜{RAW_MAX}文字。\n"
    "・必ず句点「。」で終える。箇条書きや番号（1. ・ - など）やラベルは付けない。\n"
    "・出力はテキストのみ（JSON/記号/枠なし）。\n"
)

def build_user_prompt(product: str, top_terms: list, forbidden_words: list):
    # ルータが渡す“この商品に効く語彙/骨子”
    router_hint = "、".join(top_terms[:30]) if top_terms else ""
    forbid_txt  = "、".join(sorted(set(forbidden_words)))
    # 構成ヒント（テンプレではなく自然に）
    structure = "商品スペック→コアコンピタンス→どんな人→利用シーン→便益（自然な日本語、詰め込みすぎない）"
    return (
        f"商品名: {product}\n"
        f"知見ヒント: {router_hint}\n"
        f"構成ヒント: {structure}\n"
        f"禁止語: {forbid_txt}\n"
        "出力: 20行の自然文（各行1〜2文）。句点で終える。"
    )

# ─────────────────────────────────────────────────────────
# 6) OpenAI 呼び出し（堅牢・バックオフ付き）
# ─────────────────────────────────────────────────────────
def call_openai_lines(client, model, fallback_model, mode, temperature, max_tokens, system_prompt, user_prompt, retry=4, wait=6):
    last_err = None
    use_model = model
    for attempt in range(retry):
        try:
            res = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "text"},
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Empty content from OpenAI.")
            return content
        except Exception as e:
            last_err = e
            # 429 / 一部の不明エラーは待機→リトライ
            err_msg = str(e)
            if "insufficient_quota" in err_msg or "429" in err_msg:
                time.sleep(wait * (attempt + 1))
            elif "model_not_found" in err_msg or "does not exist" in err_msg:
                # フォールバック
                use_model = fallback_model
                time.sleep(2)
            else:
                time.sleep(wait)
    raise RuntimeError(f"OpenAI応答取得に失敗: {last_err}")

# ─────────────────────────────────────────────────────────
# 7) ローカル整形（自然カット/禁則/重複/補完）
# ─────────────────────────────────────────────────────────
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    t = (text or "").strip()
    if not t: return ""
    # 列挙頭除去
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")
    # 余分スペース圧縮・読点連続調整
    t = WS_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    # 句点終止
    if not t.endswith("。"):
        t += "。"
    # 長すぎる → 120まで許容、近い句点まで前詰め
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        t = cut if p == -1 else cut[:p+1]
    # 禁止語の完全除去（痕跡が出ないよう文字列置換）
    for ng in FORBIDDEN_GLOBAL:
        if ng and ng in t:
            t = t.replace(ng, "")
    return t.strip()

def refine_lines(raw_lines):
    # 句点終止 & 自然カット
    norm = []
    for ln in raw_lines:
        if not ln: continue
        ln = soft_clip_sentence(ln)
        if len(ln) < 15:
            continue
        norm.append(ln)

    # 重複除去（完全一致）
    uniq = []
    seen = set()
    for ln in norm:
        if ln not in seen:
            uniq.append(ln); seen.add(ln)

    # 20本に調整
    def fill_line(seed: str) -> str:
        # 語尾ゆる変（体言止め混在を許す）
        s = seed
        s = s.replace("します。", "です。")
        s = s.replace("できます。", "しやすいです。")
        if not s.endswith("。"):
            s += "。"
        return soft_clip_sentence(s)

    i = 0
    while len(uniq) < 20 and uniq:
        uniq.append(fill_line(uniq[i % len(uniq)]))
        i += 1

    return uniq[:20]

def sanitize_model_bullets(text: str):
    """
    OpenAIが列挙してきた場合に備え、行ごとに整形しやすい形へ。
    """
    lines = [LEADING_ENUM_RE.sub("", ln).strip("・-—●　") for ln in text.split("\n") if ln.strip()]
    # 先頭60本まで保持（過剰生成対策）
    return lines[:60]

def minimal_fallback(product: str):
    # 空行補完テンプレ（最小構文）
    return f"{product} の使い勝手を高め、日常の不便を解消する設計です。"

# ─────────────────────────────────────────────────────────
# 8) 書き出し
# ─────────────────────────────────────────────────────────
def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_raw(products, all_raw):
    with RAW_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_raw):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_refined(products, all_refined):
    with REF_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_refined):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_diff(products, all_raw, all_refined):
    with DIFF_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_refined_{i+1}" for i in range(20)]
        w.writerow(header)
        for p, r, ref in zip(products, all_raw, all_refined):
            r_line   = (r[:20]   + [""] * max(0, 20 - len(r)))
            ref_line = (ref[:20] + [""] * max(0, 20 - len(ref)))
            w.writerow([p] + r_line + ref_line)

# ─────────────────────────────────────────────────────────
# 9) メイン
# ─────────────────────────────────────────────────────────
def main():
    print("🌸 ALTライター v5.0（Semantic Router Ready + “要/かんなめ”）")
    client, model, fallback_model, mode, temperature, max_tokens = init_env_and_client()
    ensure_outdir()

    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    # 知見の吸収（柔軟）→ Semantic Router で商品ごとのトップ語彙抽出
    buckets, forbidden_all = load_knowledge_for_router()

    all_raw, all_refined = [], []

    for product in tqdm(products, desc="🧠 生成中", total=len(products)):
        try:
            top_terms = semantic_router(product, buckets, top_k=28)
            user_prompt = build_user_prompt(product, top_terms, forbidden_all)
            content = call_openai_lines(
                client, model, fallback_model, mode, temperature, max_tokens,
                SYSTEM_PROMPT, user_prompt, retry=4, wait=6
            )
            raw_lines = sanitize_model_bullets(content)

            # 空/短文を最小構文で補填（20行確保のための安全弁）
            if len(raw_lines) < 20:
                raw_lines += [minimal_fallback(product) for _ in range(20 - len(raw_lines))]

        except Exception:
            # OpenAI全滅時も欠番なしに進める
            raw_lines = [minimal_fallback(product)] * 20

        refined = refine_lines(raw_lines)

        # 行頭/行末のゴミ除去・体言止め混在許容
        refined = [ln.strip(" ・-—●") for ln in refined]

        all_raw.append(raw_lines[:20])
        all_refined.append(refined[:20])

        # 軽いスロットリング（429緩和）
        time.sleep(0.2)

    # 書き出し
    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    # ざっくり統計
    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens)))

    print("✅ 出力完了:")
    print(f"   - AI生出力 : {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print("🔒 仕様メモ:")
    print("   - “要/かんなめ”常駐、禁則強化、自然文1〜2文、句点終止、楽天ALT特化")
    print(f"   - AI目標 {RAW_MIN}〜{RAW_MAX}字 → ローカル整形 {FINAL_MIN}〜{FINAL_MAX}字")

if __name__ == "__main__":
    main()
import atlas_autosave_core
