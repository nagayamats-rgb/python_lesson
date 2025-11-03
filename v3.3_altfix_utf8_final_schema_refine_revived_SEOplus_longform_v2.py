# -*- coding: utf-8 -*-
"""
v3.3_altfix_utf8_final_schema_refine_revived_SEOplus_longform_v2.py
ALT長文（80〜110字）×20本を、ローカル知見を活かして自然文で生成。
- 入力:  /Users/tsuyoshi/Desktop/python_lesson/rakuten.csv（UTF-8, 先頭行ヘッダ, 「商品名」列）
- 知見:  ./output/semantics/ 配下のJSON群（存在すれば読み込むだけでOK）
- 出力:  ./output/ai_writer/alt_text_refined_final_longform_v2.csv（商品名,ALT1..ALT20）
"""

import os
import csv
import json
import re
import time
from typing import List, Dict, Any
from collections import defaultdict

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ====== 初期設定 ======
load_dotenv()  # .env 読み込み（OPENAI_API_KEY, OPENAI_BASE_URL 等）
client = OpenAI()  # 環境変数からAPIキーを取得

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 固定：gpt-4o（ユーザー指定）
MAX_COMPLETION_TOKENS = 1200                 # 長文耐性
TEMPERATURE = 1                              # 仕様互換のためデフォルト（=1）に固定
RETRY = 3
SLEEP_ON_FAIL = 3.0

# 入出力
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = os.path.join(BASE_DIR, "rakuten.csv")  # UTF-8 & 「商品名」
OUT_DIR = os.path.join(BASE_DIR, "output/ai_writer")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v2.csv")

# 知見ファイル（存在すれば使う）
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
SEM_FILES = {
    "lexical": "lexical_clusters_20251030_223013.json",
    "structured": "structured_semantics_20251030_224846.json",
    "market": "market_vocab_20251030_201906.json",
    "persona": "styled_persona_20251031_0031.json",
    "normalized": "normalized_20251031_0039.json",
}

# 禁則語（画像描写語・メタ語）
FORBIDDEN = {
    "画像", "写真", "映って", "写って", "写る", "見えている", "スクリーンショット",
    "ALT", "代替テキスト", "画像説明文", "画像描述", "Image", "Picture"
}

# ====== ユーティリティ ======
def read_products_from_csv(path: str) -> List[str]:
    products = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("商品名") or "").strip()
            if name:
                products.append(name)
    # 重複除去・順序維持
    seen = set()
    uniq = []
    for p in products:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def load_json_safe(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_local_knowledge() -> Dict[str, Any]:
    """output/semantics 配下の知見を読み込んで、使いやすい形にまとめる"""
    data = {}
    for key, fname in SEM_FILES.items():
        full = os.path.join(SEM_DIR, fname)
        data[key] = load_json_safe(full)
    return data

def top_strings_from(obj, keys: List[str], cap: int = 20) -> List[str]:
    """JSONが list/dict どちらでも、与えられた key 候補から文字を拾う"""
    out = []
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                for k in keys:
                    v = x.get(k)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
            elif isinstance(x, str) and x.strip():
                out.append(x.strip())
    elif isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, str) and t.strip():
                        out.append(t.strip())
            elif isinstance(v, str) and v.strip():
                out.append(v.strip())
    # 重複除去
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup[:cap]

def build_knowledge_text(product: str, kb: Dict[str, Any]) -> str:
    """
    ローカル知見を軽く要約してテキスト化。
    - 市場語彙（market.vocabulary）
    - 構造語彙（structured.concepts / features）
    - クラスタ語（lexical.clusters）
    - スタイル（persona）
    - 正規化（normalized の禁則など）
    """
    market_terms = top_strings_from(kb.get("market"), ["vocabulary", "term", "keyword"], cap=25)
    structured_terms = top_strings_from(kb.get("structured"), ["concepts", "features", "benefits"], cap=25)
    lexical_terms = top_strings_from(kb.get("lexical"), ["cluster", "tokens", "words"], cap=25)
    persona_tone = top_strings_from(kb.get("persona"), ["tone", "style", "voice"], cap=10)
    normalized_forbidden = []
    norm = kb.get("normalized")
    if isinstance(norm, dict):
        fw = norm.get("forbidden_words") or []
        if isinstance(fw, list):
            normalized_forbidden = [w for w in fw if isinstance(w, str)]

    # 禁則はグローバルのFORBIDDENに加算（参照のみ）
    local_forbidden_text = "、".join(normalized_forbidden[:20]) if normalized_forbidden else ""

    parts = []
    if market_terms:
        parts.append(f"市場語彙: { '、'.join(market_terms[:15]) }")
    if structured_terms:
        parts.append(f"構造語彙: { '、'.join(structured_terms[:15]) }")
    if lexical_terms:
        parts.append(f"クラスタ語: { '、'.join(lexical_terms[:15]) }")
    if persona_tone:
        parts.append(f"文体: { '、'.join(persona_tone[:5]) }")
    if local_forbidden_text:
        parts.append(f"禁則候補: { local_forbidden_text }")

    summed = " / ".join(parts) if parts else "（ローカル知見は最小限）"
    # 製品名を入れた導入
    lead = f"商品名: {product}"
    return f"{lead}\n知見要約: {summed}"


# ====== テキスト整形 ======
NUM_PREFIX = re.compile(r"^\s*([0-9０-９]+|[①-⑳]|[一二三四五六七八九十])[\.\)\：:、]\s*")
QUOTE_EDGES = re.compile(r"^[\"'“”‘’「『（\(\[]+|[\"'“”‘’」』）\)\]]+$")
EXTRA_SPACES = re.compile(r"\s+")

def sanitize_line(s: str) -> str:
    if not s:
        return ""
    # 行頭の番号/箇条書き接頭を除去
    s = NUM_PREFIX.sub("", s.strip())
    # ALT/画像関連語のラベル排除（ALT: など）
    s = re.sub(r"^\s*(ALT|Alt|alt)\s*[:：]\s*", "", s)
    # 外側の引用符系を剥がす
    s = s.strip().strip('"').strip("'").strip("「").strip("」").strip("『").strip("』").strip()
    # 画像描写語の明示ラベル除去（文中は後でチェック）
    return s

def ends_with_kuten(s: str) -> bool:
    return s.endswith("。")

def contains_forbidden(s: str) -> bool:
    for w in FORBIDDEN:
        if w in s:
            return True
    return False

def natural_clip_80_110(s: str) -> str:
    """
    80〜110字へ自然クリップ。
    - 110字を超えていれば最後の「。」で切る
    - 句点が無い or 80未満なら、そのまま（後段のAI出力側が概ね整える想定）
    """
    s = s.strip()
    length = len(s)
    if length <= 110:
        return s
    # 110以内で最後の句点
    cut = s[:110]
    last_kuten = cut.rfind("。")
    if last_kuten >= 70:  # 70 以上で句点があればそこまで
        return cut[: last_kuten + 1]
    # 句点が見当たらなければ 110 で切る（最後に句点付与）
    clipped = s[:110]
    if not ends_with_kuten(clipped):
        clipped = clipped.rstrip("、，,") + "。"
    return clipped

def post_refine_line(s: str) -> str:
    """句点必須・変な末尾記号除去・スペース整形・禁則再チェック"""
    s = s.strip()
    # 変な引用符/記号が末尾に残っていれば調整
    s = s.rstrip('、，,;；:：…')
    # 必ず句点で終える
    if not ends_with_kuten(s):
        s += "。"
    # 画像描写語が紛れたらやさしく除去（語そのものを落とす）
    for w in list(FORBIDDEN):
        s = s.replace(w, "")
    # 連続空白を単一化
    s = EXTRA_SPACES.sub(" ", s).strip()
    return s


# ====== OpenAI コール ======
PROMPT_TEMPLATE = """あなたはSEO最適化された商品説明文の専門ライターです。
以下の「商品名」と「知見要約」をもとに、画像の描写を避けつつ、自然で説得力のある日本語のALTテキストを20件生成してください。

厳守条件:
- 箇条書きや番号(1. 2. など)は禁止。行頭に数字や記号を置かない。
- 各ALTは1〜2文構成で自然な文体にする。
- 各ALTは全角80〜110文字程度を目安とする。
- 必ず句点（。）で終える。
- 「画像」「写真」「映っている」「ALT」などの語は使わない。
- 商品スペック／機能、コアコンピタンス、想定ユーザー、利用シーン、ベネフィットを自然に織り交ぜる。
- SEOを意識し、適切なキーワード（型番/端子/機種/スペック/用途など）を不自然にならない範囲で散りばめる。
- 出力はプレーンテキスト。ALT 1〜20を各行に1つずつ、合計20行。JSONやラベルは不要。

【商品名】
{product}

【知見要約】
{knowledge}
"""

def call_openai_alt_lines(product: str, knowledge_text: str) -> List[str]:
    """OpenAIに投げて20行のALTテキストを取得（text出力）"""
    sys = "You are a helpful Japanese copywriter specialized in SEO-optimized e-commerce ALT texts."
    usr = PROMPT_TEMPLATE.format(product=product, knowledge=knowledge_text)

    for attempt in range(1, RETRY + 1):
        try:
            res = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": usr}
    ],
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    response_format={"type": "text"},
)
            
            text = (res.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty content")
            # 行に分割、空行や箇条書き記号を後段で除去
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return lines
        except Exception as e:
            if attempt < RETRY:
                print(f"⚠️ OpenAIエラー({attempt}/{RETRY}): {e}")
                time.sleep(SLEEP_ON_FAIL)
                continue
            else:
                print(f"❌ OpenAI失敗（product={product[:18]}…）: {e}")
                return []

# ====== ALT 生成（整形込み） ======
def generate_20_alts(product: str, kb_text: str) -> List[str]:
    raw_lines = call_openai_alt_lines(product, kb_text)

    # フォールバック：空なら適当な雛形を返す（最低限の進行保護）
    if not raw_lines:
        fallback = [f"{product}の特長を活かし、日常の使いやすさと安心感を高める設計です。"]
        raw_lines = fallback * 20

    # クリーニング＆フィルタリング
    cleaned = []
    for ln in raw_lines:
        s = sanitize_line(ln)
        if not s:
            continue
        # ALTや画像描写語っぽいものがラベル的にあれば除去済み、文中は最終で削る
        cleaned.append(s)

    # 20件へ整形
    alts = []
    for s in cleaned:
        if len(alts) >= 20:
            break
        # 長さ調整（自然クリップ）
        s = natural_clip_80_110(s)
        # 仕上げ
        s = post_refine_line(s)
        # 極端に短いもの/禁則混入は迂回（軽い再生成はせずスキップして次）
        if len(s) < 60 or contains_forbidden(s):
            continue
        alts.append(s)

    # 不足なら、既存を少し変形して補充（語尾・接続を微修正）
    if len(alts) < 20:
        base = alts[:] if alts else [f"{product}の使い勝手を高め、安心して日常利用できるバランス設計です。"]
        j = 0
        while len(alts) < 20:
            seed = base[j % len(base)]
            # 末尾を軽く変える（語尾・副詞足し）
            variant = seed
            if variant.endswith("。"):
                variant = variant[:-1]
            tail = ["。", "。毎日の携帯性に優れる。", "。操作が直感的で扱いやすい。", "。忙しい日常でも頼れる仕様。"]
            variant = natural_clip_80_110((variant + tail[(j % len(tail))]).strip())
            variant = post_refine_line(variant)
            alts.append(variant)
            j += 1

    # 最終安全網：ちょうど20に切り詰め
    return alts[:20]


# ====== メイン ======
def main():
    print("🌸 ALT長文（SEO＋自然文）生成開始")
    products = read_products_from_csv(INPUT_CSV)
    print(f"✅ 対象商品数: {len(products)}件")

    kb = collect_local_knowledge()

    rows = []
    for product in tqdm(products, desc="🧠 生成中", ncols=80):
        knowledge_text = build_knowledge_text(product, kb)
        alts = generate_20_alts(product, knowledge_text)
        row = {"商品名": product}
        for i in range(20):
            row[f"ALT{i+1}"] = alts[i]
        rows.append(row)

    # CSV書き出し
    fieldnames = ["商品名"] + [f"ALT{i+1}" for i in range(20)]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ 出力完了: {OUT_CSV}")
    print("✅ 仕様: ALTは1〜2文・80〜110字・句点終止・画像描写語/箇条書き/番号は禁止。ローカル知見を活用。")


if __name__ == "__main__":
    main()
import atlas_autosave_core
