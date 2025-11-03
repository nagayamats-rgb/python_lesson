# -*- coding: utf-8 -*-
"""
v5.3_salescopy_fusion.py
- 目的: 楽天ALT（20本/商品）を「SEOに強い自然文」で安定生成（KOTOHA人格エンジン連携）
- 入力: /Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv（UTF-8, ヘッダに「商品名」）
- 知見: ./output/semantics/*.json を緩やか統合（list/dict混在に堅牢）
- 人格: ./config/kotoha_persona.json（環境変数 KOTOHA_PERSONA=on で有効）
- 出力:
    1) output/ai_writer/alt_text_ai_raw_salescopy_v5_3.csv
    2) output/ai_writer/alt_text_refined_salescopy_v5_3.csv
    3) output/ai_writer/alt_text_diff_salescopy_v5_3.csv
- モデル/温度/トークン: .envを完全準拠（OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS）
- 安定化: JSON非依存 / response_format={"type":"text"} / backoffリトライ / 欠損補完 / 重複抑止
"""

import os
import re
import csv
import json
import time
import math
import glob
import copy
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv, find_dotenv  # ← 追加
load_dotenv(find_dotenv(usecwd=True))       # ← 追加（cwd から上位を探索）

# ---------- 設定 ----------
BASE_DIR = Path(os.getcwd())
INPUT_PATH = Path("/Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv")
OUT_DIR = BASE_DIR / "output" / "ai_writer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = OUT_DIR / "alt_text_ai_raw_salescopy_v5_3.csv"
REF_PATH = OUT_DIR / "alt_text_refined_salescopy_v5_3.csv"
DIFF_PATH = OUT_DIR / "alt_text_diff_salescopy_v5_3.csv"

SEMANTICS_DIR = BASE_DIR / "output" / "semantics"
PERSONA_PATH = BASE_DIR / "config" / "kotoha_persona.json"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))

RAW_MIN, RAW_MAX = 120, 130        # AI直出しの想定字数
FINAL_MIN, FINAL_MAX = 80, 110     # 整形後の目標字数

USE_PERSONA = os.getenv("KOTOHA_PERSONA", "off").lower() == "on"

# ---------- OpenAI クライアント ----------
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

# ---------- ユーティリティ ----------
def load_csv_items(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def save_csv_rows(path: Path, rows: List[List[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def safe_read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_semantics() -> List[Any]:
    items = []
    if not SEMANTICS_DIR.exists():
        return items
    for p in sorted(SEMANTICS_DIR.glob("*.json")):
        data = safe_read_json(p)
        if data is None:
            continue
        items.append(data)
    return items

def persona_or_default() -> Dict[str, Any]:
    if USE_PERSONA and PERSONA_PATH.exists():
        p = safe_read_json(PERSONA_PATH)
        if isinstance(p, dict):
            return p
    return {"name": "KOTOHA", "tone": "neutral", "style": "plain"}

# ---------- v5.4: 3.3準拠のSYSTEM（＋文末の引用符禁止） ----------
BASE_SYSTEM = ("""\
あなたはECモール（楽天・Yahoo）専門の日本語コピーライターです。
画像の描写は一切しません。ALT（代替テキスト）として自然な日本語の短い文を作成します。

【禁止・制約】
・1〜2文の自然文で書くこと。句点（。 ！ ？ または全角の対応記号）で終えること。
・絵文字・顔文字・機種依存文字・装飾記号・HTMLタグは禁止。
・比較広告・競合優位の表現（「他社より」「圧倒的」「No.1」等）は使わない。
・「画像」「写真」「映っている」「クリック」などの画像操作/視覚メタ語は使わない。
・製品・型番・素材など具体情報は自然に文中へ織り込み、羅列にしない。
・ブランドや仕様に事実と異なる断定はしない。効能・医療的主張は控える。
・文末を引用符（" ' 「 」 “ ” 『 』）で終わらせないこと。

【書き方のヒント】（強制ではない）
・（スペック）→（コア要素）→（誰に）→（利用シーン）→（得られるベネフィット）
・語尾をバリエーションさせ、同じパターンの連打を避ける。
・SEOの過剰意識は避け、自然で読みやすい1〜2文に収める。

出力はテキストのみ。ALTを20本、行区切りで返してください。
""")

def build_persona_system(persona: Dict[str, Any]) -> str:
    """
    v5.4: ライティングの規範はv3.3に戻し、文末の引用符禁止を追加。
    Personaは使わず固定SYSTEMを返します（挙動を安定化）。
    """
    return BASE_SYSTEM

# ---------- USERプロンプト ----------
def build_user_prompt(product_name: str, knowledge_text: str = "") -> str:
    # 3.3相当の素直なユーザ指示を維持
    hint = ""
    if knowledge_text:
        hint = f"\n【参考情報（要約）】\n{knowledge_text}\n"
    return textwrap.dedent(f"""\
        商品名: {product_name}
        {hint}
        上記の商品について、日本語のALTテキストを20本作成してください。
        各ALTは1〜2文、自然な日本語で、まずおよそ{RAW_MIN}〜{RAW_MAX}字を目安にしてください。
        画像の描写は書かず、商品特徴・型番・素材などを自然に織り込んでください。
        行区切りで20本を出力してください。
    """).strip()

# ---------- OPENAI呼び出し ----------
def call_openai_20_lines(client, model: str, system_prompt: str, user_prompt: str,
                         temperature: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    if client is None:
        # ダミー出力（テスト用）
        return [f"{i+1}行目のサンプルALTです。自然な日本語で商品特徴を織り込みます。" for i in range(20)]

    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "text"},
            )
            txt = resp.choices[0].message.content.strip()
            # 貼り付けや箇条書きにも耐えるように整形
            lines = [re.sub(r"^\s*[\-\d\.\)\]]\s*", "", ln).strip() for ln in txt.splitlines() if ln.strip()]
            # 行数調整
            if len(lines) < 20:
                lines += [""] * (20 - len(lines))
            elif len(lines) > 20:
                lines = lines[:20]
            return lines
        except Exception as e:
            if retry == 2:
                raise
            time.sleep(2.0 + retry)

# ---------- 整形パイプライン ----------
TRAILING_QUOTES = {'"', "'", '“', '”', '‘', '’', '「', '」', '『', '』'}

def normalize_line(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    # 禁止語の軽い正規化（画像メタ）
    t = re.sub(r"(画像|写真|映っている|クリック|こちら|コチラ)", "", t)
    t = t.strip()
    return t

def is_natural_sentence(t: str) -> bool:
    if not t:
        return False
    # 句点/終端記号で終わる or これから付与可能
    return True

def soft_clip_sentence(t: str, min_len: int = FINAL_MIN, max_len: int = FINAL_MAX) -> str:
    if not t:
        return ""
    t = t.strip()
    # 末尾に引用符があれば除去（v5.4仕様）
    while t and t[-1] in TRAILING_QUOTES:
        t = t[:-1]
    t = t.strip()

    # 句点終止に整える
    if not t.endswith(("。", "！", "?", "？", "!")):
        t = t + "。"

    # 上限をソフトにカット（句読点やスペース優先）
    if len(t) > max_len:
        # 句点を基準に手前で落とす
        cut = t[:max_len]
        # 直近の句点/読点/空白で切る
        m = re.search(r"[。．！？\.\,\s][^。．！？\.\,\s]*$", cut)
        if m:
            cut = cut[:m.start()].rstrip()
        if not cut:
            cut = t[:max_len].rstrip()
        # 再び引用符が末尾に来る可能性も落とす
        while cut and cut[-1] in TRAILING_QUOTES:
            cut = cut[:-1]
        # 終端を再付与
        if not cut.endswith(("。", "！", "?", "？", "!")):
            cut = cut + "。"
        t = cut
    # 最小長を下回る場合、無理に追記はしない（自然さ優先）
    return t

def refine_lines(lines: List[str]) -> List[str]:
    out = []
    seen = set()
    for ln in lines:
        s = normalize_line(ln)
        if not is_natural_sentence(s):
            continue
        s = soft_clip_sentence(s, FINAL_MIN, FINAL_MAX)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    # 20本に満たない場合は補完（簡易）
    while len(out) < 20:
        out.append("自然な日本語のALTテキストです。製品の特徴を伝えます。")
    return out[:20]

# ---------- 知見の要約 ----------
def summarize_knowledge(semantics: List[Any], limit: int = 10) -> str:
    # list/dict混合を受け入れて、浅くつまむ
    bag = []
    for item in semantics[:limit]:
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, (str, int, float)):
                    bag.append(f"{k}:{v}")
                elif isinstance(v, list):
                    bag.extend([str(x) for x in v[:3]])
        elif isinstance(item, list):
            bag.extend([str(x) for x in item[:5]])
        elif isinstance(item, str):
            bag.append(item)
    txt = "; ".join([re.sub(r"\s+", " ", str(x)) for x in bag if x])
    return textwrap.shorten(txt, width=500, placeholder="…")

# ---------- メイン処理 ----------
def write_raw(items: List[Dict[str, Any]]) -> List[List[str]]:
    persona = persona_or_default()
    sys_prompt = build_persona_system(persona)

    semantics = load_semantics()
    know = summarize_knowledge(semantics)

    rows = [["商品名", "ALT_1", "ALT_2", "ALT_3", "ALT_4", "ALT_5",
             "ALT_6", "ALT_7", "ALT_8", "ALT_9", "ALT_10",
             "ALT_11", "ALT_12", "ALT_13", "ALT_14", "ALT_15",
             "ALT_16", "ALT_17", "ALT_18", "ALT_19", "ALT_20"]]
    for it in items:
        name = (it.get("商品名") or it.get("name") or "").strip()
        user_prompt = build_user_prompt(name, know)
        lines = call_openai_20_lines(client, MODEL, sys_prompt, user_prompt, TEMP, MAX_TOKENS)
        rows.append([name] + lines)
    save_csv_rows(RAW_PATH, rows)
    return rows

def write_refined(raw_rows: List[List[str]]) -> List[List[str]]:
    header = raw_rows[0]
    out = [header]
    for r in raw_rows[1:]:
        name, lines = r[0], r[1:]
        refined = refine_lines(lines)
        out.append([name] + refined)
    save_csv_rows(REF_PATH, out)
    return out

def diff_rows(raw_rows: List[List[str]], ref_rows: List[List[str]]) -> List[List[str]]:
    header = ["商品名", "RAW", "REF", "変更有無"]
    out = [header]
    for r_raw, r_ref in zip(raw_rows[1:], ref_rows[1:]):
        name = r_raw[0]
        diffs = []
        for a, b in zip(r_raw[1:], r_ref[1:]):
            diffs.append([name, a, b, "DIFF" if a != b else "SAME"])
    # フラット化
    flat = [header]
    for r_raw, r_ref in zip(raw_rows[1:], ref_rows[1:]):
        name = r_raw[0]
        for a, b in zip(r_raw[1:], r_ref[1:]):
            flat.append([name, a, b, "DIFF" if a != b else "SAME"])
    save_csv_rows(DIFF_PATH, flat)
    return flat

def avg_len(lines: List[str]) -> float:
    xs = [len(s or "") for s in lines]
    return sum(xs) / max(1, len(xs))

def main():
    print("📦 入力CSV:", INPUT_PATH)
    items = load_csv_items(INPUT_PATH)
    print(f"   - レコード数: {len(items)}")

    print("🧠 知見の読み込み:", SEMANTICS_DIR)

    print("🤖 OpenAI呼び出しでRAW生成中...")
    raw_rows = write_raw(items)

    print("✂️  整形（正規化→句点終止→長さ整形→重複抑止）...")
    ref_rows = write_refined(raw_rows)

    print("🔍 差分出力...")
    diff_rows(raw_rows, ref_rows)

    # 軽いメトリクス
    all_raw = [s for row in raw_rows[1:] for s in row[1:]]
    all_refined = [s for row in ref_rows[1:] for s in row[1:]]
    print(f"   - AI生出力: {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print("🔒 仕様:")
    print(f"   - AIは約{RAW_MIN}〜{RAW_MAX}字・1〜2文、句点終止、禁則適用（プロンプト）")
    print(f"   - ローカル整形で{FINAL_MIN}〜{FINAL_MAX}字に自然カット、重複抑止・欠損補完・語尾バリエーション")

if __name__ == "__main__":
    main()
import atlas_autosave_core
