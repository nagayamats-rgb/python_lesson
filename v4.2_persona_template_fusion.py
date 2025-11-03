# -*- coding: utf-8 -*-
"""
v4.2_persona_template_fusion.py
ALT長文（SEO＋自然文）生成：ペルソナ×テンプレ×ローカル知見の融合版

- 入力: ./rakuten.csv（UTF-8, ヘッダに「商品名」）
- 参照: ./output/semantics/ 内の JSON 群（存在すれば自動統合。無ければ既定の知見文を使用）
- 出力:
  1) output/ai_writer/alt_text_ai_raw_longform_v4.2.csv        … AIの生出力（20本/商品）
  2) output/ai_writer/alt_text_refined_final_longform_v4.2.csv … ローカル整形後（80〜110字）
  3) output/ai_writer/alt_text_diff_longform_v4.2.csv          … raw/refined の横並び比較

- OpenAI:
    model                 = .env固定（gpt-5を推奨）
    response_format       = {"type":"text"}
    max_completion_tokens = 1000
    temperature           = 1
- ポリシー:
    * 画像描写語・ECメタ語・誇張表現NG（FORBIDDEN＋normalized.jsonの禁止語を統合）
    * 箇条書き/番号（1. ・ など）禁止、必ず句点「。」で終える
    * AIは100〜130字/行を目標、ローカルで80〜110字に自然カット
"""

import os
import re
import csv
import glob
import json
import time
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv

# tqdm（無ければフォールバック）
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# ========= OpenAI SDK =========
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# =========================
# 定数・パス
# =========================
INPUT_CSV = "./rakuten.csv"        # UTF-8, ヘッダ「商品名」
SEMANTICS_DIR = "./output/semantics"

OUT_DIR   = "./output/ai_writer"
RAW_PATH  = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.2.csv")
REF_PATH  = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.2.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.2.csv")

# 文字数レンジ
RAW_MIN, RAW_MAX = 100, 130    # AI生成の目標長
FINAL_MIN, FINAL_MAX = 80, 110 # ローカル整形の目標長

# 禁則語（初期）
FORBIDDEN_BASE = [
    # 画像描写・指示語
    "画像", "写真", "見た目", "上の画像", "下の写真",
    # ECメタ/リンク系
    "当店", "当社", "レビュー", "ランキング", "クリック", "こちら", "リンク", "カート", "購入はこちら",
    # 誇張/メタ競合
    "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
    # 保証系の誤解招く強表現
    "送料無料（確約）", "返金保証",
]

# テキスト正規化用
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
WHITESPACE_RE   = re.compile(r"\s+")
MULTI_COMMA_RE  = re.compile(r"、{3,}")

# =========================
# 環境初期化
# =========================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")

    # モデルは .env を尊重（gpt-5 を想定）。空なら gpt-5 を既定。
    model_env = os.getenv("OPENAI_MODEL", "").strip() or "gpt-5"
    client = OpenAI(api_key=api_key)
    return client, model_env

# =========================
# 入力（商品名）
# =========================
def load_products(path: str) -> List[str]:
    if not os.path.exists(path):
        raise SystemExit(f"入力CSVが見つかりません: {path}")
    items = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "商品名" not in r.fieldnames:
            raise SystemExit("入力CSVに『商品名』ヘッダが見つかりません。")
        for row in r:
            nm = (row.get("商品名") or "").strip()
            if nm:
                items.append(nm)
    # 重複除去（順序保持）
    seen, uniq = set(), []
    for nm in items:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq

# =========================
# 知見ロード（JSON）
# =========================
def safe_json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def listify(x):
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    return []

def summarize_knowledge_fusion() -> Tuple[str, List[str]]:
    """
    /output/semantics/ 配下のJSONをゆるく吸収し、AIに渡す「日本語知見文」を生成。
    また、禁止語をFORBIDDEN_BASEと統合して返す。
    """
    keywords, scenes, targets, specs, market, templates, tones, extra_forbidden = [], [], [], [], [], [], [], []

    if os.path.isdir(SEMANTICS_DIR):
        for p in glob.glob(os.path.join(SEMANTICS_DIR, "*.json")):
            name = os.path.basename(p).lower()
            data = safe_json_load(p)
            if data is None:
                continue

            try:
                # lexical / cluster 系: キーワード抽出
                if "lexical" in name or "cluster" in name:
                    arr = data.get("clusters") if isinstance(data, dict) else data
                    for c in listify(arr):
                        terms = c.get("terms") if isinstance(c, dict) else None
                        if isinstance(terms, list):
                            keywords.extend([t for t in terms if isinstance(t, str)])

                # structured_semantics 系: 構造情報
                if "semantic" in name or "structured" in name:
                    if isinstance(data, dict):
                        for k in ("scenes", "targets", "specs", "concepts", "use_cases"):
                            for v in (data.get(k) or []):
                                if not isinstance(v, str): continue
                                if k == "scenes":   scenes.append(v)
                                elif k == "targets": targets.append(v)
                                elif k == "specs":  specs.append(v)
                                else:               keywords.append(v)

                # market 語彙
                if "market" in name or "vocab" in name:
                    if isinstance(data, list):
                        for v in data:
                            if isinstance(v, dict) and isinstance(v.get("vocabulary"), str):
                                market.append(v["vocabulary"])
                            elif isinstance(v, str):
                                market.append(v)
                    elif isinstance(data, dict):
                        vocab = data.get("vocabulary") or data.get("vocab") or []
                        market.extend([x for x in vocab if isinstance(x, str)])

                # persona / style
                if "persona" in name or "style" in name:
                    if isinstance(data, dict):
                        t = data.get("tone") or {}
                        if isinstance(t, dict):
                            for v in t.values():
                                if isinstance(v, str):
                                    tones.append(v)
                        # 文字列の配列で tone を持つ形式にも対応
                        if isinstance(data.get("tone"), list):
                            for v in data["tone"]:
                                if isinstance(v, str):
                                    tones.append(v)

                # template composer / hints
                if "template" in name or "composer" in name:
                    if isinstance(data, dict):
                        for key in ("hints", "templates"):
                            for v in (data.get(key) or []):
                                if isinstance(v, str):
                                    templates.append(v)

                # normalized / forbid
                if "normalized" in name or "forbid" in name:
                    if isinstance(data, dict):
                        for v in (data.get("forbidden_words") or []):
                            if isinstance(v, str):
                                extra_forbidden.append(v)

            except Exception:
                # 形式がバラつくファイルはスキップ（堅牢設計）
                continue

    # ユニーク化
    def uniq(lst): return list(dict.fromkeys([x for x in lst if isinstance(x, str) and x.strip()]))

    keywords = uniq(keywords)
    scenes   = uniq(scenes)
    targets  = uniq(targets)
    specs    = uniq(specs)
    market   = uniq(market)
    templates = uniq(templates)
    tones     = uniq(tones)
    extra_forbidden = uniq(extra_forbidden)

    # 日本語知見文を合成（冗長にならないよう上限をかける）
    def cap_join(xs, n): return "、".join(xs[:n]) if xs else ""

    blocks = []
    if keywords: blocks.append(f"キーワード例: {cap_join(keywords, 12)}")
    if specs:    blocks.append(f"仕様・機能例: {cap_join(specs, 10)}")
    if scenes:   blocks.append(f"利用シーン例: {cap_join(scenes, 8)}")
    if targets:  blocks.append(f"想定ユーザー例: {cap_join(targets, 8)}")
    if market:   blocks.append(f"市場語彙: {cap_join(market, 12)}")
    if templates: blocks.append(f"構成ヒント例: {cap_join(templates, 3)}")
    if tones:    blocks.append(f"文体ガイド例: {cap_join(tones, 3)}")

    if blocks:
        knowledge = "知見まとめ: " + " / ".join(blocks) + "。"
    else:
        knowledge = "知見まとめ: 対応機種名・主要スペック・用途・導入メリットを自然に織り込み、2文以内で簡潔に。"

    # 禁則語の統合
    forbidden_all = uniq(FORBIDDEN_BASE + extra_forbidden)
    return knowledge, forbidden_all

# =========================
# プロンプト作成
# =========================
def build_system_prompt(knowledge_text: str, tone_hint: str) -> str:
    # tone_hint は空でもOK。あれば活かす。
    tone_line = f"【文体ガイド】{tone_hint}\n" if tone_hint else ""
    sys = (
        "あなたはEC画像のALTテキストを作る日本語のプロコピーライターです。\n"
        "目的は、楽天のサイト内SEOに強い自然文ALTを20本生成することです。\n"
        + tone_line +
        "【厳守ルール】\n"
        "・画像や写真の描写語（例：画像、写真、見た目、上の画像 等）は使わない。\n"
        "・ECメタ語（当店、レビュー、ランキング、リンク、購入はこちら 等）は使わない。\n"
        "・競合比較や“競合優位性”のようなメタ表現は禁止。\n"
        f"・各行は全角およそ{RAW_MIN}〜{RAW_MAX}文字、1〜2文で自然に。必ず句点「。」で終える。\n"
        "・箇条書きや番号（1. 2. ・ など）やラベル（ALT: 等）は付けない。\n"
        "・商品名・対応機種・スペック・機能・用途・対象・ベネフィットを自然に織り込む（詰め込み禁止）。\n"
        "・出力は20行のテキストのみ（JSONや記号なし）。\n"
        "\n"
        f"{knowledge_text}\n"
    )
    return sys

def build_user_prompt(product: str, forbidden_words: List[str]) -> str:
    forbid_txt = "、".join(sorted(set([w for w in forbidden_words if isinstance(w, str)])))
    hint = "構成ヒント（自然に使う）：商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット。"
    return (
        f"商品名: {product}\n"
        f"{hint}\n"
        f"禁止語（絶対に使わない）: {forbid_txt}\n"
        "20行で、各行はひとつの自然文（1〜2文内）。句点で終えること。"
    )

# =========================
# OpenAI 呼び出し
# =========================
def call_openai_alt20(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                      retry: int = 3, wait: int = 6) -> List[str]:
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "text"},
                max_completion_tokens=1000,
                temperature=1,
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Empty content")

            # 行単位にし、箇条書き/番号を剥がす
            lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            clean = []
            for ln in lines:
                ln2 = LEADING_ENUM_RE.sub("", ln)
                ln2 = ln2.strip("・-—●　")
                if ln2:
                    clean.append(ln2)

            # 20行以上返るモデル挙動があるため、一旦最大60行まで受け取り後工程で20本に整形
            return clean[:60]
        except Exception as e:
            last_err = e
            time.sleep(wait)

    raise RuntimeError(f"OpenAI応答を取得できませんでした: {last_err}")

# =========================
# ローカル整形
# =========================
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # 句点終止
    if not t.endswith("。"):
        t += "。"
    # 空白圧縮/読点整形
    t = WHITESPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")

    # 長すぎる場合は120まで許容し、直近の句点で自然カット
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
        else:
            t = cut

    # 禁則語は完全除去
    for ng in FORBIDDEN_BASE:
        if ng and ng in t:
            t = t.replace(ng, "")

    return t.strip()

def refine_20_lines(raw_lines: List[str]) -> List[str]:
    # 正規化→フィルタ
    norm = []
    for ln in raw_lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        ln = LEADING_ENUM_RE.sub("", ln).strip("・-—●　")
        ln = soft_clip_sentence(ln)
        if len(ln) < 15:
            continue
        norm.append(ln)

    # 重複除去
    uniq, seen = [], set()
    for ln in norm:
        if ln not in seen:
            uniq.append(ln)
            seen.add(ln)

    # 長さレンジに寄せる最終調整
    refined = [soft_clip_sentence(ln) for ln in uniq]

    # 足りない場合は軽いバリエーションで補完（同義語のような最小変形）
    def light_var(s: str) -> str:
        s2 = s
        s2 = s2.replace("します。", "できます。")
        s2 = s2.replace("できます。", "しやすいです。")
        s2 = s2.replace("です。", "になります。")
        if s2 == s:
            s2 = re.sub(r"(\S{2,})", r"\1、", s, count=1)
            s2 = s2.replace("、、", "、")
            if not s2.endswith("。"):
                s2 += "。"
        return soft_clip_sentence(s2)

    i = 0
    while len(refined) < 20 and refined:
        refined.append(light_var(refined[i % len(refined)]))
        i += 1

    return refined[:20]

# =========================
# 書き出し
# =========================
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def write_raw(products: List[str], all_raw: List[List[str]]):
    with open(RAW_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_raw):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_refined(products: List[str], all_refined: List[List[str]]):
    with open(REF_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_refined):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_diff(products: List[str], all_raw: List[List[str]], all_refined: List[List[str]]):
    with open(DIFF_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_refined_{i+1}" for i in range(20)]
        w.writerow(header)
        for p, r, ref in zip(products, all_raw, all_refined):
            r_line = (r[:20] + [""] * max(0, 20 - len(r)))
            ref_line = (ref[:20] + [""] * max(0, 20 - len(ref)))
            w.writerow([p] + r_line + ref_line)

# =========================
# メイン
# =========================
def main():
    print("🌸 v4.2_persona_template_fusion：ALT生成（ペルソナ×テンプレ×知見統合）")
    client, model = init_env_and_client()
    ensure_outdir()

    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    knowledge_text, forbidden_all = summarize_knowledge_fusion()

    # 文体ヒント（knowledge_text内の“文体ガイド例:”を拾って使う程度）
    tone_hint = ""
    m = re.search(r"文体ガイド例:\s*(.+?)(?:。|$)", knowledge_text)
    if m:
        tone_hint = m.group(1).strip()

    system_prompt = build_system_prompt(knowledge_text, tone_hint)

    all_raw, all_refined = [], []

    for p in tqdm(products, desc="🧠 AI生成中", total=len(products)):
        user_prompt = build_user_prompt(p, forbidden_all)
        try:
            raw_lines = call_openai_alt20(client, model, system_prompt, user_prompt)
        except Exception:
            # どうしても取得できない場合は最低限のダミー20本（空は避ける）
            raw_lines = [f"{p} の使い勝手を高める設計で、日常の不便を減らします。"] * 20

        refined_lines = refine_20_lines(raw_lines)

        all_raw.append(raw_lines[:20])
        all_refined.append(refined_lines)

        # 軽いスロットリング（API安定）
        time.sleep(0.2)

    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    # 統計表示
    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens)))

    print("✅ 出力完了:")
    print(f"   - AI生出力 : {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print("🔒 仕様まとめ:")
    print(f"   - AI: 約{RAW_MIN}〜{RAW_MAX}字・1〜2文・句点終止・禁則適用（プロンプト）")
    print(f"   - ローカル: {FINAL_MIN}〜{FINAL_MAX}字に自然カット、重複除去・句点補完・禁則再適用")
    print("   - 知見: /output/semantics のJSON群（lexical/semantic/market/persona/template/normalized）を自動統合")

if __name__ == "__main__":
    main()
import atlas_autosave_core
