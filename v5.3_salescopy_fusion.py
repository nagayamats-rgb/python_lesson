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
import glob
import json
import time
import math
from collections import defaultdict

from dotenv import load_dotenv

# 進捗
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# OpenAI SDK（新）
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("❌ openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# ==========
# 定数群
# ==========
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INPUT_CSV = "/Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv"  # 固定
OUT_DIR   = os.path.join(BASE_DIR, "output", "ai_writer")
RAW_PATH  = os.path.join(OUT_DIR, "alt_text_ai_raw_salescopy_v5_3.csv")
REF_PATH  = os.path.join(OUT_DIR, "alt_text_refined_salescopy_v5_3.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_salescopy_v5_3.csv")

SEMANTICS_DIR = os.path.join(BASE_DIR, "output", "semantics")
PERSONA_PATH  = os.path.join(BASE_DIR, "config", "kotoha_persona.json")

# 生成レンジと整形レンジ
RAW_MIN, RAW_MAX   = 100, 130   # AIはまずこの長さを狙う
FINAL_MIN, FINAL_MAX = 80, 110  # ローカルでこの範囲に収める

# 禁則（画像描写語・ECメタ語・誇大）
FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "イメージ図",
    "当店", "当社", "レビュー", "ランキング", "クリック", "こちら", "リンク", "購入はこちら",
    "最安", "No.1", "ナンバーワン", "売上No1", "業界最高", "競合", "競合優位性", "返金保証"
]

# 正規表現
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
MULTI_COMMA_RE  = re.compile(r"、{3,}")
SPACE_RE        = re.compile(r"\s+")
EXTRA_BRACKETS  = re.compile(r"[【】\[\]]")  # 不要な装飾括弧を軽く除去

# 体言止めを許可（強制ではない）
ALLOW_TAIGEN = True

# =================
# 0) 環境/クライアント
# =================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")
    model = (os.getenv("OPENAI_MODEL") or "gpt-5").strip()
    temperature = float(os.getenv("OPENAI_TEMPERATURE") or "0.9")
    max_tokens  = int(os.getenv("OPENAI_MAX_TOKENS") or "1500")
    mode = (os.getenv("OPENAI_MODE") or "chat").strip().lower()  # 互換
    persona_switch = (os.getenv("KOTOHA_PERSONA") or "on").strip().lower() in ("on", "true", "1", "yes")
    client = OpenAI(api_key=api_key)
    return client, model, temperature, max_tokens, mode, persona_switch

# =================
# 1) データ読み込み
# =================
def load_products(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"❌ 入力CSVが見つかりません: {path}")
    products = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("❌ 入力CSVに『商品名』ヘッダが見つかりません。")
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                products.append(nm)
    # 順序維持の重複除去
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

# =================
# 2) 人格ロード
# =================
def load_persona(path: str, enabled: bool):
    if not enabled:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 期待キー: core_values, tone, style, seo, guardrails
        return data
    except Exception:
        return None

def build_persona_system(persona):
    """
    KOTOHA人格 → SYSTEMメッセージ化
    """
    if not persona:
        return None
    core = persona.get("core_values") or {}
    tone = persona.get("tone") or {}
    style = persona.get("style") or {}
    seo = persona.get("seo") or {}
    guard = persona.get("guardrails") or {}

    parts = []
    parts.append("あなたは『SEOに強い日本語コピーライター』です。楽天のサイト内SEOを最大化しつつ、自然で読みやすい文章を作成します。")
    if core:
        parts.append(f"信条: {', '.join([f'{k}:{v}' for k,v in core.items() if isinstance(v,str)])}")
    if tone:
        parts.append(f"トーン: {', '.join([f'{k}:{v}' for k,v in tone.items() if isinstance(v,str)])}")
    if style:
        parts.append(f"文体: {', '.join([f'{k}:{v}' for k,v in style.items() if isinstance(v,str)])}")
    if seo:
        parts.append(f"SEO指針: {', '.join([f'{k}:{v}' for k,v in seo.items() if isinstance(v,str)])}")
    if guard:
        parts.append(f"ガードレール: {', '.join([f'{k}:{v}' for k,v in guard.items() if isinstance(v,str)])}")
    parts.append("出力は20行のテキスト（各行1〜2文の自然文、句点「。」で終える、JSONやラベルなし）。")
    return " ".join(parts)

# =================
# 3) 知見サマリ
# =================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    """
    ./output/semantics/*.json を緩やかに吸収し、テキスト知見と追加禁則を返す。
    list/dict 混在・キー揺れに堅牢。
    """
    if not os.path.isdir(SEMANTICS_DIR):
        base = "知見: 商品名・対応機種・スペック・機能・用途・対象・ベネフィットを自然に織り込み、画像描写語は使わず、2文以内、句点終止。"
        return base, []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    clusters, market, semantics, persona_tone, template = [], [], [], [], []
    forbidden_local = []

    def flatten(x):
        if isinstance(x, list):
            for v in x:
                if isinstance(v, str):
                    yield v
                elif isinstance(v, dict):
                    for vv in flatten(list(v.values())):
                        yield vv
        elif isinstance(x, dict):
            for vv in flatten(list(x.values())):
                yield vv

    for p in files:
        name = os.path.basename(p).lower()
        data = safe_load_json(p)
        if data is None:
            continue
        try:
            if "lexical" in name:
                # 例: {"clusters":[{"terms":[...]}]} / [{"terms":[...]}] / ["語","彙"]
                if isinstance(data, dict):
                    arr = data.get("clusters") or data.get("lexical") or []
                else:
                    arr = data
                for c in flatten(arr):
                    if isinstance(c, str):
                        clusters.append(c)
            elif "market_vocab" in name or "market" in name:
                for v in flatten(data):
                    if isinstance(v, str):
                        market.append(v)
            elif "structured_semantics" in name or "semantic" in name:
                # 例: {"concepts":[...], "scenes":[...], "targets":[...], "use_cases":[...]}
                if isinstance(data, dict):
                    for k in ["concepts", "semantics", "frames", "features", "facets", "benefits", "targets", "scenes", "use_cases"]:
                        for v in data.get(k, []) or []:
                            if isinstance(v, str):
                                semantics.append(v)
                else:
                    for v in flatten(data):
                        if isinstance(v, str):
                            semantics.append(v)
            elif "styled_persona" in name or "persona" in name:
                if isinstance(data, dict):
                    t = data.get("tone") or {}
                    for v in t.values():
                        if isinstance(v, str):
                            persona_tone.append(v)
                else:
                    for v in flatten(data):
                        if isinstance(v, str):
                            persona_tone.append(v)
            elif "normalized" in name or "forbid" in name:
                if isinstance(data, dict):
                    fw = data.get("forbidden_words") or []
                    for w in fw:
                        if isinstance(w, str):
                            forbidden_local.append(w)
            elif "template_composer" in name or "template" in name:
                for v in flatten(data):
                    if isinstance(v, str):
                        template.append(v)
        except Exception:
            # 形式不一致は無視して続行
            pass

    # ユニーク化して詰めすぎない
    cap = lambda xs, n: [x for i, x in enumerate(xs) if isinstance(x, str) and xs.index(x) == i][:n]
    clusters = cap(clusters, 15)
    market   = cap(market,   15)
    semantics= cap(semantics,10)
    persona_tone = cap(persona_tone, 6)
    template = cap(template, 4)

    parts = []
    if clusters: parts.append("語彙: " + "、".join(clusters))
    if market:   parts.append("市場語: " + "、".join(market))
    if semantics:parts.append("構造: " + "、".join(semantics))
    if template: parts.append("骨子: " + "、".join(template))
    if persona_tone: parts.append("トーン: " + "、".join(persona_tone))

    text = "知見: "
    if parts:
        text += " / ".join(parts) + "。"
    text += "画像描写語は使わず、楽天のサイト内SEOに効く自然文で、各行は1〜2文、句点で終える。"
    return text, list({*FORBIDDEN, *forbidden_local})

# =================
# 4) プロンプト生成
# =================
BASE_SYSTEM = (
    "あなたは『SEOに強い日本語コピーライター』です。"
    "楽天のサイト内SEOを最大化しつつ、自然で読みやすいALTテキストを生成します。"
    "出力は20行のテキスト（各行1〜2文、句点「。」で終える、JSONやラベルなし）。"
)

def build_user_prompt(product: str, knowledge_text: str, forbidden_words):
    forbid_txt = "、".join(sorted(set([w for w in forbidden_words if isinstance(w, str)])))
    hint = (
        "構成ヒント（テンプレではなく自然に）："
        "商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット。"
    )
    rules = (
        f"禁止語（必ず使わない）: {forbid_txt}\n"
        f"各行は全角約{RAW_MIN}〜{RAW_MAX}文字。体言止めは必要に応じて可。"
    )
    return (
        f"商品名: {product}\n"
        f"{knowledge_text}\n"
        f"{hint}\n"
        f"{rules}\n"
        "20行で出力。各行は自然文（1〜2文）で、句点「。」で終えること。"
    )

# =================
# 5) OpenAI 呼び出し
# =================
def call_openai_20_lines(client, model, temperature, max_tokens, system_prompt, user_prompt, retry=3, wait=6):
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "text"},
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            content = (res.choices[0].message.content or "").strip()
            if content:
                lines = [LEADING_ENUM_RE.sub("", ln).strip("・-—●　").strip() for ln in content.split("\n")]
                lines = [EXTRA_BRACKETS.sub("", ln) for ln in lines if ln.strip()]
                return lines[:80]  # 念のため多めに保持
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答を取得できませんでした: {last_err}")

# =================
# 6) ローカル整形
# =================
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 異常な番号・箇条書き剥がし
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")
    # スペース圧縮 & 連続読点縮約
    t = SPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    # 禁則語の完全除去（単純置換）
    for ng in FORBIDDEN:
        if ng and ng in t:
            t = t.replace(ng, "")
    t = t.strip()

    # 句点終止を強制。ただし体言止め許容の場合は、2文以内で句点ナシも残すが最終1文は基本「。」で締める
    if not ALLOW_TAIGEN or ("。" not in t):
        if not t.endswith("。"):
            t += "。"

    # 120超なら文末「。」までカット
    if len(t) > 120:
        cut = t[:120]
        pos = cut.rfind("。")
        t = cut[:pos+1] if pos != -1 else cut

    return t.strip()

def is_sentence_like(s: str) -> bool:
    if not s: return False
    # 10文字未満は短すぎ
    if len(s) < 10:
        return False
    # 句点や助詞がまるで無い単語羅列は落とす
    has_punct = ("。" in s) or ("、" in s)
    has_particle = any(p in s for p in ["を", "に", "で", "が", "と", "も", "へ", "より", "から"])
    return has_punct or has_particle

def de_duplicate(lines):
    uniq, seen = [], set()
    for ln in lines:
        key = ln
        if key not in seen:
            uniq.append(ln)
            seen.add(key)
    return uniq

def light_variation(s: str) -> str:
    if not s: return s
    s2 = s
    # 語尾の軽いブレで重複回避
    s2 = s2.replace("します。", "できます。")
    s2 = s2.replace("できます。", "しやすいです。")
    s2 = s2.replace("です。", "になります。")
    if s2 == s:
        s2 = re.sub(r"([^\s、。]{2,})", r"\1、", s, count=1)
        s2 = s2.replace("、、", "、")
        if not s2.endswith("。"):
            s2 += "。"
    return soft_clip_sentence(s2)

def fallback_template(product: str) -> list:
    """
    欠損/短文用の2文テンプレ（自然文ベース）
    """
    base = [
        f"{product}の機能とスペックを活かし、日常の不便を減らす設計。使いやすさと耐久性を両立し、幅広い機種で快適に使えます。",
        f"{product}はビジネスから普段使いまでマルチに活躍。装着や設定が簡単で、持ち運びやすく、毎日の小さな手間を減らします。",
        f"{product}は高い互換性と安定性が特長。複数デバイスの切替えや外出先でもスムーズに使え、仕事とプライベートの両立を支えます。",
        f"{product}は軽量・コンパクトなうえ耐久性にも配慮。自宅やオフィス、旅行先でも取り回しが良く、ストレスなく使えます。",
    ]
    return [soft_clip_sentence(x) for x in base]

def refine_block(raw_lines, product: str):
    # 正規化 → 自然文フィルタ → 重複除去 → 長さ整形
    norm = []
    for ln in raw_lines:
        if not ln: 
            continue
        ln = EXTRA_BRACKETS.sub("", ln)
        ln = soft_clip_sentence(ln)
        if is_sentence_like(ln):
            norm.append(ln)

    # 重複除去
    norm = de_duplicate(norm)

    # 足りなければテンプレ補完（2文単位）
    i = 0
    while len(norm) < 20:
        fb = fallback_template(product)
        # 軽いバリエーションを足す
        norm.extend([light_variation(x) for x in fb])
        norm = de_duplicate(norm)
        i += 1
        if i > 5:  # 無限拡張防止
            break

    # 多すぎれば先頭20本
    return norm[:20]

# =================
# 7) 書き出し
# =================
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def write_raw(products, all_raw):
    with open(RAW_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_raw):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_refined(products, all_refined):
    with open(REF_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["商品名"] + [f"ALT_{i+1}" for i in range(20)])
        for p, lines in zip(products, all_refined):
            row = [p] + (lines[:20] + [""] * max(0, 20 - len(lines)))
            w.writerow(row)

def write_diff(products, all_raw, all_refined):
    with open(DIFF_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_refined_{i+1}" for i in range(20)]
        w.writerow(header)
        for p, r, ref in zip(products, all_raw, all_refined):
            r_line = (r[:20] + [""] * max(0, 20 - len(r)))
            ref_line = (ref[:20] + [""] * max(0, 20 - len(ref)))
            w.writerow([p] + r_line + ref_line)

# =================
# 8) メイン
# =================
def main():
    print("🌸 ALT生成 v5.3（SEO自然文＋KOTOHA人格＋知見連携）")
    client, model, temperature, max_tokens, mode, persona_on = init_env_and_client()
    ensure_outdir()

    # 人格
    persona = load_persona(PERSONA_PATH, persona_on)
    system_prompt = build_persona_system(persona) if persona else BASE_SYSTEM

    # 知見
    kb_text, forbidden_all = summarize_knowledge()

    # 商品
    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    all_raw, all_refined = [], []

    for p in tqdm(products, desc="🧠 生成中", total=len(products)):
        user_prompt = build_user_prompt(p, kb_text, forbidden_all)
        try:
            raw_lines = call_openai_20_lines(
                client, model, temperature, max_tokens,
                system_prompt, user_prompt, retry=3, wait=6
            )
        except Exception:
            # ダウン時のフェイルセーフ
            raw_lines = fallback_template(p) * 5

        refined = refine_block(raw_lines, p)

        all_raw.append(raw_lines[:20])
        all_refined.append(refined)

        time.sleep(0.2)  # 軽いスロットリング

    # 書き出し
    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens))) if lens else 0.0

    print("✅ 出力完了:")
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
