# -*- coding: utf-8 -*-
"""
v4.3_natural_language_reinforce.py
ALT長文（楽天専用 / 自然文強化・体言止め許容・知見ゆる結合・raw/refined/diff 出力）

入力:
  - ./rakuten.csv  （UTF-8 / ヘッダに「商品名」必須）

知見（任意・あれば活用）:
  - ./output/semantics/*.json   （形式バラバラOK、ゆる結合で要約）
    例: lexical_clusters_*.json / structured_semantics_*.json / market_vocab_*.json /
        styled_persona_*.json / normalized_*.json / template_composer.json など

出力:
  - ./output/ai_writer/alt_text_ai_raw_longform_v4.3.csv
  - ./output/ai_writer/alt_text_refined_final_longform_v4.3.csv
  - ./output/ai_writer/alt_text_diff_longform_v4.3.csv

OpenAI:
  - .env にて固定（例）
      OPENAI_API_KEY="..."
      OPENAI_MODEL="gpt-5"
      OPENAI_MODE="chat"
      OPENAI_TEMPERATURE="1"
      OPENAI_MAX_TOKENS="1000"

呼び出し固定:
  - response_format={"type": "text"}
  - temperature は .env 値があっても 1 を強制（安定最優先）
  - max_completion_tokens は .env 値があっても 1000 を強制
  - 禁止: 画像描写語 / 店舗メタ語 / 競合優位メタ / クリック誘導 等

仕様ポイント:
  - まずAIで 100〜130字・1〜2文・句点終止・自然文（体言止め許容）で20行生成（raw）
  - ローカル整形で 80〜110字へ自然カット、句点補完、禁則再適用、重複/短文/名詞羅列ケア（refined）
  - raw と refined の横並び diff も保存（QA用）
  - 「要（かんなめ）」埋め込み（KANNAME_BANNER）
"""

import os
import re
import csv
import glob
import json
import time
import random
from collections import defaultdict

from dotenv import load_dotenv

# tqdm（無くても動くフォールバック）
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# =========================
# かんなめ（要）
# =========================
KANNAME_BANNER = "【要】KOTOHA-ALT v4.3 / Natural Language Reinforce / Rakuten専用"

# =========================
# OpenAI SDK
# =========================
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# =========================
# 定数
# =========================
INPUT_CSV  = "./rakuten.csv"
OUT_DIR    = "./output/ai_writer"
RAW_PATH   = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.3.csv")
REF_PATH   = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.3.csv")
DIFF_PATH  = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.3.csv")

SEMANTICS_DIR = "./output/semantics"

# まずAIで目指す長さ → ローカルで最終整形
RAW_MIN, RAW_MAX     = 100, 130    # AI 目標
FINAL_MIN, FINAL_MAX = 80, 110     # ローカル整形目標

# 楽天ALT専用 禁則語（画像描写・店舗/誘導メタ・競合メタ 等）
FORBIDDEN_BASE = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "図", "イラスト",
    "当店", "当社", "ショップ", "販売店", "レビュー", "ランキング", "口コミ",
    "クリック", "こちら", "ページ", "リンク", "カート", "購入はこちら", "今すぐ", "限定", "最安",
    "No.1", "ナンバーワン", "世界一", "優位性", "競合", "他社",
    "送料無料（確約）", "返金保証", "割引", "SALE", "セール", "ポイント還元",
]

# 正規表現
LEADING_ENUM_RE  = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
MULTI_COMMA_RE   = re.compile(r"、{3,}")
WHITESPACE_RE    = re.compile(r"\s+")
PARENS_TRIM_RE   = re.compile(r"[（(]\s*[)）]\s*")  # 空括弧消し
LATIN_LISTY_RE   = re.compile(r"[A-Za-z0-9]+(?:\s*[／/・,]\s*[A-Za-z0-9]+){2,}")  # ラテン記号列挙検知

# 名詞羅列の簡易検知（雑だが実用重視）
JAGGED_LISTY_RE  = re.compile(r"(?:[^\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF]{2,}|・|／|/|,){3,}")

# =========================
# 環境 & クライアント
# =========================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
    # .env 推奨: OPENAI_MODEL="gpt-5" / OPENAI_MODE="chat" だが、実行時は固定で安定化
    model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"
    client = OpenAI(api_key=api_key)
    return client, model

# =========================
# 入力（商品名）
# =========================
def load_products_from_csv(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"入力CSVが見つかりません: {path}")
    products = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("入力CSVに『商品名』ヘッダが見つかりません。")
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                products.append(nm)
    # 重複除去（順序維持）
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

# =========================
# 知見（ゆる結合で要約）
# =========================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge_lite():
    """
    ./output/semantics/*.json をゆるく集約 → テキスト化
      - 用語群（clusters, vocabulary 等）
      - 構造（scenes/targets/use_cases 等）
      - テンプレ/骨子（hints/templates）
      - トーン（tone）
      - 禁則（forbidden_words）
    * 形式は様々でも「拾えるだけ拾う」姿勢で堅牢に
    """
    clusters, market, semantics, templates, tones, forbid_local = [], [], [], [], [], []
    if os.path.isdir(SEMANTICS_DIR):
        for p in glob.glob(os.path.join(SEMANTICS_DIR, "*.json")):
            data = safe_load_json(p)
            if data is None:
                continue
            try:
                if isinstance(data, list):
                    # list 型は語彙と見なして回収
                    for v in data:
                        if isinstance(v, dict):
                            if "terms" in v and isinstance(v["terms"], list):
                                clusters.extend([t for t in v["terms"] if isinstance(t, str)])
                            if "vocabulary" in v and isinstance(v["vocabulary"], str):
                                market.append(v["vocabulary"])
                        elif isinstance(v, str):
                            clusters.append(v)
                elif isinstance(data, dict):
                    # clusters
                    arr = data.get("clusters") or data.get("lexical") or []
                    if isinstance(arr, list):
                        for c in arr:
                            if isinstance(c, dict) and isinstance(c.get("terms"), list):
                                clusters.extend([t for t in c["terms"] if isinstance(t, str)])
                    # market vocab
                    mv = data.get("market_vocab") or data.get("market") or []
                    if isinstance(mv, list):
                        for x in mv:
                            if isinstance(x, dict) and isinstance(x.get("vocabulary"), str):
                                market.append(x["vocabulary"])
                            elif isinstance(x, str):
                                market.append(x)
                    elif isinstance(mv, dict):
                        vv = mv.get("vocabulary") or mv.get("vocab") or []
                        if isinstance(vv, list):
                            market.extend([x for x in vv if isinstance(x, str)])
                    # semantics
                    for k in ["concepts", "scenes", "targets", "use_cases"]:
                        arr2 = data.get(k) or []
                        if isinstance(arr2, list):
                            semantics.extend([x for x in arr2 if isinstance(x, str)])
                    # templates
                    for k in ["hints", "templates"]:
                        arr3 = data.get(k) or []
                        if isinstance(arr3, list):
                            templates.extend([x for x in arr3 if isinstance(x, str)])
                    # tones
                    tone = data.get("tone") or {}
                    if isinstance(tone, dict):
                        for v in tone.values():
                            if isinstance(v, str):
                                tones.append(v)
                    # forbidden
                    fw = data.get("forbidden_words") or []
                    if isinstance(fw, list):
                        forbid_local.extend([w for w in fw if isinstance(w, str)])
            except Exception:
                # 形式バラつきは握りつぶして継続
                pass

    # 禁則マージ
    forbidden_all = list(dict.fromkeys(FORBIDDEN_BASE + forbid_local))

    def cap_join(xs, n):
        xs = [x for x in xs if isinstance(x, str) and x.strip()]
        return "、".join(list(dict.fromkeys(xs))[:n])

    cluster_txt = cap_join(clusters, 12)
    market_txt  = cap_join(market,   12)
    sem_txt     = cap_join(semantics, 8)
    tmpl_txt    = cap_join(templates, 3)
    tone_txt    = cap_join(tones,     4)

    kb = "知見: "
    parts = []
    if cluster_txt:
        parts.append(f"語彙:{cluster_txt}")
    if market_txt:
        parts.append(f"市場語:{market_txt}")
    if sem_txt:
        parts.append(f"構造:{sem_txt}")
    if tmpl_txt:
        parts.append(f"骨子:{tmpl_txt}")
    if tone_txt:
        parts.append(f"トーン:{tone_txt}")
    kb += " / ".join(parts) + ("。" if parts else "")
    kb += "画像描写語や販促メタは使わず、自然な日本語の1〜2文で。"
    return kb, forbidden_all

# =========================
# プロンプト
# =========================
SYSTEM_PROMPT = (
    "あなたは楽天市場のSEOに最適化された画像ALTテキストを書くプロの日本語コピーライターです。"
    "各ALT文は自然な日本語の1〜2文で構成します。名詞の羅列は禁止。"
    "基本は「〜です」「〜する」など用言終止ですが、体言止め（名詞で終える）も自然なら許可。"
    "句点「。」で必ず終える。読点「、」は1行につき最大2回まで。"
    "画像や写真の描写語、店舗メタ語、競合優位メタ、クリック誘導は一切使わない。"
    "対応機種・スペック・機能・用途・対象・便益を自然に織り込む。"
    "出力はALT文20行のみ（JSON/番号/記号なし）。"
)

def build_user_prompt(product: str, knowledge_text: str, forbidden_words):
    forbid_txt = "、".join(sorted(set([w for w in forbidden_words if isinstance(w, str)])))
    hint = (
        "構成ヒント（テンプレではなく自然文で）:"
        "『スペック→強み→誰に→どんなシーン→ベネフィット』を1〜2文で自然につなぐ。"
        "名詞列挙は禁止。助詞でつなぎ、体言止めも自然ならOK。"
    )
    return (
        f"{KANNAME_BANNER}\n"
        f"商品名: {product}\n"
        f"{knowledge_text}\n"
        f"{hint}\n"
        f"禁止語: {forbid_txt}\n"
        "出力: ALT文を20行。各行は自然な1〜2文、句点「。」で終える。JSONや番号は不要。"
    )

# =========================
# OpenAI 呼び出し
# =========================
def call_openai_20_lines(client, model, product, knowledge_text, forbidden_words, retry=3, wait=6):
    user_prompt = build_user_prompt(product, knowledge_text, forbidden_words)
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "text"},
                max_completion_tokens=1000,  # 安定固定
                temperature=1,               # 安定固定
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Empty content")
            lines = [LEADING_ENUM_RE.sub("", ln).strip("・-—●　").strip()
                     for ln in content.split("\n") if ln.strip()]
            # 長文行や余分な説明が混ざることがあるので上限60まで保持（後で20抽出）
            return [ln for ln in lines if ln][:60]
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答取得失敗: {last_err}")

# =========================
# ローカル整形（自然文ゲート）
# =========================
TAIGEN_OK_RATIO = 0.35  # 体言止め許容率（最終20本のうちおよそ35%まで）

def is_taigen_stop(s: str) -> bool:
    # 末尾が「です。」「ます。」等でなければ名詞終止の可能性 → 句点は前提
    if not s.endswith("。"):
        return False
    tail = s[:-1].strip()
    # 明示の用言終止を排除
    for yogen in ("です", "ます", "でした", "でした。", "します", "できます", "となります", "になります"):
        if tail.endswith(yogen):
            return False
    # 「〜対応」「〜仕様」「〜設計」「〜構造」などは体言扱い可
    return True

def hard_forbid(text: str, forbids):
    t = text
    for ng in forbids:
        if ng and ng in t:
            t = t.replace(ng, "")
    return t

def normalize_sentence_core(s: str):
    t = s.strip()
    t = PARENS_TRIM_RE.sub("", t)
    t = WHITESPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    t = t.strip("・-—●　")
    # 句点付与
    if not t.endswith("。"):
        t += "。"
    return t

def soft_clip_sentence(s: str, forbids):
    """
    上限120字までを目安に、最後の「。」で自然カット → 禁則再適用
    """
    t = normalize_sentence_core(s)
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
        else:
            t = cut
            if not t.endswith("。"):
                t += "。"
    t = hard_forbid(t, forbids)
    return t.strip()

def looks_like_listy(s: str) -> bool:
    # ラテン記号列挙や記号まみれの羅列を嫌う
    if LATIN_LISTY_RE.search(s):
        return True
    if JAGGED_LISTY_RE.search(s):
        return True
    # 読点が4つ以上 → 羅列っぽい
    if s.count("、") >= 4:
        return True
    return False

def naturalize_short(s: str) -> str:
    """
    短すぎ文（〜20字台）への軽補完。名詞→用言/体言に近づける。
    """
    t = s.strip("。").strip()
    if not t:
        return ""
    # ごく軽い補助句
    addons = [
        "の設計です", "に対応します", "が魅力", "を実現", "をサポート", "に最適", "で安心"
    ]
    t2 = t
    if not t2.endswith(("です", "ます", "最適", "魅力", "設計", "仕様", "対応")):
        t2 = t2 + random.choice(addons)
    return t2 + "。"

def refine_20_lines(raw_lines, forbids):
    """
    1) 正規化・禁則・名詞羅列/短文除去
    2) 120字までで自然カット → 最終80〜110字レンジ狙い
    3) 類似/重複除去
    4) 20本成形（体言止め比率を35%程度に）
    """
    norm = []
    for ln in raw_lines:
        if not ln:
            continue
        ln = LEADING_ENUM_RE.sub("", ln).strip("・-—●　").strip()
        ln = normalize_sentence_core(ln)

        if looks_like_listy(ln):
            # 名詞羅列くさい → 軽補正
            ln = ln.replace("・", "、").replace("/", "、").replace("／", "、")
            ln = re.sub(r"\s*[,、]\s*", "、", ln)
            ln = re.sub(r"(、){3,}", "、、", ln)

        # 短すぎるとき軽補完
        if len(ln) < 25:
            ln = naturalize_short(ln)

        # 120字に柔らかく切って禁則再適用
        ln = soft_clip_sentence(ln, forbids)

        # 最終: 極端短文の捨て
        if len(ln) < 15:
            continue

        norm.append(ln)

    # 重複除去
    uniq, seen = [], set()
    for ln in norm:
        if ln not in seen:
            uniq.append(ln); seen.add(ln)

    # 体言止め比率をざっくり制御（多すぎる場合は一部を用言化）
    taigen = [i for i, s in enumerate(uniq) if is_taigen_stop(s)]
    limit = max(0, int(len(uniq) * TAIGEN_OK_RATIO))
    if len(taigen) > limit:
        over = taigen[limit:]
        for i in over:
            s = uniq[i]
            s = s[:-1] + "です。"
            uniq[i] = s

    # 80〜110字帯へ寄せる（長すぎは末尾句点まで詰め、短すぎは軽補完）
    refined = []
    for s in uniq:
        t = s
        if len(t) > FINAL_MAX:
            cut = t[:FINAL_MAX]
            p = cut.rfind("。")
            if p != -1 and p >= FINAL_MIN:
                t = cut[:p+1]
        if len(t) < FINAL_MIN:
            t = naturalize_short(t)
        # 読点密度（最大2回）を超える場合、不要な読点を1つ落とす
        while t.count("、") > 2:
            t = t.replace("、", "", 1)
        refined.append(t)

    # 20本に整形（足りない時は軽い言い換えで補完）
    def light_variation(s: str) -> str:
        v = s
        # 軽い語尾バリエーション
        repls = [("します。", "できます。"),
                 ("できます。", "しやすいです。"),
                 ("です。", "になります。")]
        for a, b in repls:
            if v.endswith(a):
                v = v[:-len(a)] + b
                break
        if v == s:
            # 先頭の一単語の後ろに読点（重複は抑制）
            v = re.sub(r"^(\S{2,5})", r"\1、", v, count=1)
            v = v.replace("、、", "、")
            if not v.endswith("。"):
                v += "。"
        return soft_clip_sentence(v, forbids)

    i = 0
    while len(refined) < 20 and refined:
        refined.append(light_variation(refined[i % len(refined)]))
        i += 1

    return refined[:20]

# =========================
# 書き出し
# =========================
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
            r_line   = (r[:20]   + [""] * max(0, 20 - len(r)))
            ref_line = (ref[:20] + [""] * max(0, 20 - len(ref)))
            w.writerow([p] + r_line + ref_line)

# =========================
# メイン
# =========================
def main():
    print("🌸 ALT長文 v4.3（自然文＋体言止め許容＋知見融合＋raw/refined/diff）")
    print(KANNAME_BANNER)
    client, model = init_env_and_client()
    ensure_outdir()

    products = load_products_from_csv(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    knowledge_text, forbidden_all = summarize_knowledge_lite()

    all_raw, all_refined = [], []
    for p in tqdm(products, desc="🧠 AI生成中", total=len(products)):
        # 1) AI生成
        try:
            raw_lines = call_openai_20_lines(client, model, p, knowledge_text, forbidden_all)
        except Exception:
            # ダミー（空は避ける）
            raw_lines = [f"{p} の使い勝手を高め、日常の小さな不便を解消する設計です。"] * 20

        # 2) ローカル整形
        refined_lines = refine_20_lines(raw_lines, forbidden_all)

        all_raw.append(raw_lines[:20])
        all_refined.append(refined_lines)

        time.sleep(0.2)  # スロットリング

    # 3) 書き出し
    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    # 4) 検証ログ
    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens)))

    # 自然文率（句点終止かつ羅列っぽくない）
    def natural_rate(blocks):
        total = 0
        ok = 0
        for lines in blocks:
            for s in lines:
                if not s:
                    continue
                total += 1
                if s.endswith("。") and not looks_like_listy(s):
                    ok += 1
        return (ok / total * 100) if total else 0.0

    # 体言止め率
    def taigen_rate(blocks):
        total = 0
        tg = 0
        for lines in blocks:
            for s in lines:
                if not s:
                    continue
                total += 1
                if is_taigen_stop(s):
                    tg += 1
        return (tg / total * 100) if total else 0.0

    print("✅ 出力完了:")
    print(f"   - AI生出力 : {RAW_PATH}")
    print(f"   - 整形後    : {REF_PATH}")
    print(f"   - 差分比較  : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print(f"💬 自然文率   : {natural_rate(all_refined):.1f}%")
    print(f"◎ 体言止め率 : {taigen_rate(all_refined):.1f}%（目標 ~{int(TAIGEN_OK_RATIO*100)}%）")

if __name__ == "__main__":
    main()
import atlas_autosave_core
