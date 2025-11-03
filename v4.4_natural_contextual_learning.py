# -*- coding: utf-8 -*-
"""
ALT長文（自然文学習＋知見コンテキスト） v4.4
- 入力: ./rakuten.csv （UTF-8, ヘッダに「商品名」）
- 出力:
  1) output/ai_writer/alt_text_ai_raw_longform_v4.4.csv
  2) output/ai_writer/alt_text_refined_final_longform_v4.4.csv
  3) output/ai_writer/alt_text_diff_longform_v4.4.csv
- 知見: ./output/semantics 配下のJSON群を“背景知識”として assistant メッセージで注入
- OpenAI: .env で指定
    OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-5.1 (例) ※未指定時は gpt-4o
    OPENAI_TEMPERATURE=1.0
    OPENAI_MAX_TOKENS=1000
"""

import os
import re
import csv
import glob
import json
import time
from collections import defaultdict

from dotenv import load_dotenv

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# ====== 設定 ======
INPUT_CSV = "./rakuten.csv"
OUT_DIR   = "./output/ai_writer"
RAW_PATH  = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.4.csv")
REF_PATH  = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.4.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.4.csv")

SEMANTICS_DIR = "./output/semantics"

# 禁則語（画像描写語・メタ語）
FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真",
    "当店", "当社", "レビュー", "ランキング", "クリック", "こちら",
    "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
    "リンク", "ページ", "カート", "購入はこちら", "送料無料（確約）", "返金保証",
]

# 文字数方針
RAW_MIN, RAW_MAX     = 100, 130
FINAL_MIN, FINAL_MAX =  80, 110

# 正規表現
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
MULTI_COMMA_RE  = re.compile(r"、{3,}")
WS_RE           = re.compile(r"\s+")

# ====== 環境とクライアント ======
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o"
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "1").strip())
    except:
        temperature = 1.0
    try:
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000").strip())
    except:
        max_tokens = 1000

    client = OpenAI(api_key=api_key)
    return client, model, temperature, max_tokens

# ====== 入力（商品名） ======
def load_products(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"入力CSVが見つかりません: {path}")
    items = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "商品名" not in r.fieldnames:
            raise SystemExit("入力CSVに『商品名』ヘッダが見つかりません。")
        for row in r:
            nm = (row.get("商品名") or "").strip()
            if nm: items.append(nm)
    # 重複除去（順序維持）
    seen, uniq = set(), []
    for nm in items:
        if nm not in seen:
            uniq.append(nm); seen.add(nm)
    return uniq

# ====== 知見 読み込み・要約 ======
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge_structured():
    """
    ./output/semantics 内のJSONをゆるく集約し、構造化ペイロードとテキスト要約を返す
    """
    payload = {
        "clusters": [],      # 用語クラスタ
        "market_vocab": [],  # 市場系語彙
        "concepts": [],      # 概念/用途/対象/シーン
        "templates": [],     # 表現骨子
        "tone": {},          # トーン
        "forbidden_local": []
    }
    if not os.path.isdir(SEMANTICS_DIR):
        text = "主要キーワード・用途・対象・スペック・関連機種名を自然に含める。画像描写語やECメタ語は使わない。"
        return payload, text, FORBIDDEN[:]

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    for p in files:
        name = os.path.basename(p).lower()
        data = safe_load_json(p)
        if data is None: 
            continue

        # 配列/辞書 混在を吸収
        def listify(x):
            return x if isinstance(x, list) else ([x] if x else [])

        if "lexical" in name or "cluster" in name:
            # 例: {"clusters":[{"terms":[...]}]} / [{"terms":[...]}] / ["MagSafe", ...]
            for item in listify(data.get("clusters") if isinstance(data, dict) else data):
                if isinstance(item, dict) and isinstance(item.get("terms"), list):
                    payload["clusters"].extend([t for t in item["terms"] if isinstance(t, str)])
                elif isinstance(item, str):
                    payload["clusters"].append(item)

        elif "market" in name and "vocab" in name or "market_vocab" in name:
            # 例: [{"vocabulary":"MagSafe"}, "PD", ...] / {"vocabulary":[...]}
            if isinstance(data, dict) and isinstance(data.get("vocabulary"), list):
                payload["market_vocab"].extend([x for x in data["vocabulary"] if isinstance(x, str)])
            elif isinstance(data, list):
                for v in data:
                    if isinstance(v, dict) and isinstance(v.get("vocabulary"), str):
                        payload["market_vocab"].append(v["vocabulary"])
                    elif isinstance(v, str):
                        payload["market_vocab"].append(v)

        elif "structured_semantics" in name or "semantic" in name:
            # 例: {"concepts":[...], "targets":[...], "use_cases":[...], "scenes":[...]}
            if isinstance(data, dict):
                for k in ["concepts", "targets", "use_cases", "scenes"]:
                    payload["concepts"] += [x for x in (data.get(k) or []) if isinstance(x, str)]
            elif isinstance(data, list):
                payload["concepts"] += [x for x in data if isinstance(x, str)]

        elif "template_composer" in name or "template" in name:
            # 例: {"hints":[...]} / {"templates":[...]} / [...]
            if isinstance(data, dict):
                payload["templates"] += [x for x in (data.get("hints") or []) if isinstance(x, str)]
                payload["templates"] += [x for x in (data.get("templates") or []) if isinstance(x, str)]
            elif isinstance(data, list):
                payload["templates"] += [x for x in data if isinstance(x, str)]

        elif "persona" in name or "styled_persona" in name or "tone" in name:
            # 例: {"tone":{"style":"〜","register":"〜"}}
            if isinstance(data, dict):
                t = data.get("tone") or {}
                if isinstance(t, dict):
                    for k, v in t.items():
                        if isinstance(v, str):
                            payload["tone"][k] = v

        elif "normalized" in name or "forbid" in name:
            # 例: {"forbidden_words":[...]}
            if isinstance(data, dict):
                payload["forbidden_local"] += [w for w in (data.get("forbidden_words") or []) if isinstance(w, str)]
            elif isinstance(data, list):
                payload["forbidden_local"] += [w for w in data if isinstance(w, str)]

    # テキスト要約
    def cap(xs, n):
        xs = [x for x in xs if isinstance(x, str)]
        return "、".join(list(dict.fromkeys(xs))[:n])

    parts = []
    c = cap(payload["clusters"], 12)
    m = cap(payload["market_vocab"], 12)
    z = cap(payload["concepts"], 8)
    t = cap(payload["templates"], 3)
    if c: parts.append(f"語彙: {c}")
    if m: parts.append(f"市場語: {m}")
    if z: parts.append(f"構造: {z}")
    if t: parts.append(f"骨子: {t}")
    text = " / ".join(parts) + ("。" if parts else "")
    text += "自然で読みやすい日本語で、過剰な詰め込みは避ける。"
    forbid_all = list({*FORBIDDEN, *payload["forbidden_local"]})
    return payload, text, forbid_all

# ====== プロンプト ======
SYSTEM_PROMPT = (
    "あなたはEC画像のALTテキストを書く日本語コピーライターです。"
    "目的は、楽天のサイト内SEOに強い、自然な日本語のALTを20本です。"
    "必須ルール：\n"
    f"・各ALTは全角{RAW_MIN}〜{RAW_MAX}字を目安に1〜2文。必ず句点「。」で終える。\n"
    "・画像や写真の描写語（例：画像、写真、見た目）やECメタ語（当店、レビュー、ランキング等）は使わない。\n"
    "・競合比較や「競合優位性」のようなメタ表現は禁止。\n"
    "・対応機種・スペック・機能・用途・対象・便益のキーワードは、不自然に詰め込まず“自然に”含める。\n"
    "・箇条書き・番号・ラベル（ALT: 等）は付けない。\n"
    "・出力は20行のテキストのみ。JSONや記号は不要。\n"
    "\n"
    "▼悪い例（禁止）：\n"
    "「耐久性・防水・軽量・シンプルデザイン・使いやすい仕様。」（名詞の羅列）\n"
    "「高速充電、急速充電、PD、USB-C、iPhone、Android。」（読点でのキーワード並べ）\n"
    "\n"
    "▼良い例（模倣）：\n"
    "「MagSafe対応の薄型充電器。軽量かつ安定した吸着で、デスクでも就寝時でも扱いやすい設計です。」\n"
    "「9H硬度のガラスフィルムで擦り傷に強く、指すべりも滑らか。貼り付け補助枠が付属し、誰でも簡単に貼れます。」\n"
    "「Type-C 240Wケーブル。ノートPCのPD高速充電に対応し、耐屈曲メッシュで持ち運びにも安心です。」\n"
)

def build_user_prompt(product: str, forbid_words):
    forbid_txt = "、".join(sorted(set([w for w in forbid_words if isinstance(w, str)])))
    hint = (
        "自然文ヒント（テンプレではなく自然に）："
        "商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット。"
    )
    return (
        f"商品名: {product}\n"
        f"{hint}\n"
        f"禁止語（絶対に使わない）: {forbid_txt}\n"
        "20行で、各行は助詞で自然につないだ1〜2文の日本語にしてください。"
    )

# ====== OpenAI 呼び出し ======
def call_openai_20_lines(client, model, temperature, max_tokens,
                         product, knowledge_text, forbid_words, retry=3, wait=6):
    """
    20行以上返ってくることがあるので後段で整形。知見は assistant メッセージで供給。
    """
    user_prompt = build_user_prompt(product, forbid_words)
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "以下はこの商品の背景知識です。参考にしてください。"},
                    {"role": "assistant", "content": knowledge_text},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "text"},
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = (res.choices[0].message.content or "").strip()
            if content:
                # 行分割
                lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
                # 箇条書き番号の剥離
                cleaned = []
                for ln in lines:
                    ln2 = LEADING_ENUM_RE.sub("", ln).strip("・-—●　")
                    if ln2: cleaned.append(ln2)
                return cleaned[:80]  # 念のため多めに保持
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答失敗: {last_err}")

# ====== ローカル整形 ======
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not t.endswith("。"):
        t += "。"
    t = WS_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")

    # 長すぎる場合は120まで許容し、最後の「。」で切る
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
        else:
            t = cut

    # 禁則語は完全除去
    for ng in FORBIDDEN:
        if ng and ng in t:
            t = t.replace(ng, "")
    return t.strip()

def refine_20_lines(raw_lines):
    # 正規化
    norm = []
    for ln in raw_lines:
        if not ln: continue
        ln = LEADING_ENUM_RE.sub("", ln).strip("・-—●　")
        ln = soft_clip_sentence(ln)
        if len(ln) < 18:  # 異常に短いものは棄却（後で補完）
            continue
        norm.append(ln)

    # 重複除去
    uniq, seen = [], set()
    for ln in norm:
        if ln not in seen:
            uniq.append(ln); seen.add(ln)

    refined = [soft_clip_sentence(ln) for ln in uniq]

    # 足りない場合の補完（テンプレ最小文・自然文）
    def synth(product):
        core = f"{product} の使い勝手を高める設計で、日常の不便を軽減します。"
        return soft_clip_sentence(core)

    i = 0
    while len(refined) < 20:
        seed = refined[i % len(refined)] if refined else ""
        add  = synth("本製品") if not seed else seed.replace("です。", "になります。")
        if not add.endswith("。"): add += "。"
        refined.append(soft_clip_sentence(add))
        i += 1

    return refined[:20]

# ====== 書き出し ======
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

# ====== メイン ======
def main():
    print("🌸 ALT長文生成 v4.4（自然文学習＋知見コンテキスト）")
    client, model, temperature, max_tokens = init_env_and_client()
    ensure_outdir()

    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    kb_payload, kb_text, forbidden_all = summarize_knowledge_structured()

    all_raw, all_refined = [], []
    for p in tqdm(products, desc="🧠 AI生成", total=len(products)):
        try:
            raw_lines = call_openai_20_lines(
                client, model, temperature, max_tokens,
                p, kb_text, forbidden_all
            )
        except Exception:
            # 完全失敗時のフェイルセーフ（20本ダミー）
            raw_lines = [f"{p} は使いやすさと耐久性を両立し、日常の不便を減らします。"] * 20

        refined = refine_20_lines(raw_lines)
        all_raw.append(raw_lines[:20])
        all_refined.append(refined)
        time.sleep(0.2)  # スロットリング

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
    print("🔒 仕様: 良い例/悪い例のfew-shot誘導＋assistant知見注入・禁則/句点/箇条書き剥がし・欠損自動補完")

if __name__ == "__main__":
    main()
import atlas_autosave_core
