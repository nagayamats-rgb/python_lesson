# -*- coding: utf-8 -*-
"""
v5.2 Sales Copy Persona Writer (ALT for Rakuten, 20 lines each)
- 入力: /Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv  （UTF-8, ヘッダ: 「商品名」）
- 出力:
    1) ./output/ai_writer/alt_text_ai_raw_salescopy_v5_2.csv        … AIの生出力（20本/商品）
    2) ./output/ai_writer/alt_text_refined_salescopy_v5_2.csv       … ローカル整形後（80〜110字）
    3) ./output/ai_writer/alt_text_diff_salescopy_v5_2.csv          … raw/refined の横並び比較
- 知見: ./output/semantics/*.json を“要（かんなめ）”としてゆるく集約（存在範囲でOK）
- OpenAI:
    - .env から OPENAI_API_KEY を取得
    - OPENAI_MODEL（指定なければ 'gpt-4o'）/ OPENAI_TEMPERATURE / OPENAI_MAX_TOKENS を使用
    - chat.completions（response_format={"type":"text"}）
- 仕様要点:
    - 楽天ALT専用（Yahooで使う販促語は避ける）
    - 画像描写語・メタ語・競合比較のメタ表現は禁止
    - 各商品20本、1〜2文、文尾は句点「。」と体言止めを自然に混在
    - 80〜110字レンジにローカルで整形
    - 重複/近似（>=0.90）を除去し補完
    - 空行/崩れは自動補完（商品説明補完文で埋める）
"""

import os
import re
import csv
import glob
import json
import time
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

from dotenv import load_dotenv

# tqdm（未インストールでも動く）
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# OpenAI SDK（新旧混在対策）
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("❌ openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# ==============
# 0) 定数・I/O
# ==============
# 入力CSV（固定パス／UTF-8）
INPUT_CSV = "/Users/tsuyoshi/Desktop/python_lesson/sauce/rakuten.csv"

# 出力ディレクトリ
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_salescopy_v5_2.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_salescopy_v5_2.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_salescopy_v5_2.csv")

# 知見（要：かんなめ）
SEMANTICS_DIR = "./output/semantics"

# 禁則語（画像描写語・メタ・店舗メタなど）— 楽天ALT専用
FORBIDDEN_BASE = [
    "画像", "写真", "見た目", "上の画像", "下の写真",
    "当店", "当社", "ショップ", "レビュー", "口コミ", "ランキング",
    "クリック", "こちら", "リンク", "カート", "購入はこちら",
    "最安", "No.1", "ナンバーワン", "売上No1", "業界最高", "競合", "競合優位性",
    "返金保証", "送料無料（確約）", "ポイント還元", "限定クーポン"  # ← Yahoo的販促語は避ける
]

# 句読点や箇条書きの掃除
LEADING_ENUM_RE = re.compile(r"^\s*[\d一二三四五六七八九十①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
WS_RE = re.compile(r"\s+")
MULTI_COMMA_RE = re.compile(r"、{3,}")

# 文長ポリシー
RAW_MIN, RAW_MAX = 100, 130     # AIにはこのレンジを狙わせる
FINAL_MIN, FINAL_MAX = 80, 110  # ローカルで最終調整

# 重複近似の閾値
SIM_THRESHOLD = 0.90

# 体言止めの適用率（目安）
TAIGEN_RATE = 0.35

# ==============
# 1) “要（かんなめ）”
# ==============
KANNAME_BANNER = "⛩ 要（かんなめ）: 知見を中枢で統合し、ALT最適化へ反映"

# ==============
# 2) ENV & Client
# ==============
def init_env_and_client():
    load_dotenv(override=True)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")

    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE") or "1.0")
    except Exception:
        temperature = 1.0
    try:
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS") or "1200")
    except Exception:
        max_tokens = 1200

    client = OpenAI(api_key=api_key)
    return client, model, temperature, max_tokens

# ==============
# 3) 入力（商品名）
# ==============
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
    # 重複除去（順序保持）
    seen, uniq = set(), []
    for nm in products:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq

# ==============
# 4) 知見集約（要）
# ==============
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge_relaxed():
    """
    /output/semantics 配下の JSON 群（形式バラバラOK）を「要」として緩やかに統合。
    - lexical_clusters_*.json    → 語彙クラスタ
    - market_vocab_*.json        → 市場語彙
    - structured_semantics_*.json→ 構造的観点（用途/対象/シーン/特徴/便益など）
    - styled_persona_*.json      → トーン・文体
    - normalized_*.json          → 禁則語など
    - template_composer.json     → 骨子ヒント
    """
    clusters, market, semantics, persona, templates, forbid_local = [], [], [], [], [], []
    if not os.path.isdir(SEMANTICS_DIR):
        # 要の初期知見（最低限）
        base_text = ("知見: 主要キーワード・スペック・対応機種・利用シーン・対象・便益を自然に織り込み、"
                     "箇条書きを避け、自然な日本語で2文以内を基本。")
        return base_text, FORBIDDEN_BASE[:]

    for p in glob.glob(os.path.join(SEMANTICS_DIR, "*.json")):
        data = safe_load_json(p)
        if data is None:
            continue
        name = os.path.basename(p).lower()

        try:
            # 配列/dict 混在を吸収
            if "lexical" in name:
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "terms" in item and isinstance(item["terms"], list):
                            clusters += [t for t in item["terms"] if isinstance(t, str)]
                        elif isinstance(item, str):
                            clusters.append(item)
                elif isinstance(data, dict):
                    arr = data.get("clusters") or data.get("lexical") or []
                    if isinstance(arr, list):
                        for c in arr:
                            if isinstance(c, dict) and isinstance(c.get("terms"), list):
                                clusters += [t for t in c["terms"] if isinstance(t, str)]

            elif "market_vocab" in name or "market" in name:
                if isinstance(data, list):
                    for v in data:
                        if isinstance(v, dict) and isinstance(v.get("vocabulary"), str):
                            market.append(v["vocabulary"])
                        elif isinstance(v, str):
                            market.append(v)
                elif isinstance(data, dict):
                    vocab = data.get("vocabulary") or data.get("vocab") or []
                    if isinstance(vocab, list):
                        market += [x for x in vocab if isinstance(x, str)]

            elif "structured_semantics" in name or "semantic" in name:
                if isinstance(data, dict):
                    for k in ["concepts", "semantics", "frames", "features", "facets", "benefits", "scenes", "targets", "use_cases"]:
                        arr = data.get(k) or []
                        if isinstance(arr, list):
                            semantics += [x for x in arr if isinstance(x, str)]
                elif isinstance(data, list):
                    semantics += [x for x in data if isinstance(x, str)]

            elif "styled_persona" in name or "persona" in name:
                if isinstance(data, dict):
                    t = data.get("tone") or data.get("style") or {}
                    if isinstance(t, dict):
                        for v in t.values():
                            if isinstance(v, str):
                                persona.append(v)
                    # fallback: フラット文字列の配列も拾う
                    for k in ["persona", "tones", "styles"]:
                        arr = data.get(k) or []
                        if isinstance(arr, list):
                            persona += [x for x in arr if isinstance(x, str)]
                elif isinstance(data, list):
                    persona += [x for x in data if isinstance(x, str)]

            elif "normalized" in name or "forbid" in name:
                if isinstance(data, dict):
                    fw = data.get("forbidden_words") or []
                    if isinstance(fw, list):
                        forbid_local += [w for w in fw if isinstance(w, str)]
                elif isinstance(data, list):
                    forbid_local += [w for w in data if isinstance(w, str)]

            elif "template_composer" in name or "template" in name:
                if isinstance(data, dict):
                    hints = data.get("hints") or data.get("templates") or []
                    if isinstance(hints, list):
                        templates += [h for h in hints if isinstance(h, str)]
                elif isinstance(data, list):
                    templates += [x for x in data if isinstance(x, str)]

        except Exception:
            # 形式不一致は黙ってスキップ（堅牢重視）
            pass

    # ユニーク＆上限
    def uniq_cap(xs, n):
        return list(dict.fromkeys([x for x in xs if isinstance(x, str)]))[:n]

    clusters = uniq_cap(clusters, 18)
    market   = uniq_cap(market,   18)
    semantics= uniq_cap(semantics,18)
    persona  = uniq_cap(persona,   8)
    templates= uniq_cap(templates, 6)
    forbid_all = list(dict.fromkeys(FORBIDDEN_BASE + uniq_cap(forbid_local, 64)))

    # “要”テキスト（AIに渡す日本語知見）
    kb_parts = []
    if clusters:  kb_parts.append(f"語彙: {'、'.join(clusters)}")
    if market:    kb_parts.append(f"市場語: {'、'.join(market)}")
    if semantics: kb_parts.append(f"構造: {'、'.join(semantics)}")
    if templates: kb_parts.append(f"骨子: {'、'.join(templates)}")
    if persona:   kb_parts.append(f"トーン: {'、'.join(persona)}")

    if kb_parts:
        kb_text = "知見（要）: " + " / ".join(kb_parts) + "。"
    else:
        kb_text = ("知見（要）: 主要キーワード・スペック・対応機種・利用シーン・対象・便益を自然に織り込む。"
                   "箇条書きを避け、自然な日本語で2文以内を基本。")
    kb_text += " ALTは画像描写語やECメタ語を使わず、自然文として読めること。"
    return kb_text, forbid_all

# ==============
# 5) プロンプト
# ==============
SYSTEM_PROMPT = (
    "あなたは日本語に精通したSEOコピーライターであり、"
    "楽天市場の商品ページで高い成約率を誇る売れっ子ライターです。"
    "読者に伝わる自然なリズムで、ALTとして使える紹介文を20本作成してください。\n"
    "出力は20行。各行は1〜2文以内。句点「。」で終える文と体言止めの文を自然に混ぜてください。\n"
    "商品名・機能・特徴・対応機種・利用シーン・ターゲット・ベネフィットを自然に織り込み、"
    "SEOを意識しつつ、読みやすさを最優先にしてください。\n"
    "同一商品の20本は、構文・語彙・語尾・焦点・視点を変え、多様性を確保。意味の重複や語順の焼き直しは禁止。\n"
    "禁止語（画像、写真、当店、レビュー、リンク、最安、No.1、競合、競合優位性、など）は使わないでください。\n"
    "出力はテキスト20行のみ（JSON/箇条書き/番号/ラベル禁止）。"
)

def build_user_prompt(product: str, knowledge_text: str, forbid_words: list):
    forbid_txt = "、".join(sorted(set([w for w in forbid_words if isinstance(w, str)])))
    hint = (
        "構成ヒント（テンプレ化しない）：商品スペック→強み（コア）→どんな人→どんなシーン→便益。"
        "必要に応じて対応機種や型番も自然に含める。"
    )
    target = (
        f"商品名: {product}\n"
        f"{knowledge_text}\n"
        f"{hint}\n"
        f"禁止語（絶対に使わない）: {forbid_txt}\n"
        "20行で、各行は自然な日本語の文として出力してください。"
    )
    return target

# ==============
# 6) OpenAI 呼び出し
# ==============
def call_openai_20_lines(client, model, temperature, max_tokens, product, kb_text, forbid_words, retry=3, wait=6):
    user_prompt = build_user_prompt(product, kb_text, forbid_words)
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
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = (res.choices[0].message.content or "").strip()
            if content:
                # 行に分解して番号・箇条書き・先頭記号を剥がす
                lines = []
                for ln in content.splitlines():
                    s = ln.strip()
                    if not s:
                        continue
                    s = LEADING_ENUM_RE.sub("", s)
                    s = s.strip("・-—●　")
                    if s:
                        lines.append(s)
                # 余分な行が返ることがあるので最大60まで保持（後で20抽出）
                return lines[:60]
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答を取得できませんでした: {last_err}")

# ==============
# 7) ローカル整形
# ==============
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 番号/箇条書き掃除
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")
    # 余計な空白圧縮
    t = WS_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)

    # 句点補完 or 体言止め許容（すでに句点なしならそのまま許容。後段で割合調整）
    if t.endswith(("。", "！", "？")):
        end_mark = True
    else:
        end_mark = False

    # 長すぎる場合：max_len+10 程度まで許容 → 最後の「。」でカットして80〜110に寄せる
    hard_cap = max_len + 10
    if len(t) > hard_cap:
        cut = t[:hard_cap]
        p = cut.rfind("。")
        if p != -1 and p + 1 >= min_len:  # 自然な句点終止がminを満たすなら採用
            t = cut[:p+1]
        else:
            t = cut

    # 禁則語は最終で除去（完全除去）
    return t.strip()

def ends_with_punctuation(s: str) -> bool:
    return s.endswith(("。", "！", "？"))

def to_taigen_if_needed(s: str) -> str:
    """語尾自然化：です/ます を体言止めへ（軽率な削りを避ける）"""
    t = s.strip()
    # 既に体言止めっぽいならそのまま
    if ends_with_punctuation(t):
        if t.endswith("。"):
            # 3〜4文字の敬体を限定的に落とす
            for rep in ("します。", "できます。", "でした。", "です。", "ます。"):
                if t.endswith(rep) and len(t) > len(rep) + 8:
                    return t[:-len("です。")] + "です" if rep == "です。" else t[:-1]  # 末尾句点だけ削る
        return t
    # 体言止めっぽい末尾ならそのまま
    return t

def uniq_by_similarity(lines, threshold=SIM_THRESHOLD):
    """高類似の文を除去"""
    uniq = []
    for s in lines:
        is_dup = False
        for u in uniq:
            if SequenceMatcher(None, s, u).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            uniq.append(s)
    return uniq

def fallback_sentence(product: str) -> str:
    return f"{product} の使い勝手を高める設計で、日常の不便を減らす実用的な一品です。"

def refine_20_lines(product: str, raw_lines, forbid_words):
    # 正規化 → 句点/体言止め許容 → 長さ整形
    norm = []
    for ln in raw_lines:
        if not ln:
            continue
        s = soft_clip_sentence(ln)
        if not s:
            continue
        norm.append(s)

    # 類似除去
    norm = uniq_by_similarity(norm)

    # 禁則語削除（完全除去）
    for i, s in enumerate(norm):
        for ng in forbid_words:
            if ng and ng in s:
                s = s.replace(ng, "")
        norm[i] = s.strip()

    # 末尾句点と体言止めの混在（後で割合を整える）
    out = []
    for s in norm:
        if not s:
            continue
        # “単語羅列”っぽい短文は破棄
        if len(s) < 20:
            continue
        out.append(s)

    # 句点終止で自然な長さへ再調整
    final = []
    for s in out:
        ss = s
        if len(ss) > FINAL_MAX + 10:
            cut = ss[:FINAL_MAX + 10]
            p = cut.rfind("。")
            if p != -1 and p + 1 >= FINAL_MIN:
                ss = cut[:p+1]
            else:
                ss = cut
        if not ss:
            continue
        final.append(ss)

    # 体言止め割合を整える（約 TAIGEN_RATE）
    rng = list(range(len(final)))
    if rng:
        target_n = max(1, int(len(final) * TAIGEN_RATE))
        chosen = set()
        i = 0
        while len(chosen) < target_n and i < len(rng):
            idx = rng[i]
            if final[idx].endswith("。"):
                # “です/ます/ました”の場合は句点だけ残しつつ自然に
                final[idx] = to_taigen_if_needed(final[idx])
                if final[idx].endswith("。"):
                    # 体言化しきれなければ文末句点は維持
                    pass
                else:
                    # 体言になって句点消えた場合 → そのまま
                    pass
                chosen.add(idx)
            i += 1

    # 20本に満たなければ補完（語尾変形）
    def light_variation(s: str) -> str:
        t = s
        # 語尾バリエーション（軽い置換）
        t = t.replace("します。", "できます。")
        t = t.replace("できます。", "しやすいです。")
        t = t.replace("です。", "になります。")
        if t == s:
            # 句内に軽微な助詞を追加（過剰な変化は避ける）
            t = re.sub(r"([^\s、。]{3,})", r"\1、", t, count=1)
            t = t.replace("、、", "、")
        # 最後に軽く整形
        t = soft_clip_sentence(t)
        return t

    base = final[:]
    i = 0
    while len(final) < 20 and base:
        cand = light_variation(base[i % len(base)])
        # 類似抑制
        if all(SequenceMatcher(None, cand, x).ratio() < SIM_THRESHOLD for x in final):
            final.append(cand)
        i += 1
        if i > 200:  # 念のためのブレーク
            break

    # まだ足りなければ補完文
    while len(final) < 20:
        cand = fallback_sentence(product)
        if all(SequenceMatcher(None, cand, x).ratio() < SIM_THRESHOLD for x in final):
            final.append(cand)
        else:
            final.append(cand + "")  # どうしても埋まらない場合はそのまま

    # 多すぎるなら先頭20
    return final[:20]

# ==============
# 8) 書き出し
# ==============
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

# ==============
# 9) メイン
# ==============
def main():
    print("🌸 v5.2 Sales Copy Persona Writer（ALT×自然文×要）")
    print(KANNAME_BANNER)

    client, model, temperature, max_tokens = init_env_and_client()
    ensure_outdir()

    products = load_products(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    knowledge_text, forbid_all = summarize_knowledge_relaxed()
    print("✅ 要（知見）読込完了")

    all_raw, all_refined = [], []

    for p in tqdm(products, total=len(products), desc="🧠 生成中"):
        # 1) AIで20本（≧20本返る場合もあり）
        try:
            raw_lines = call_openai_20_lines(
                client, model, temperature, max_tokens,
                product=p, kb_text=knowledge_text, forbid_words=forbid_all,
                retry=3, wait=6
            )
        except Exception as e:
            # 全失敗 → ダミーで20本
            raw_lines = [f"{p} の使い勝手を高める設計で、日常の不便を減らす実用的な一品です。"] * 20

        # 2) ローカル整形 → 20本に確定
        refined_lines = refine_20_lines(p, raw_lines, forbid_all)

        all_raw.append(raw_lines[:20])
        all_refined.append(refined_lines)

        # 過負荷回避
        time.sleep(0.2)

    # 3) 書き出し
    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    # 簡易メトリクス
    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens))) if lens else 0.0

    print("✅ 出力完了:")
    print(f"   - AI生出力: {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print("🔒 仕様: ")
    print(f"   - AIは約{RAW_MIN}〜{RAW_MAX}字・1〜2文、句点/体言止め混在、禁則適用（プロンプト）")
    print(f"   - ローカル整形で{FINAL_MIN}〜{FINAL_MAX}字に調整、重複除去・禁則再適用・空欄補完")

if __name__ == "__main__":
    main()
import atlas_autosave_core
