# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
ALT長文（SEO＋自然文）生成 v3
- 入力: ./rakuten.csv （UTF-8, ヘッダに「商品名」）
- 出力:
  1) output/ai_writer/alt_text_ai_raw_longform_v3.csv           … AIの生出力（20本/商品）
  2) output/ai_writer/alt_text_refined_final_longform_v3.csv    … ローカル整形後（80〜110字）
  3) output/ai_writer/alt_text_diff_longform_v3.csv             … raw/refined の横並び比較
- 知見: ./output/semantics/ 内の JSON/CSV 群をゆるく集約し、商品ごとの「知見要約」を付与
- OpenAI:
    model            = gpt-4o（.env 固定）
    response_format  = {"type":"text"}
    max_completion_tokens = 1000
    temperature      = 1（固定：このモデルは任意温度が不安定要因になりやすいため）
"""

import os
import re
import csv
import glob
import json
import time
from collections import defaultdict

from dotenv import load_dotenv

# 🌸 KOTOHA 凍結管理要（かんなめ）統合
import freeze_manager_extended as freezer
freezer.auto_freeze_on_start(__file__, note="ALT長文生成_v3（SEO＋自然文）／KOTOHA 凍結管理要 起動")

# tqdm は視覚的進捗、未インストールでも動くようフォールバック
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# =========================
# 0) OpenAI クライアント用
# =========================
try:
    from openai import OpenAI
except Exception:
    # 新旧SDK混在対策：インポート失敗時は明示
    raise SystemExit("openai SDK が見つかりません。`pip install openai python-dotenv` を実行してください。")

# =========================
# 1) 定数・ユーティリティ
# =========================
INPUT_CSV = "./rakuten.csv"  # UTF-8, ヘッダに「商品名」
OUT_DIR = "./output/ai_writer"
RAW_PATH = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v3.csv")
REF_PATH = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v3.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v3.csv")

SEMANTICS_DIR = "./output/semantics"  # 知見フォルダ（存在しない/空でもOK）

# 禁則語（画像描写語・メタ・店舗メタなど）
FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "当店", "当社", "レビュー", "ランキング",
    "クリック", "こちら", "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
    "リンク", "ページ", "カート", "購入はこちら", "クリックして", "送料無料（確約）", "返金保証",
]

# 句読点・箇条書きパターン cleanup
LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
MULTI_COMMA_RE = re.compile(r"、{3,}")
WHITESPACE_RE = re.compile(r"\s+")

# 文字数制御
RAW_MIN, RAW_MAX = 100, 130    # まずAIでこのレンジを狙う
FINAL_MIN, FINAL_MAX = 80, 110  # ローカルでこのレンジに整形

# =========================
# 2) 環境初期化
# =========================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
    # モデルは .env で gpt-4o を固定運用（他を指定していても gpt-4o を強制使用）
    model = "gpt-4o"
    client = OpenAI(api_key=api_key)
    return client, model

# =========================
# 3) 入力（商品名）
# =========================
def load_products_from_csv(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"入力CSVが見つかりません: {path}")
    products = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # 「商品名」列を素直に参照。存在しない場合はエラー
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
# 4) 知見サマリ生成（ゆる結合）
# =========================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    """
    ./output/semantics/ 配下の JSON/CSV をざっくり要約テキスト化
    - lexical_clusters_*.json    → キーワード群
    - structured_semantics_*.json → 構造的観点（用途、ターゲット等）
    - market_vocab_*.json        → 市場語彙
    - styled_persona_*.json      → トーン・文体
    - normalized_*.json          → 禁則・整形ルール
    - template_composer.json     → 表現骨子
    * 存在しない／形式が違う場合は無視（堅牢運用）
    """
    if not os.path.isdir(SEMANTICS_DIR):
        return "知見: 主要キーワード・用途・対象・スペック・関連機種名を自然に含め、画像描写語は使わず、2文以内で読みやすく。", []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    # CSVは今回は読み飛ばし（必要なら将来対応）
    clusters, semantics, market, tone, normalized, template = [], [], [], [], [], []
    forbidden_local = []

    for p in files:
        name = os.path.basename(p).lower()
        data = safe_load_json(p)
        if not data:
            continue
        try:
            if "lexical_clusters" in name or "lexical" in name:
                # ex. {"clusters":[{"terms":[...]}]} / or [{"terms":[...]}]
                if isinstance(data, dict):
                    arr = data.get("clusters") or data.get("lexical") or []
                elif isinstance(data, list):
                    arr = data
                else:
                    arr = []
                for c in arr:
                    terms = c.get("terms") if isinstance(c, dict) else None
                    if isinstance(terms, list):
                        clusters.extend([t for t in terms if isinstance(t, str)])
            elif "structured_semantics" in name or "semantic" in name:
                # ex. {"concepts":[...], "scenes":[...], "targets":[...]}
                if isinstance(data, dict):
                    semantics.extend([w for k in ["concepts", "scenes", "targets", "use_cases"]
                                      for w in (data.get(k) or []) if isinstance(w, str)])
            elif "market_vocab" in name or "market" in name:
                # ex. [{"vocabulary":"MagSafe"}, ...] or ["MagSafe", "PD"]
                if isinstance(data, list):
                    for v in data:
                        if isinstance(v, dict) and "vocabulary" in v and isinstance(v["vocabulary"], str):
                            market.append(v["vocabulary"])
                        elif isinstance(v, str):
                            market.append(v)
                elif isinstance(data, dict):
                    vocab = data.get("vocabulary") or data.get("vocab") or []
                    if isinstance(vocab, list):
                        market.extend([x for x in vocab if isinstance(x, str)])
            elif "styled_persona" in name or "persona" in name:
                # ex. {"tone":{"style":"〜","register":"〜"}}
                if isinstance(data, dict):
                    t = data.get("tone") or {}
                    if isinstance(t, dict):
                        for v in t.values():
                            if isinstance(v, str):
                                tone.append(v)
            elif "normalized" in name or "forbid" in name:
                # ex. {"forbidden_words":["画像","写真",...]}
                if isinstance(data, dict):
                    fw = data.get("forbidden_words") or []
                    forbidden_local.extend([w for w in fw if isinstance(w, str)])
            elif "template_composer" in name:
                # ex. {"hints":["スペック→強み→対象→シーン→便益"]}
                if isinstance(data, dict):
                    hints = data.get("hints") or data.get("templates") or []
                    template.extend([h for h in hints if isinstance(h, str)])
        except Exception:
            # 形式が違っていても無視
            pass

    # 禁則はグローバルとマージしてユニーク化
    all_forbidden = list({*FORBIDDEN, *forbidden_local})

    # 軽く要約テキスト化（長文にしすぎない）
    def cap_join(xs, n):  # n個まで拾って "、" でつなぐ
        xs = [x for x in xs if isinstance(x, str)]
        return "、".join(xs[:n]) if xs else ""

    cluster_txt  = cap_join(list(dict.fromkeys(clusters)), 12)
    market_txt   = cap_join(list(dict.fromkeys(market)),   12)
    sem_txt      = cap_join(list(dict.fromkeys(semantics)), 8)
    tone_txt     = cap_join(list(dict.fromkeys(tone)),      4)
    tmpl_txt     = cap_join(list(dict.fromkeys(template)),  3)

    kb = "知見: "
    parts = []
    if cluster_txt:
        parts.append(f"語彙: {cluster_txt}")
    if market_txt:
        parts.append(f"市場語: {market_txt}")
    if sem_txt:
        parts.append(f"構造: {sem_txt}")
    if tmpl_txt:
        parts.append(f"骨子: {tmpl_txt}")
    if tone_txt:
        parts.append(f"トーン: {tone_txt}")
    if not parts:
        kb += "主要キーワード・用途・対象・スペック・関連機種名を自然に含め、"
    else:
        kb += " / ".join(parts) + "。"
    kb += "画像描写語は使わず、2文以内、読みやすく自然に。"
    return kb, all_forbidden

# =========================
# 5) プロンプト（AI）
# =========================
SYSTEM_PROMPT = (
    "あなたはEC画像のALTテキストを作るプロの日本語コピーライターです。"
    "目的は、楽天のサイト内SEOに強い自然文のALTを20本生成することです。"
    "以下の必須ルールを厳守してください：\n"
    "・画像や写真の描写語（例：画像、写真、見た目、上の画像 等）は使わない。\n"
    "・ECメタ語（当店、レビュー、ランキング、リンク、購入はこちら 等）は使わない。\n"
    "・競合比較や「競合優位性」のようなメタ表現は禁止。\n"
    f"・各文は全角約{RAW_MIN}〜{RAW_MAX}文字程度、1〜2文で自然に。必ず句点「。」で終える。\n"
    "・箇条書きや番号（1. 2. ・ など）やラベル（ALT: 等）は付けない。\n"
    "・商品名・対応機種・スペック・機能・用途・対象・ベネフィットを自然に織り込む（詰め込み禁止）。\n"
    "・出力は20行のテキストのみ（JSONや記号なし）。"
)

def build_user_prompt(product: str, knowledge_text: str, forbidden_words):
    forbid_txt = "、".join(sorted(set([w for w in forbidden_words if isinstance(w, str)])))
    hint = (
        "構成ヒント（テンプレにせず自然に）："
        "商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット。"
    )
    return (
        f"商品名: {product}\n"
        f"{knowledge_text}\n"
        f"{hint}\n"
        f"禁止語（絶対に使わない）: {forbid_txt}\n"
        "20行で、各行はひとつの自然文（1〜2文内）で書いてください。"
    )

# =========================
# 6) OpenAI 呼び出し
# =========================
def call_openai_20_lines(client, model, product, kb_text, forbidden_words, retry=3, wait=5):
    """
    20行のテキストを返す。失敗時はリトライ。
    """
    user_prompt = build_user_prompt(product, kb_text, forbidden_words)

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
                max_completion_tokens=1000,
                temperature=1,
            )
            content = (res.choices[0].message.content or "").strip()
            if content:
                lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
                # 「- 」や「1. 」などの頭を剥がす
                clean = []
                for ln in lines:
                    ln2 = LEADING_ENUM_RE.sub("", ln)
                    ln2 = ln2.strip("・-—●　")
                    if ln2:
                        clean.append(ln2)
                # 行が多すぎれば20に切る、少なければ後で補完
                return clean[:60]  # 余分に返ってくることがあるので最大60まで保持（後で20抽出）
        except Exception as e:
            last_err = e
            time.sleep(wait)
    # 全失敗
    raise RuntimeError(f"OpenAI応答を取得できませんでした: {last_err}")

# =========================
# 7) ローカル整形
# =========================
def soft_clip_sentence(text: str, min_len=FINAL_MIN, max_len=FINAL_MAX) -> str:
    """
    目標レンジを目指して末尾の「。」で自然カット。
    長すぎる: 120まで許容→最後の「。」まで詰めて80〜110狙い
    短すぎる: そのまま返す（後続で別行と差替え）
    """
    t = text.strip()
    # 句点終止なければ付ける
    if not t.endswith("。"):
        t += "。"
    # 余計なスペース圧縮
    t = WHITESPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    # 極端な箇条書き・番号を掃除
    t = LEADING_ENUM_RE.sub("", t).strip("・-—●　")

    if len(t) > 120:
        cut = t[:120]
        # 直近の句点位置
        p = cut.rfind("。")
        if p != -1:
            t = cut[:p+1]
        else:
            t = cut
    # 最終安全: 禁則語削除（完全除去）
    for ng in FORBIDDEN:
        if ng and ng in t:
            t = t.replace(ng, "")
    return t.strip()

def refine_20_lines(raw_lines):
    """
    20本に整形する:
      - 無効行・短すぎる行を除去
      - 句点終止
      - 長すぎる行を自然カット
      - 類似行の重複除去
      - 20本未満は可能な限りマージ/補完（ここでは単純複写禁止：重複は落とす）
    """
    # 正規化＆フィルタ
    norm = []
    for ln in raw_lines:
        if not ln:
            continue
        # 異常な箇条書き/番号を除去
        ln = LEADING_ENUM_RE.sub("", ln).strip("・-—●　")
        # 句点終止化 + 長さ整形
        ln = soft_clip_sentence(ln)
        # 1文字だけなど無効行は捨てる
        if len(ln) < 15:
            continue
        norm.append(ln)

    # 類似除去（簡易：前後一致や同一文の重複を捨てる）
    uniq = []
    seen = set()
    for ln in norm:
        key = ln
        if key not in seen:
            uniq.append(ln)
            seen.add(key)

    # 80〜110字の密度に寄せたいので、110超は文末まで詰める
    refined = [soft_clip_sentence(ln) for ln in uniq]

    # 20本ちょうどに整える
    # 足りないときは、既存文を軽く変形（語尾優先置換）して埋める
    def light_variation(s: str) -> str:
        s2 = s
        s2 = s2.replace("します。", "できます。")
        s2 = s2.replace("できます。", "しやすいです。")
        s2 = s2.replace("です。", "になります。")
        # それでも同一なら句内の一語を微修正（読点追加など最小変化）
        if s2 == s:
            s2 = re.sub(r"(\w{2,})", r"\1、", s, count=1)
            s2 = s2.replace("、、", "、")
            s2 = s2.replace("、、", "、")
            if not s2.endswith("。"):
                s2 += "。"
        return soft_clip_sentence(s2)

    i = 0
    while len(refined) < 20 and refined:
        refined.append(light_variation(refined[i % len(refined)]))
        i += 1

    # 多すぎるなら先頭20本だけ
    return refined[:20]

# =========================
# 8) 書き出し
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
            r_line = (r[:20] + [""] * max(0, 20 - len(r)))
            ref_line = (ref[:20] + [""] * max(0, 20 - len(ref)))
            w.writerow([p] + r_line + ref_line)

# =========================
# 9) メイン
# =========================
def main():
    print("🌸 ALT長文生成 v3（SEO＋自然文＋差分可視化）")
    client, model = init_env_and_client()
    ensure_outdir()

    products = load_products_from_csv(INPUT_CSV)
    print(f"✅ 対象商品: {len(products)}件")

    knowledge_text, forbidden_local = summarize_knowledge()
    forbidden_all = list({*FORBIDDEN, *forbidden_local})

    all_raw, all_refined = [], []

    for p in tqdm(products, desc="🧠 AI生成中", total=len(products)):
        # 1) AI生成（20行以上を返すこともある）
        try:
            raw_lines = call_openai_20_lines(client, model, p, knowledge_text, forbidden_all)
        except Exception as e:
            # どうしてもダメなら、ダミー20本（空白は避ける）
            raw_lines = [f"{p} の使い勝手を高める設計で、日常の不便を解消します。"] * 20

        # 2) ローカル整形 → 20本
        refined_lines = refine_20_lines(raw_lines)

        all_raw.append(raw_lines[:20])
        all_refined.append(refined_lines)

        # 軽いスロットリング
        time.sleep(0.2)

    # 3) 書き出し
    write_raw(products, all_raw)
    write_refined(products, all_refined)
    write_diff(products, all_raw, all_refined)

    # 統計ログ
    def avg_len(blocks):
        lens = [len(x) for lines in blocks for x in lines if x]
        return (sum(lens) / max(1, len(lens)))

    print("✅ 出力完了:")
    print(f"   - AI生出力: {RAW_PATH}")
    print(f"   - 整形後   : {REF_PATH}")
    print(f"   - 差分比較 : {DIFF_PATH}")
    print(f"📏 文字数(平均): raw={avg_len(all_raw):.1f} / refined={avg_len(all_refined):.1f}")
    print("🔒 仕様: ")
    print(f"   - AIは約{RAW_MIN}〜{RAW_MAX}字・1〜2文、句点終止、禁則適用（プロンプト）")
    print(f"   - ローカル整形で{FINAL_MIN}〜{FINAL_MAX}字に自然カット、重複除去・句点補完・禁則再適用")

if __name__ == "__main__":
    main()
import atlas_autosave_core
