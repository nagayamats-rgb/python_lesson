# -*- coding: utf-8 -*-
"""
writer_splitter_perfect_v3_5_unified.py
- 全件AI生成（楽天/Yahoo/ALT）＋ローカル知見要約＋禁則/長さ整形
-# 入力CSVを楽天／Yahooで別指定
RAKUTEN_INPUT = "/Users/tsuyoshi/Desktop/python_lesson/rakuten.csv"
YAHOO_INPUT   = "/Users/tsuyoshi/Desktop/python_lesson/yahoo.csv"
PRODUCT_NAME_COL = "商品名"
- 出力:
    ./output/ai_writer/rakuten_copy_YYYYMMDD_HHMM.csv
    ./output/ai_writer/yahoo_copy_YYYYMMDD_HHMM.csv
    ./output/ai_writer/alt_text_YYYYMMDD_HHMM.csv
    ./output/ai_writer/split_full_YYYYMMDD_HHMM.jsonl
"""

import os, sys, csv, json, time, re, unicodedata
from datetime import datetime
from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ====== OpenAI client (official SDK 1.x) ======
from openai import OpenAI
from openai import OpenAIError

# -------------------------------
# 設定
# -------------------------------
OUTPUT_DIR = "./output/ai_writer"
INPUT_CSV = "./input.csv"  # Shift-JIS, ヘッダあり, 「商品名」列
PRODUCT_NAME_COL = "商品名"

# 画像描写を避ける（ユーザー要望）
IMAGE_WORDS = [
    "画像", "写真", "見た目", "映える", "画面上", "写真のよう", "イメージ図", "サムネイル", "画像説明", "見た感じ",
    "色合いの写真", "写真はイメージ", "ビジュアル"
]

# 知見JSONのデフォルト探索パス
DEFAULT_PATHS = {
    "lexical": "./output/semantics/lexical_clusters_20251030_223013.json",
    "market": "./output/semantics/market_vocab_20251030_201906.json",
    "semantic": "./output/semantics/structured_semantics_20251030_224846.json",
    "persona": "./output/semantics/styled_persona_20251031_0031.json",
    "normalized": "./output/semantics/normalized_20251031_0039.json",
    "template": "./output/semantics/template_composer.json",
}

# 文字数ルール
RULES = {
    "rakuten_copy_min": 60, "rakuten_copy_max": 87,
    "rakuten_sp_min": 100, "rakuten_sp_max": 300,
    "yahoo_headline_min": 25, "yahoo_headline_max": 30,
    "yahoo_exp_min": 200, "yahoo_exp_max": 600,
    "yahoo_meta_min": 60, "yahoo_meta_max": 80,
    "alt_min": 80, "alt_max": 110,
    # ALTは生成時の長さ（ロング）→トリム
    "alt_gen_min": 120, "alt_gen_max": 150,
}

RETRY = 3
RETRY_WAIT = 6  # seconds

# -------------------------------
# ユーティリティ
# -------------------------------

def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M")

def zenkaku_len(s: str) -> int:
    """全角換算文字数（簡易）：半角=1, 全角=2 を合算し 2で割って四捨五入"""
    l = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("F", "W", "A"):
            l += 2
        else:
            l += 1
    # 全角基準に丸め込み
    return int(round(l / 2.0))

def clamp_by_len(s: str, min_len: int, max_len: int) -> str:
    """全角換算でmin〜maxに収める。長い場合は文末で自然に切る。短い場合はそのまま（AI側で再試行する前提）。"""
    if not s:
        return s
    # いったんトリム
    s = s.strip()
    # 長過ぎるとき：句点や読点・「、。!」等を目安に自然切り
    def natural_cut(text: str, limit: int) -> str:
        if zenkaku_len(text) <= limit:
            return text
        # 文末候補で切る
        marks = ["。", "！", "?", "？", "、", "，", "．", ".", "；", ";", "：", ":"]
        cut_idx = None
        current = ""
        for i, ch in enumerate(text):
            current += ch
            if zenkaku_len(current) > limit:
                break
            if ch in marks:
                cut_idx = i + 1
        if cut_idx is None:
            # 仕方なく生カット
            # 全角換算でlimitに相当する実長を概算
            acc = 0
            out = ""
            for ch in text:
                w = 2 if unicodedata.east_asian_width(ch) in ("F","W","A") else 1
                if int(round((acc + w)/2.0)) > limit:
                    break
                acc += w
                out += ch
            return out.rstrip("、。,.!！？　 ")
        return text[:cut_idx].rstrip("、。,.!！？　 ")

    if zenkaku_len(s) > max_len:
        s = natural_cut(s, max_len)

    # 最終：余計な空白・記号を落とす
    return s.strip("　 ").strip()

def remove_forbidden_words(s: str, forbidden: List[str]) -> str:
    if not s:
        return s
    t = s
    for w in forbidden + IMAGE_WORDS:
        if not w:
            continue
        t = t.replace(w, "")
    # 置換で変な空白ができたら整える
    t = re.sub(r"[ 　]{2,}", " ", t).strip()
    return t

def is_list_like(x):
    return isinstance(x, (list, tuple))

def jload_safe(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def extract_forbidden(normalized_cfg: Any) -> List[str]:
    # normalized が dict or list の両対応
    if isinstance(normalized_cfg, dict):
        fw = normalized_cfg.get("forbidden_words") or []
        return fw if is_list_like(fw) else []
    if is_list_like(normalized_cfg):
        out = []
        for item in normalized_cfg:
            if isinstance(item, dict) and "forbidden_words" in item:
                ws = item["forbidden_words"]
                if is_list_like(ws):
                    out.extend(ws)
        return list(OrderedDict.fromkeys(out))
    return []

def cap_from_list(xs: Any, key: Optional[str], limit: int) -> List[str]:
    out = []
    if not xs:
        return out
    if isinstance(xs, dict):
        xs = xs.get("items") or xs.get("list") or []
    if not is_list_like(xs):
        return out
    for el in xs[:limit]:
        if isinstance(el, dict):
            if key and key in el and isinstance(el[key], str):
                out.append(el[key])
        elif isinstance(el, str):
            out.append(el)
    # 重複排除
    seen = OrderedDict()
    for w in out:
        if w and w not in seen:
            seen[w] = True
    return list(seen.keys())

def summarize_knowledge(cfgs: Dict[str, Any]) -> Tuple[str, List[str]]:
    """ローカル知見JSON群を要約テキスト化＋禁則語抽出"""
    persona = cfgs.get("persona")
    lexical = cfgs.get("lexical")
    market  = cfgs.get("market")
    sem     = cfgs.get("semantic")
    template= cfgs.get("template")
    normalized = cfgs.get("normalized")

    # 軽量サマリ（テンプレではなく“構文ヒント”を自然言語化）
    tones = []
    if isinstance(persona, dict):
        t = persona.get("tone")
        if isinstance(t, dict):
            tones = [f"{k}:{v}" for k,v in t.items() if isinstance(v, str)]
        elif is_list_like(t):
            tones = [str(x) for x in t if isinstance(x, str)]
    persona_hint = "／".join(tones[:5]) if tones else "落ち着いた自然文・誇張なし・誠実"

    cluster_terms = cap_from_list(lexical, key="term", limit=12)
    market_terms  = cap_from_list(market,  key="vocabulary", limit=12)
    concepts      = cap_from_list(sem,     key="concept", limit=12)

    # テンプレートの有用キュレーション
    tmpl_snips = []
    if isinstance(template, dict):
        for k,v in template.items():
            if isinstance(v, str) and len(v) < 120:
                tmpl_snips.append(v)
            elif is_list_like(v):
                tmpl_snips.extend([x for x in v if isinstance(x, str) and len(x)<120])
    tmpl_hint = "｜".join(tmpl_snips[:6])

    knowledge_text = (
        f"文体ヒント:{persona_hint}\n"
        f"頻出語（クラスタ）:{'、'.join(cluster_terms)}\n"
        f"市場語彙:{'、'.join(market_terms)}\n"
        f"概念/訴求:{'、'.join(concepts)}\n"
        f"表現ヒント:{tmpl_hint}"
    ).strip()

    forbidden = extract_forbidden(normalized)
    # 画像描写禁止ワードも追加（最後でまとめ除去）
    forbidden = list(OrderedDict.fromkeys(forbidden + IMAGE_WORDS))
    return knowledge_text, forbidden

def load_all_knowledge(paths: Dict[str, str]) -> Dict[str, Any]:
    return {
        "lexical": jload_safe(paths["lexical"]),
        "market": jload_safe(paths["market"]),
        "semantic": jload_safe(paths["semantic"]),
        "persona": jload_safe(paths["persona"]),
        "normalized": jload_safe(paths["normalized"]),
        "template": jload_safe(paths["template"]),
    }

def read_product_names(input_csv: str) -> List[str]:
    # Shift-JIS読込、ヘッダあり、「商品名」列を探す
    try_encs = ["cp932", "shift_jis", "utf-8-sig", "utf-8"]
    df = None
    last_err = None
    for enc in try_encs:
        try:
            df = pd.read_csv(input_csv, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        print(f"❌ CSV読込失敗: {input_csv} / {last_err}")
        return []
    if PRODUCT_NAME_COL not in df.columns:
        print(f"❌ ヘッダに『{PRODUCT_NAME_COL}』がありません。列一覧: {list(df.columns)[:10]} ...")
        return []

    names = [str(x).strip() for x in df[PRODUCT_NAME_COL].tolist() if isinstance(x, str) and x.strip()]
    # 重複除去（順序保持）
    seen = OrderedDict()
    for n in names:
        if n not in seen:
            seen[n] = True
    return list(seen.keys())

# -------------------------------
# OpenAI コール
# -------------------------------

def call_openai_json(client: OpenAI, model: str, messages: List[Dict[str, str]], max_completion_tokens: int = 800) -> Optional[Dict[str, Any]]:
    """
    Chat Completions(JSON). 温度は未指定（デフォルト=1）。max_completion_tokens を使用。
    """
    for _ in range(RETRY):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )
            content = (res.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty content")
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ OpenAIエラー: {e}")
            time.sleep(RETRY_WAIT)
    return None

# -------------------------------
# プロンプト
# -------------------------------

def build_system_prompt(knowledge_text: str, forbidden: List[str]) -> str:
    # ユーザー哲学に沿って“テンプレ禁止”ではなく“構成ヒントを自然に含める”
    return (
        "あなたは日本語のEC商品コピーの専門ライターです。以下を厳守して自然な文を作成します。\n"
        "・各フィールドは1〜2文で自然な日本語。\n"
        "・構成ヒント（テンプレではない）:『商品スペック→コアコンピタンス→どんな人→シーン→ベネフィット』を無理なく含める。\n"
        "・絵文字・特殊記号・HTMLタグは禁止。改行禁止の欄では改行を入れない。\n"
        "・禁止語は使用しない。競合比較や“競合優位性”のようなメタ表現は避ける。\n"
        "・文字数は“全角換算”で各フィールドのmin〜maxに入る程度に意識。\n"
        "・ALTは画像の描写をしない（被写体や構図の説明はしない）。\n"
        "・返答は JSON オブジェクト（各キー：rakuten_copy, rakuten_sp, yahoo_headline, yahoo_explanation, "
        "yahoo_meta, alt_list）で返す。\n"
        "・alt_list は 20本の文（120〜150字を目安）を生成。\n"
        "――――――――\n"
        f"【知見要約】\n{knowledge_text}\n"
        "――――――――\n"
        f"【禁止語（含む画像描写NG語）】\n{', '.join(forbidden)}\n"
    )

def build_user_prompt(product_name: str) -> str:
    return (
        f"商品名: {product_name}\n"
        "出力要件（全角換算）:\n"
        f"- 楽天キャッチコピー: {RULES['rakuten_copy_min']}〜{RULES['rakuten_copy_max']}文字\n"
        f"- 楽天スマホ説明: {RULES['rakuten_sp_min']}〜{RULES['rakuten_sp_max']}文字\n"
        f"- Yahoo headline: {RULES['yahoo_headline_min']}〜{RULES['yahoo_headline_max']}文字\n"
        f"- Yahoo explanation: {RULES['yahoo_exp_min']}〜{RULES['yahoo_exp_max']}文字\n"
        f"- Yahoo meta-desc: {RULES['yahoo_meta_min']}〜{RULES['yahoo_meta_max']}文字\n"
        f"- ALT×20: {RULES['alt_gen_min']}〜{RULES['alt_gen_max']}文字で生成（後で80〜110に調整）\n"
        "返答はJSON（例）：\n"
        "{\n"
        '  "rakuten_copy": "...",\n'
        '  "rakuten_sp": "...",\n'
        '  "yahoo_headline": "...",\n'
        '  "yahoo_explanation": "...",\n'
        '  "yahoo_meta": "...",\n'
        '  "alt_list": ["...", "...", "...（20本）"]\n'
        "}\n"
    )

# -------------------------------
# 整形（ローカル）
# -------------------------------

def refine_field(s: str, min_len: int, max_len: int, forbidden: List[str]) -> str:
    s = (s or "").replace("\n", "").strip()
    s = remove_forbidden_words(s, forbidden)
    s = clamp_by_len(s, min_len, max_len)
    return s

def refine_alt_list(alts: List[str], forbidden: List[str]) -> List[str]:
    # 120〜150で生まれたALTを 80〜110 に自然トリム
    out = []
    for a in alts[:20]:
        t = remove_forbidden_words((a or "").replace("\n"," ").strip(), forbidden)
        t = clamp_by_len(t, RULES["alt_min"], RULES["alt_max"])
        if t:
            out.append(t)
    # 20本に満たない場合は空を埋めず、短欠落のまま（欠落率の可視化のため）
    return out

# -------------------------------
# メイン
# -------------------------------

def main():
    print("🌸 writer_splitter_perfect_v3_5_unified 実行開始（全件AI＋知見要約＋禁則/長さ整形）")

    # 環境
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が未設定です。.envを確認してください。")
        sys.exit(1)
    model = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

    ensure_output_dirs()
    tag = now_tag()

    # 知見ロード＋要約
    cfg = load_all_knowledge(DEFAULT_PATHS)
    knowledge_text, forbidden = summarize_knowledge(cfg)

    # 入力商品名
    names = read_product_names(INPUT_CSV)
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")
    if not names:
        print("❌ 商品名が見つかりません。処理を終了します。")
        sys.exit(1)

    client = OpenAI()

    # 出力準備
    path_rak = os.path.join(OUTPUT_DIR, f"rakuten_copy_{tag}.csv")
    path_yah = os.path.join(OUTPUT_DIR, f"yahoo_copy_{tag}.csv")
    path_alt = os.path.join(OUTPUT_DIR, f"alt_text_{tag}.csv")
    path_jsonl = os.path.join(OUTPUT_DIR, f"split_full_{tag}.jsonl")

    # CSV ヘッダ
    rak_cols = ["商品名", "楽天_キャッチコピー", "楽天_スマホ説明"]
    yah_cols = ["商品名", "Yahoo_headline", "Yahoo_explanation", "Yahoo_meta_desc"]
    alt_cols = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]

    # 初期化
    with open(path_rak, "w", newline="", encoding="utf-8") as fr, \
         open(path_yah, "w", newline="", encoding="utf-8") as fy, \
         open(path_alt, "w", newline="", encoding="utf-8") as fa, \
         open(path_jsonl, "w", encoding="utf-8") as fj:

        wr_rak = csv.writer(fr)
        wr_yah = csv.writer(fy)
        wr_alt = csv.writer(fa)
        wr_rak.writerow(rak_cols)
        wr_yah.writerow(yah_cols)
        wr_alt.writerow(alt_cols)

        for nm in tqdm(names, desc="🧠 商品別AI生成中", ncols=100):
            sys_msg = {"role": "system", "content": build_system_prompt(knowledge_text, forbidden)}
            user_msg = {"role": "user", "content": build_user_prompt(nm)}

            raw = call_openai_json(client, model, [sys_msg, user_msg],
                                   max_completion_tokens=900)  # やや余裕

            # フェイルセーフ
            rak_copy = rak_sp = yah_h = yah_e = yah_m = ""
            alts = []

            if isinstance(raw, dict):
                rak_copy = raw.get("rakuten_copy", "") or ""
                rak_sp   = raw.get("rakuten_sp", "") or ""
                yah_h    = raw.get("yahoo_headline", "") or ""
                yah_e    = raw.get("yahoo_explanation", "") or ""
                yah_m    = raw.get("yahoo_meta", "") or ""
                tmp_alts = raw.get("alt_list", []) or []
                if is_list_like(tmp_alts):
                    alts = [str(x) for x in tmp_alts if isinstance(x, str)]

            # ローカル整形（禁則・長さ）
            rak_copy = refine_field(rak_copy, RULES["rakuten_copy_min"], RULES["rakuten_copy_max"], forbidden)
            rak_sp   = refine_field(rak_sp,   RULES["rakuten_sp_min"],   RULES["rakuten_sp_max"],   forbidden)
            yah_h    = refine_field(yah_h,    RULES["yahoo_headline_min"], RULES["yahoo_headline_max"], forbidden)
            yah_e    = refine_field(yah_e,    RULES["yahoo_exp_min"],      RULES["yahoo_exp_max"],   forbidden)
            yah_m    = refine_field(yah_m,    RULES["yahoo_meta_min"],     RULES["yahoo_meta_max"],  forbidden)
            alts_ref = refine_alt_list(alts, forbidden)

            # CSV書き出し
            wr_rak.writerow([nm, rak_copy, rak_sp])
            wr_yah.writerow([nm, yah_h, yah_e, yah_m])

            row_alt = [nm] + alts_ref + [""] * (20 - len(alts_ref))
            wr_alt.writerow(row_alt)

            # JSONL ログ（生/整形後の両方）
            log_item = {
                "product_name": nm,
                "raw": raw,
                "refined": {
                    "rakuten_copy": rak_copy,
                    "rakuten_sp": rak_sp,
                    "yahoo_headline": yah_h,
                    "yahoo_explanation": yah_e,
                    "yahoo_meta": yah_m,
                    "alt_list": alts_ref
                }
            }
            fj.write(json.dumps(log_item, ensure_ascii=False) + "\n")

    print("✅ 出力完了:")
    print(f"   - 楽天: {path_rak}")
    print(f"   - Yahoo: {path_yah}")
    print(f"   - ALT20: {path_alt}")
    print(f"   - JSONL: {path_jsonl}")
    print("✅ 共通ALT20は『alt_text_*.csv』にALT1〜ALT20として横持ちします。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
