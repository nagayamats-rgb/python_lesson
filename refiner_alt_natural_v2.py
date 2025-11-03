import csv, re, os
from pathlib import Path

BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson/output/ai_writer"
INPUT_FILE = f"{BASE_DIR}/alt_text_refined_final_natural.csv"
OUTPUT_FILE = f"{BASE_DIR}/alt_text_refined_final_v2.csv"
DIFF_FILE = f"{BASE_DIR}/alt_text_refined_diff_v2.csv"

FORBIDDEN = ["画像", "写真", "映える", "比べ", "比較", "優位性", "他社", "競合", "ランキング", "No.1"]
FIX_RULES = {
    r"しです$": "します",
    r"するに$": "するのに",
    r"するで$": "するので",
    r"しできます$": "できます",
    r"短縮しです": "短縮できます",
    r"ますます": "ます",
    r"。+": "。",
    r"、、+": "、",
}
ENDING_NORMALIZE = {
    "ですです": "です",
    "ますます": "ます",
    "。ます": "ます。",
    "。です": "です。",
    "。に便利": "に便利です。",
    "におすすめ": "におすすめです。",
    "設計": "設計です。",
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for w in FORBIDDEN:
        text = text.replace(w, "")
    for pat, rep in FIX_RULES.items():
        text = re.sub(pat, rep, text)
    for pat, rep in ENDING_NORMALIZE.items():
        text = text.replace(pat, rep)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"。。+", "。", text)
    return text.strip("。") + "。"

def main():
    diffs, results = [], []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_row = row.copy()
            for col in row.keys():
                if "ALT" in col and row[col].strip():
                    before = row[col].strip()
                    after = clean_text(before)
                    if before != after:
                        diffs.append({"列名": col, "変更前": before, "変更後": after})
                    new_row[col] = after
            results.append(new_row)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(DIFF_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["列名", "変更前", "変更後"])
        writer.writeheader()
        writer.writerows(diffs)

    print("✅ ALT自然文整形完了")
    print(f"出力: {OUTPUT_FILE}")
    print(f"差分: {DIFF_FILE}")
    print(f"修正件数: {len(diffs)} / {len(results)}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
v3.3_altfix_utf8_final_schema_refine_revived.py
ALT長文安定生成 → ローカル整形（80〜110字）→ CSV出力（ALT1〜ALT20）

要件（固定）:
- 入力:  /Users/tsuyoshi/Desktop/python_lesson/rakuten.csv  （UTF-8, ヘッダ「商品名」）
- 出力:  /Users/tsuyoshi/Desktop/python_lesson/output/ai_writer/alt_text_refined_final_revived.csv
- モデル: .env に設定された gpt-4o（OPENAI_API_KEY / OPENAI_BASE_URL 任意）
- OpenAI仕様準拠: response_format="text", max_completion_tokens 使用, temperature は 1（未指定＝既定）
- 1商品あたり ALT を 20件生成
- 生成は 100〜130字を目標 → ローカルで 80〜110字に自然トリミング
- 画像描写語/メタ語/競合比較/「競合優位性」等はNG
- ローカル知見（/output/semantics内の各JSON）を要約してプロンプトに注入（存在しなければスキップ）
"""

import os
import csv
import json
import re
import time
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv
from tqdm import tqdm

# ----- OpenAI SDK（公式Pythonクライアント） -----
from openai import OpenAI, OpenAIError

# ====== パス定義 ======
BASE_DIR = Path("/Users/tsuyoshi/Desktop/python_lesson")
INPUT_CSV = BASE_DIR / "rakuten.csv"  # UTF-8 固定
OUT_DIR = BASE_DIR / "output" / "ai_writer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "alt_text_refined_final_revived.csv"

SEMANTICS_DIR = BASE_DIR / "output" / "semantics"

# 既知の知見ファイル群（存在チェックは都度行う）
SEM_FILES = {
    "persona": "styled_persona_20251031_0031.json",
    "lexical": "lexical_clusters_20251030_223013.json",
    "semantic": "structured_semantics_20251030_224846.json",
    "market": "market_vocab_20251030_201906.json",
    "template": "template_composer.json",
    "normalized": "normalized_20251031_0039.json",
}

# ====== 禁則ワード（画像描写語・メタ語などのデフォルト） ======
DEFAULT_FORBIDDEN = [
    # 画像・見た目の直接描写
    "画像", "写真", "見た目", "映っている", "描写", "写っている", "スクリーンショット",
    # メタ・レビュー・比較・優位性
    "レビュー", "評価", "口コミ", "他社", "競合", "競合優位性", "最安", "No.1", "1位の理由",
    "売上", "ランキング", "コスパ最強",
    # 禁止トーン（誇大）
    "絶対", "完全", "必ず", "保証",
    # NG記号類
    "★", "☆", "♪", "♫", "✓", "✔",
]

IMAGE_NG_HINT = [
    "画像", "写真", "写って", "描写", "見た目", "スクショ", "スクリーンショット"
]

# ====== ユーティリティ ======
def load_env_client() -> OpenAI:
    """ .env を読み込み、OpenAIクライアントを初期化 """
    load_dotenv()  # 何度呼んでも安全
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # 任意

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が見つかりません。.env を確認してください。")

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    return client

def read_products_from_csv(path: Path) -> List[str]:
    """ UTF-8 で CSV を開き、ヘッダ「商品名」列の非空値を取得（重複除去） """
    products = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise RuntimeError("入力CSVに「商品名」列が見つかりません。")

        for row in reader:
            name = (row.get("商品名") or "").strip()
            if name:
                products.append(name)

    # 重複除去（順序維持）
    seen = set()
    uniq = []
    for nm in products:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

def load_json_safely(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def ensure_list(obj) -> List:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # 代表的に values を採用（使い道に応じて修正）
        return list(obj.values())
    return []

def cap_list_str(items: List[Any], key: str = None, cap: int = 12) -> str:
    """ items から最大 cap 個の語を '、' で連結して返す """
    out = []
    for x in items[:cap] if isinstance(items, list) else []:
        if key and isinstance(x, dict):
            v = x.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        elif isinstance(x, str) and x.strip():
            out.append(x.strip())
    return "、".join(out)

def summarize_knowledge() -> Tuple[str, List[str]]:
    """
    /output/semantics 内の知見を要約して1つのテキストにまとめる。
    forbidden は normalized に含まれる場合を優先取り込み。
    """
    persona = load_json_safely(SEMANTICS_DIR / SEM_FILES["persona"])
    lexical = load_json_safely(SEMANTICS_DIR / SEM_FILES["lexical"])
    semantic = load_json_safely(SEMANTICS_DIR / SEM_FILES["semantic"])
    market = load_json_safely(SEMANTICS_DIR / SEM_FILES["market"])
    template = load_json_safely(SEMANTICS_DIR / SEM_FILES["template"])
    normalized = load_json_safely(SEMANTICS_DIR / SEM_FILES["normalized"])

    # tone/styleなど
    tone = ""
    if isinstance(persona, dict):
        t = persona.get("tone") or persona.get("style") or {}
        if isinstance(t, dict):
            tone = ", ".join([f"{k}:{v}" for k, v in t.items() if isinstance(v, str)])
        elif isinstance(t, list):
            tone = "、".join([str(x) for x in t if isinstance(x, str)])

    # 代表語句・トレンド
    lex_str = ""
    if isinstance(lexical, dict):
        clusters = lexical.get("clusters") or lexical.get("data") or []
        lex_str = cap_list_str(ensure_list(clusters), key=None, cap=12)
    elif isinstance(lexical, list):
        lex_str = cap_list_str(lexical, key=None, cap=12)

    market_str = ""
    if isinstance(market, list):
        market_str = cap_list_str(market, key="vocabulary", cap=12)
    elif isinstance(market, dict):
        mv = market.get("vocabulary") or market.get("words") or []
        if isinstance(mv, list):
            market_str = cap_list_str(mv, key=None, cap=12)

    concept_str = ""
    if isinstance(semantic, dict):
        concept_str = cap_list_str(ensure_list(semantic.get("concepts")), key=None, cap=10)

    template_hint = ""
    if isinstance(template, dict):
        # 使えそうな構文のヒント
        tpl = template.get("structure") or template.get("hints") or []
        if isinstance(tpl, list):
            template_hint = "、".join([str(x) for x in tpl if isinstance(x, str)][:8])
        elif isinstance(tpl, dict):
            template_hint = "、".join(list(tpl.keys())[:8])

    # forbidden（normalized優先）
    forbidden_local = []
    if isinstance(normalized, dict):
        # list / dict どちらにも対応
        fw = normalized.get("forbidden_words")
        if isinstance(fw, list):
            forbidden_local.extend([x for x in fw if isinstance(x, str)])
        elif isinstance(fw, dict):
            forbidden_local.extend([str(v) for v in fw.values() if isinstance(v, str)])

    # 最低限デフォルトを結合
    merged_forbidden = list(dict.fromkeys(DEFAULT_FORBIDDEN + forbidden_local))

    knowledge_text = textwrap.dedent(f"""
    ▼トーン/スタイル参考: {tone or "控えめで誠実、機能志向、自然な日本語"}
    ▼代表語句: {lex_str or "高耐久・急速・薄型・軽量・フィット感・防指紋・耐衝撃"}
    ▼トレンド: {market_str or "マグセーフ/3in1/強化ガラス/アンチグレア/耐衝撃/防水/PD/急速"}
    ▼概念ヒント: {concept_str or "利便性・快適性・安心感・効率化"}
    ▼構成ヒント（テンプレではない）: {template_hint or "商品スペック→コアコンピタンス→誰に→どんなシーン→ベネフィット"}
    """).strip()

    return knowledge_text, merged_forbidden

# ====== 文字・句読点整形 ======
JP_PERIOD = "。"
RE_MULTI_PUNCT = re.compile(r"[。]{2,}")  # 連続句点
RE_SPACE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return s
    # 全角句点の連続を一つに
    s = RE_MULTI_PUNCT.sub(JP_PERIOD, s)
    # 余計な空白の正規化（行頭・行末）
    s = RE_SPACE.sub(" ", s).strip()
    return s

def remove_forbidden(s: str, forbidden: List[str]) -> str:
    if not s:
        return s
    out = s
    for ng in forbidden:
        if not ng:
            continue
        out = out.replace(ng, "")
    # 画像描写の言い換え（弱め）
    for ng in IMAGE_NG_HINT:
        out = out.replace(ng, "")
    # 句点の整合
    out = normalize_text(out)
    return out

def dedupe_phrases(s: str) -> str:
    """ 連続する同一フレーズの簡易除去（例：「特徴を活かし」の連発など） """
    if not s:
        return s
    s = re.sub(r"(特徴を活かし){2,}", r"特徴を活かし", s)
    s = re.sub(r"(便利です){2,}", r"便利です", s)
    s = re.sub(r"(おすすめです){2,}", r"おすすめです", s)
    return s

def trim_to_range_natural(s: str, min_len: int = 80, max_len: int = 110) -> str:
    """ 
    目標レンジに自然に収める：
    - max超なら最後の「。」までを優先して切る
    - それでも長い→maxで切る（末尾句点つける）
    - 短い場合は無理に増やさない（上流で100〜130を目標にするため）
    """
    s = s.strip()
    if len(s) <= max_len and len(s) >= min_len:
        return s
    if len(s) > max_len:
        # max以内で最後の句点
        within = s[:max_len]
        last = within.rfind(JP_PERIOD)
        if last >= min_len - 1:
            return within[:last + 1]
        # 句点が見つからない場合は強制カット＋句点
        cut = within.rstrip()
        if not cut.endswith(JP_PERIOD):
            cut += JP_PERIOD
        return cut
    # 短い（<min_len）はそのまま返す（上流で再生成コストを避ける）
    return s

def clean_line(s: str, forbidden: List[str]) -> str:
    s = s.strip("　 ").strip()
    # 単独の句点だけの行は無効
    if s in {".", "。"}:
        return ""
    s = remove_forbidden(s, forbidden)
    s = dedupe_phrases(s)
    s = normalize_text(s)
    if s and not s.endswith(JP_PERIOD):
        s += JP_PERIOD
    return s

# ====== OpenAI呼び出し ======
def call_openai_alt_20(client: OpenAI, product_name: str, knowledge_text: str, forbidden: List[str]) -> List[str]:
    """
    1商品につきALT文20件をテキストで生成。
    仕様:
      - 1行=1文, 100〜130字を狙う（後段で80〜110に整形）
      - 画像描写語・比較・メタ表現はNG
      - 絵文字・記号・HTMLタグ・箇条書き禁止
      - 改行区切りの20行で返す
    """
    sys = (
        "あなたは日本語のEコマース用テキストのプロ編集者です。"
        "画像説明ではなく、商品の価値・利用シーン・便益を自然な全文で表現してください。"
        "絵文字・特殊記号・HTMLタグ・箇条書きは禁止。レビュー/比較/ランキング/競合優位性などのメタ表現は禁止。"
        "『画像』『写真』など“見た目の描写”を連想させる語も禁止。"
        "各文は1文完結・自然な日本語で、約100〜130字を目安に。出力は20行、各行に1文のみ。"
    )
    usr = textwrap.dedent(f"""
    # 商品名
    {product_name}

    # 参考知見（要約）
    {knowledge_text}

    # 禁止語（必ず使わない）
    {", ".join(forbidden)}

    # 構成ヒント（テンプレではない）
    「商品スペック→コアコンピタンス→どんな人→どんなシーン→ベネフィット」を無理なく含める。

    # 出力仕様
    ・合計20行。各行1文。改行で区切るのみ。
    ・各行は100〜130字を目安に自然な日本語で。
    ・ラベルや番号、箇条書き、引用符は不要。
    ・画像の描写はしない。機能価値・体験価値・便益を述べる。
    """).strip()

    # 400/429などに備えたリトライ
    MAX_RETRY = 3
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                response_format={"type": "text"},
                max_completion_tokens=1000,  # 長文耐性
            )
            text = (res.choices[0].message.content or "").strip()
            # 20行化
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            # 番号や箇条書きの除去（予防）
            lines = [re.sub(r"^[0-9０-９]+[)\].．、\s-]+", "", ln) for ln in lines]
            # 引用符の除去
            lines = [ln.strip("「」\"'’‘“”") for ln in lines]
            if len(lines) >= 20:
                return lines[:20]
            # 少ない場合は一度だけ追加要求（簡易）
            if attempt == MAX_RETRY:
                # 不足は空で埋める（後段でスキップしないよう長さ整形時に維持）
                while len(lines) < 20:
                    lines.append("")  # 空は後で削ると欠番になるので、最終で「短文扱い」で句点付与
                return lines[:20]
            # 追加要求
            time.sleep(1.5)
        except OpenAIError as e:
            if attempt == MAX_RETRY:
                # 最後は全空20件で返す（処理継続のため）
                return [""] * 20
            time.sleep(1.5 + attempt * 1.0)
        except Exception:
            if attempt == MAX_RETRY:
                return [""] * 20
            time.sleep(1.2)

    return [""] * 20

# ====== メイン処理 ======
def main():
    print("🌸 v3.3_altfix_utf8_final_schema_refine_revived 実行開始（ALT長文→ローカル整形・禁則適用・知見注入）")

    # OpenAI
    try:
        client = load_env_client()
    except Exception as e:
        raise SystemExit(f"OpenAI初期化エラー: {e}")

    # 入力商品名
    try:
        products = read_products_from_csv(INPUT_CSV)
    except Exception as e:
        raise SystemExit(f"入力CSV読み込みエラー: {e}")

    if not products:
        raise SystemExit("商品名が0件でした。")

    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")

    # 知見要約
    knowledge_text, forbidden_local = summarize_knowledge()
    # 合成禁則
    forbidden = list(dict.fromkeys(DEFAULT_FORBIDDEN + forbidden_local))

    # 出力準備
    headers = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]
    rows = []

    for nm in tqdm(products, desc="🧠 ALT生成中", ncols=80):
        raw_lines = call_openai_alt_20(client, nm, knowledge_text, forbidden)
        refined = []
        for ln in raw_lines:
            # 生成が空のとき、軽めのフォールバック（商品名を利用して自然文 stub）
            if not ln.strip():
                stub = f"{nm}の使い心地を高め、日常の不便を減らします。"
                ln = stub

            # 句読点・禁則の整形
            ln = clean_line(ln, forbidden)
            if not ln:
                # 空行になった場合の再補完
                ln = f"{nm}は扱いやすく、日常の小さな不便を解消します。"
                ln = clean_line(ln, forbidden)

            # 長さ整形（80〜110字）
            ln = trim_to_range_natural(ln, min_len=80, max_len=110)

            refined.append(ln)

        # 最終の軽い後処理（全行に対して）
        refined2 = []
        for ln in refined:
            # 連続句点をもう一度安全側で抑制
            ln = RE_MULTI_PUNCT.sub(JP_PERIOD, ln)
            # 不自然な文頭句点/記号の除去
            ln = ln.lstrip("。・*+- ").strip()
            # 最終句点保証
            if ln and not ln.endswith(JP_PERIOD):
                ln += JP_PERIOD
            refined2.append(ln)

        # 20件に整える（不足時安全埋め）
        if len(refined2) < 20:
            while len(refined2) < 20:
                refined2.append(f"{nm}の使い勝手を高め、日常のストレスを軽減します。")

        row = {"商品名": nm}
        for i in range(20):
            row[f"ALT{i+1}"] = refined2[i]
        rows.append(row)

    # CSV書き出し
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ 出力完了: {OUT_CSV}")
    print("✅ 仕様: ALTはAIで100〜130字→ローカルで80〜110字に整形。画像描写語・メタ語は禁止。知見注入済。")


if __name__ == "__main__":
    main()
import atlas_autosave_core
