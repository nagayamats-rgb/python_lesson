# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v5.6.2
GPT-5安全モード／分割ALT生成＋再試行・ログ強化版
"""

import os, csv, json, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# パス設定
# ===============================
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
OUT_DIR = os.path.join(BASE_DIR, "output/ai_writer")
os.makedirs(OUT_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ENCODING_IN = "cp932"

# ===============================
# 中間ファイルパス
# ===============================
PATH_LEXICAL = os.path.join(SEM_DIR, "lexical_clusters_20251030_223013.json")
PATH_MARKET  = os.path.join(SEM_DIR, "market_vocab_20251030_201906.json")
PATH_SEMANT  = os.path.join(SEM_DIR, "structured_semantics_20251030_224846.json")
PATH_PERSONA = os.path.join(SEM_DIR, "styled_persona_20251031_0031.json")
PATH_NORMAL  = os.path.join(SEM_DIR, "normalized_20251031_0039.json")
PATH_TEMPLATE = os.path.join(SEM_DIR, "template_composer.json")

# ===============================
# ユーティリティ
# ===============================
def sanitize(s: str):
    return re.sub(r"\s+", " ", s.replace("\u3000", " ").strip())

def load_json(p, d=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return d or {}

def short(x):
    """プロンプト軽量化"""
    if isinstance(x, list):
        return x[:5]
    if isinstance(x, dict):
        return {k: x[k] for k in list(x.keys())[:5]}
    return x

def _parse_json_loose(text: str):
    """JSONが壊れていても緩くパース"""
    import json, re
    if not text:
        return {}
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {}

# ===============================
# GPT呼び出しロジック
# ===============================
def ai_generate(name, persona, lexical, market, sem, tmpl, norm):
    # forbidden_wordsを安全取得
    if isinstance(norm, dict):
        forbidden_words = norm.get("forbidden_words", [])
    elif isinstance(norm, list):
        forbidden_words = norm
    else:
        forbidden_words = []

    sys_msg = "あなたは日本語ECコピーの専門家です。必ずJSONオブジェクトのみを返します。"

    base_prompt = f"""
商品名: {name}

出力要件:
- 40〜60文字のキャッチコピーを "copy" に
- 80〜110文字のALT文を10件、"alts" 配列に
- 禁則語は使用しない: {json.dumps(forbidden_words, ensure_ascii=False)}

返却は次のJSONのみ（他の文字は禁止）:
{{
  "copy": "ここにキャッチコピー",
  "alts": ["ALT1","ALT2",...,"ALT10"]
}}
"""

    # -----------------------------
    # 1回目：copy+ALT(1-10)
    # -----------------------------
    try:
        res1 = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": base_prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=600
        )
        raw1 = res1.choices[0].message.content or ""
        data1 = _parse_json_loose(raw1)
        copy = (data1.get("copy") or "").strip()
        alts1 = data1.get("alts") or []
    except Exception as e:
        copy, alts1, raw1 = "", [], f"ERROR:{e}"

    # 失敗・空応答時はリトライ
    if not copy or not alts1:
        retry_prompt = f"""
商品名: {name}
JSONのみで返答:
{{"copy":"40-60文字","alts":["80-110文字ALT×10"]}}
禁則語: {json.dumps(forbidden_words, ensure_ascii=False)}
"""
        try:
            res1b = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": retry_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=550
            )
            raw1b = res1b.choices[0].message.content or ""
            data1b = _parse_json_loose(raw1b)
            copy = copy or (data1b.get("copy") or "").strip()
            alts1 = alts1 or (data1b.get("alts") or [])
        except:
            pass

    # -----------------------------
    # 2回目：ALT(11-20)
    # -----------------------------
    alt_prompt = f"""
商品名: {name}
先ほどと重複しないALT文を10件追加してください。
JSONのみ:
{{"alts":["...×10"]}}
禁則語: {json.dumps(forbidden_words, ensure_ascii=False)}
"""
    try:
        res2 = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": alt_prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=450
        )
        raw2 = res2.choices[0].message.content or ""
        data2 = _parse_json_loose(raw2)
        alts2 = data2.get("alts") or []
    except Exception:
        alts2 = []

    alts = (alts1 + alts2)[:20]

    # フォールバック
    if not copy:
        copy = "上質な使い心地を追求した人気の定番アイテム"
    while len(alts) < 20:
        alts.append("")

    return copy, alts

# ===============================
# メイン処理
# ===============================
def main():
    print("🌸 Hybrid AI Writer v5.6.2 実行開始（安全モード／分割ALT生成＋再試行）")

    cfg = {
        "lex": load_json(PATH_LEXICAL),
        "market": load_json(PATH_MARKET),
        "sem": load_json(PATH_SEMANT),
        "persona": load_json(PATH_PERSONA),
        "tmpl": load_json(PATH_TEMPLATE),
        "norm": load_json(PATH_NORMAL)
    }

    # --- CSV読込 ---
    with open(INPUT_CSV, "r", encoding=ENCODING_IN) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    name_idx = header.index("商品名")
    names = [sanitize(r[name_idx]) for r in rows[1:] if len(r) > name_idx and sanitize(r[name_idx])]
    uniq = list(dict.fromkeys(names))
    print(f"✅ 商品名抽出: {len(names)}件 → 一意化後 {len(uniq)}件")

    results = []
    csv_rows = [["商品名", "キャッチコピー"] + [f"商品画像名（ALT）{i}" for i in range(1, 21)]]

    for nm in uniq:
        print(f"🧠 生成中: {nm[:30]}...")
        copy, alts = ai_generate(nm, cfg["persona"], cfg["lex"], cfg["market"], cfg["sem"], cfg["tmpl"], cfg["norm"])
        print(f"   ├ copy:{len(copy)}字 / alts:{sum(1 for a in alts if a)}件 例:{(alts[0] or '')[:25]}…")
        results.append({"product_name": nm, "copy": copy, "alts": alts})
        csv_rows.append([nm, copy] + alts)

    now = datetime.now().strftime("%Y%m%d_%H%M")
    jpath = os.path.join(OUT_DIR, f"hybrid_writer_full_{now}.json")
    cpath = os.path.join(OUT_DIR, f"hybrid_writer_preview_{now}.csv")

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)
    with open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"💾 出力完了: {jpath}")
    print(f"🧾 目視用CSV: {cpath}")
    print(f"📊 件数: {len(results)}（全件AI生成／ALT20件統合）")

# ===============================
if __name__ == "__main__":
    main()
import atlas_autosave_core
