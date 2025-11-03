# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v5.5
GPT-5完全対応（json_schema構造出力＋ALT20件）
"""

import os, csv, json, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 初期設定
# ---------------------------------------------------------
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
OUT_DIR = os.path.join(BASE_DIR, "output/ai_writer")
os.makedirs(OUT_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ENCODING_IN = "cp932"

# ファイルパス
PATH_LEXICAL = os.path.join(SEM_DIR, "lexical_clusters_20251030_223013.json")
PATH_MARKET  = os.path.join(SEM_DIR, "market_vocab_20251030_201906.json")
PATH_SEMANT  = os.path.join(SEM_DIR, "structured_semantics_20251030_224846.json")
PATH_PERSONA = os.path.join(SEM_DIR, "styled_persona_20251031_0031.json")
PATH_NORMAL  = os.path.join(SEM_DIR, "normalized_20251031_0039.json")
PATH_TEMPLATE = os.path.join(SEM_DIR, "template_composer.json")

# ---------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------
def sanitize(s):
    s = s.replace("\u3000", " ")
    return re.sub(r"\s+", " ", s.strip())

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default or {}

def ensure_dict_or_first(x):
    if isinstance(x, list):
        return x[0] if x else {}
    elif isinstance(x, dict):
        return x
    else:
        return {}

# ---------------------------------------------------------
# AI生成ロジック
# ---------------------------------------------------------
def ai_generate(name, persona_cfg, lexical_cfg, market_cfg, sem_cfg, tmpl_cfg, norm_cfg):
    persona = ensure_dict_or_first(persona_cfg)
    sem     = ensure_dict_or_first(sem_cfg)
    lex     = ensure_dict_or_first(lexical_cfg)
    norm    = ensure_dict_or_first(norm_cfg)

    context = {
        "product_name": name,
        "persona": persona,
        "semantics": sem,
        "lexical": lex,
        "market": market_cfg.get("keywords", []),
        "templates": tmpl_cfg.get("templates", []),
        "forbidden_words": norm.get("forbidden_words", [])
    }

    # GPT-5 構造出力
    res = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "あなたは日本語ECサイトのコピーライターAIです。"},
            {
                "role": "user",
                "content": (
                    "以下のJSON構造情報をもとに、"
                    "40〜60文字の魅力的なキャッチコピーと、"
                    "SEO最適化されたALTテキスト20件を生成してください。"
                    "ALTは80〜110文字で、商品の特徴・用途・キーワードを自然に含めてください。\n\n"
                    + json.dumps(context, ensure_ascii=False, indent=2)
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ProductCopy",
                "schema": {
                    "type": "object",
                    "properties": {
                        "copy": {
                            "type": "string",
                            "description": "キャッチコピー（40〜60文字）"
                        },
                        "alts": {
                            "type": "array",
                            "description": "ALTテキスト20件（80〜110文字）",
                            "items": {"type": "string"},
                            "minItems": 20,
                            "maxItems": 20
                        }
                    },
                    "required": ["copy", "alts"]
                }
            }
        },
        max_completion_tokens=1000
    )

    msg = res.choices[0].message
    if not getattr(msg, "content", None):
        print("⚠️ 応答なし／拒否")
        return "生成失敗", ["生成失敗" for _ in range(20)]

    try:
        data = json.loads(msg.content)
        copy_t = data.get("copy", "").strip() or "生成失敗"
        alts = data.get("alts", [])
        if not alts or len(alts) < 20:
            alts += [""] * (20 - len(alts))
        return copy_t, alts[:20]
    except Exception:
        return "生成失敗", ["生成失敗" for _ in range(20)]

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    print("🌸 Hybrid AI Writer v5.5 実行開始（GPT-5構造出力対応）")

    cfg = {
        "lex": load_json(PATH_LEXICAL),
        "market": load_json(PATH_MARKET),
        "sem": load_json(PATH_SEMANT),
        "persona": load_json(PATH_PERSONA),
        "tmpl": load_json(PATH_TEMPLATE),
        "norm": load_json(PATH_NORMAL),
    }

    # 商品名抽出
    with open(INPUT_CSV, "r", encoding=ENCODING_IN, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    name_idx = header.index("商品名")
    names = [sanitize(r[name_idx]) for r in rows[1:] if len(r) > name_idx and sanitize(r[name_idx])]
    uniq = list(dict.fromkeys(names))
    print(f"✅ 商品名抽出: {len(names)}件 → 一意化後 {len(uniq)}件")

    results = []
    csv_header = ["商品名", "キャッチコピー"] + [f"商品画像名（ALT）{i}" for i in range(1, 21)]
    csv_rows = [csv_header]

    for nm in uniq:
        print(f"🧠 AI生成中: {nm[:30]}...")
        copy_t, alts = ai_generate(nm, cfg["persona"], cfg["lex"], cfg["market"], cfg["sem"], cfg["tmpl"], cfg["norm"])
        csv_rows.append([nm, copy_t] + alts)
        results.append({"product_name": nm, "copy": copy_t, "alts": alts})

    now = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join(OUT_DIR, f"hybrid_writer_full_{now}.json")
    csv_path = os.path.join(OUT_DIR, f"hybrid_writer_preview_{now}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"💾 出力完了: {json_path}")
    print(f"🧾 目視用CSV: {csv_path}")
    print(f"📊 件数: {len(results)}（全件AI生成／ALT20件構造出力）")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
import atlas_autosave_core
