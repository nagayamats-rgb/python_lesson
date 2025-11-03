# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v5.6.1
GPT-5安全モード：ALT10×2分割生成＋フォールバック
"""

import os, csv, json, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
SEM_DIR = os.path.join(BASE_DIR, "output/semantics")
OUT_DIR = os.path.join(BASE_DIR, "output/ai_writer")
os.makedirs(OUT_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ENCODING_IN = "cp932"

# --- 中間ファイル ---
PATH_LEXICAL = os.path.join(SEM_DIR, "lexical_clusters_20251030_223013.json")
PATH_MARKET  = os.path.join(SEM_DIR, "market_vocab_20251030_201906.json")
PATH_SEMANT  = os.path.join(SEM_DIR, "structured_semantics_20251030_224846.json")
PATH_PERSONA = os.path.join(SEM_DIR, "styled_persona_20251031_0031.json")
PATH_NORMAL  = os.path.join(SEM_DIR, "normalized_20251031_0039.json")
PATH_TEMPLATE = os.path.join(SEM_DIR, "template_composer.json")

def sanitize(s):
    return re.sub(r"\s+", " ", s.replace("\u3000"," ").strip())

def load_json(p, d=None):
    try:
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except: return d or {}

def short(x):
    """プロンプト軽量化"""
    if isinstance(x, list): return x[:5]
    if isinstance(x, dict): return {k: x[k] for k in list(x.keys())[:5]}
    return x

# --- GPT呼び出し ---
def ai_generate(name, persona, lexical, market, sem, tmpl, norm):
    """2段階ALT生成 + フォールバック"""
    # 🔧 forbidden_wordsを安全に取得
    if isinstance(norm, dict):
        forbidden_words = norm.get("forbidden_words", [])
    elif isinstance(norm, list):
        forbidden_words = norm
    else:
        forbidden_words = []

    base_prompt = f"""
あなたは日本語ECコピーライターです。
商品名: {name}
文体: {json.dumps(short(persona),ensure_ascii=False)}
市場語彙: {json.dumps(short(market.get('keywords',[])),ensure_ascii=False)}
意味語群: {json.dumps(short(sem),ensure_ascii=False)}
禁則語: {json.dumps(forbidden_words,ensure_ascii=False)}

出力形式:
{{
  "copy": "40〜60文字のキャッチコピー",
  "alts": ["ALT文1", "ALT文2", ...10件]
}}
ALTは80〜110文字で、自然かつSEO的に有効な説明文にしてください。
"""

    # --- copy + ALT(1-10) ---
    try:
        res1 = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role":"system","content":"あなたは日本語ECコピーの専門家です。"},
                {"role":"user","content":base_prompt}
            ],
            response_format={
                "type":"json_schema",
                "json_schema":{
                    "name":"CopyAltBlock",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "copy":{"type":"string"},
                            "alts":{
                                "type":"array","items":{"type":"string"},"maxItems":10
                            }
                        },
                        "required":["copy","alts"]
                    }
                }
            },
            max_completion_tokens=600
        )
        data1=json.loads(res1.choices[0].message.content)
        copy=data1.get("copy","生成失敗").strip()
        alts1=data1.get("alts",[])
    except Exception:
        copy,alts1="生成失敗",[]

    # --- ALT(11-20)追加 ---
    alt_prompt=f"商品名: {name}\n上記と異なる内容のALTテキストをさらに10件生成してください。"
    try:
        res2 = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role":"system","content":"あなたは日本語ECコピーライターです。"},
                {"role":"user","content":alt_prompt}
            ],
            response_format={
                "type":"json_schema",
                "json_schema":{
                    "name":"AltBlock",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "alts":{
                                "type":"array","items":{"type":"string"},"maxItems":10
                            }
                        },
                        "required":["alts"]
                    }
                }
            },
            max_completion_tokens=400
        )
        data2=json.loads(res2.choices[0].message.content)
        alts2=data2.get("alts",[])
    except Exception:
        alts2=[]

    alts=(alts1+alts2)[:20]
    if len(alts)<20: alts+=[""]*(20-len(alts))
    return copy,alts

def main():
    print("🌸 Hybrid AI Writer v5.6.1 実行開始（安全モード／分割ALT生成）")

    cfg={
        "lex":load_json(PATH_LEXICAL),
        "market":load_json(PATH_MARKET),
        "sem":load_json(PATH_SEMANT),
        "persona":load_json(PATH_PERSONA),
        "tmpl":load_json(PATH_TEMPLATE),
        "norm":load_json(PATH_NORMAL)
    }

    # --- CSV読込 ---
    with open(INPUT_CSV,"r",encoding=ENCODING_IN) as f:
        rows=list(csv.reader(f))
    header=rows[0]; name_idx=header.index("商品名")
    names=[sanitize(r[name_idx]) for r in rows[1:] if len(r)>name_idx and sanitize(r[name_idx])]
    uniq=list(dict.fromkeys(names))
    print(f"✅ 商品名抽出: {len(names)}件 → 一意化後 {len(uniq)}件")

    results=[]; csv_rows=[["商品名","キャッチコピー"]+[f"商品画像名（ALT）{i}" for i in range(1,21)]]

    for nm in uniq:
        print(f"🧠 生成中: {nm[:30]}...")
        copy,alts=ai_generate(nm,cfg["persona"],cfg["lex"],cfg["market"],cfg["sem"],cfg["tmpl"],cfg["norm"])
        results.append({"product_name":nm,"copy":copy,"alts":alts})
        csv_rows.append([nm,copy]+alts)

    now=datetime.now().strftime("%Y%m%d_%H%M")
    jpath=os.path.join(OUT_DIR,f"hybrid_writer_full_{now}.json")
    cpath=os.path.join(OUT_DIR,f"hybrid_writer_preview_{now}.csv")

    with open(jpath,"w",encoding="utf-8") as f: json.dump({"items":results},f,ensure_ascii=False,indent=2)
    with open(cpath,"w",encoding="utf-8-sig",newline="") as f: csv.writer(f).writerows(csv_rows)

    print(f"💾 出力完了: {jpath}")
    print(f"🧾 目視用CSV: {cpath}")
    print(f"📊 件数: {len(results)}（全件AI生成／ALT20件統合）")

if __name__=="__main__":
    main()
import atlas_autosave_core
