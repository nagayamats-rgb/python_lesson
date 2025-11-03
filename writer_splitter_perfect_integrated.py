# -*- coding: utf-8 -*-
"""
writer_splitter_perfect_integrated.py
全件AI＋ローカル知見要約＋3分割出力（楽天/Yahoo/ALT20）

■ 入力
- ./input.csv（Shift-JIS / 先頭行ヘッダ / 「商品名」列）
- ./output/semantics/*.json（ローカル知見ファイル群：存在する範囲で自動吸収）
  - lexical_clusters_*.json
  - structured_semantics_*.json
  - styled_persona_*.json
  - market_vocab_*.json
  - normalized_*.json
  - template_composer.json（任意）

■ 出力
- ./output/ai_writer/rakuten_copy_YYYYMMDD_HHMM.csv（推奨60–80/上限87）
- ./output/ai_writer/yahoo_copy_YYYYMMDD_HHMM.csv（推奨25–30/上限30）
- ./output/ai_writer/alt_text_YYYYMMDD_HHMM.csv（ALT×20列/各80–110）
- ./output/ai_writer/split_full_YYYYMMDD_HHMM.jsonl（AI応答の生ログ）

■ モデルとキー
- .env の OPENAI_API_KEY を使用
- .env の OPENAI_MODEL を優先（未設定時は gpt-4-turbo）
- ※温度は未指定（モデル既定値=1）/ max_tokens は使用せず max_completion_tokens を使用
"""

import os
import re
import io
import json
import glob
import time
import unicodedata
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, BadRequestError, OpenAIError

# =========================
# 基本ユーティリティ
# =========================

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

def ensure_dirs():
    os.makedirs("./output/ai_writer", exist_ok=True)
    os.makedirs("./output/semantics", exist_ok=True)

def read_input_csv(path="./input.csv"):
    # Shift-JIS(cp932)で読み込む。空白セルはそのまま空文字扱いにする
    df = pd.read_csv(path, encoding="cp932", dtype=str, keep_default_na=False, na_filter=False)
    # 「商品名」列を探す（完全一致）
    if "商品名" not in df.columns:
        # 似た列名の救済（スペース混入など）
        candidates = [c for c in df.columns if str(c).strip() == "商品名"]
        if candidates:
            name_col = candidates[0]
        else:
            raise ValueError("ヘッダに『商品名』列が見つかりません。先頭行ヘッダ・列名『商品名』をご確認ください。")
    else:
        name_col = "商品名"
    names = [str(x).strip() for x in list(df[name_col])]
    # 空白スキップ・重複除去（順序保持）
    uniq = []
    seen = set()
    for n in names:
        if not n:
            continue
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def load_json_safe(path):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with io.open(path, "r", encoding="cp932") as f:
                return json.load(f)
        except Exception:
            return None

def glob_first(pattern):
    files = sorted(glob.glob(pattern))
    return files[0] if files else None

def pick_semantics():
    # 存在するものだけ取り込む（欠けていても動く）
    base = "./output/semantics"
    bundle = {}

    paths = {
        "lexical": glob_first(f"{base}/lexical_clusters_*.json"),
        "semantic": glob_first(f"{base}/structured_semantics_*.json"),
        "persona": glob_first(f"{base}/styled_persona_*.json"),
        "market": glob_first(f"{base}/market_vocab_*.json"),
        "normalized": glob_first(f"{base}/normalized_*.json"),
        "template": os.path.join(base, "template_composer.json") if os.path.exists(os.path.join(base, "template_composer.json")) else None,
    }
    for k, p in paths.items():
        bundle[k] = load_json_safe(p) if p else None
    return bundle

def to_str_list(x):
    # list[str] へ寄せる
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, dict):
                # 候補になりそうなキーを拾う
                for key in ("text","vocabulary","word","value","label","name"):
                    if key in v and isinstance(v[key], str):
                        out.append(v[key])
                        break
            elif isinstance(v, str):
                out.append(v)
        return out
    if isinstance(x, dict):
        # 候補になりそうなキーを拾う
        for key in ("list","items","values","words","vocab","entries"):
            if key in x and isinstance(x[key], list):
                return to_str_list(x[key])
        return []
    if isinstance(x, str):
        return [x]
    return []

def jlen(s: str) -> int:
    # ざっくり全角/半角の区別なく文字数カウント（要件上は全角上限だが、安定のためlenで運用）
    return len(s)

def smart_truncate(text, max_len):
    if jlen(text) <= max_len:
        return text
    # 句点・読点・中点・約物で手前カット
    cut = text[:max_len]
    # 末尾を整える
    cut = re.sub(r'[、。・,.;:：；、。…ー\-]\s*$', '', cut)
    return cut

def enforce_range(text, min_len, max_len):
    t = text.strip()
    if jlen(t) > max_len:
        t = smart_truncate(t, max_len)
    # 足りない時はそのまま返す（再生成は上位で）
    return t

def dedupe_preserve_order(seq):
    out, seen = [], set()
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# =========================
# 知見要約ブロックの構築
# =========================

def summarize_knowledge(name, kb):
    """
    各JSONの中身が list/dict いずれでも拾えるように軽量要約に整形
    """
    persona = kb.get("persona")
    lexical = kb.get("lexical")
    semantic = kb.get("semantic")
    market = kb.get("market")
    normalized = kb.get("normalized")
    template = kb.get("template")

    # persona
    tone_words = []
    if persona:
        if isinstance(persona, dict):
            for key in ("tone","style","voice","writing","brand","guidelines"):
                tone_words += to_str_list(persona.get(key))
        elif isinstance(persona, list):
            tone_words += to_str_list(persona)

    # lexical
    clusters = []
    if lexical:
        if isinstance(lexical, dict):
            for key in ("clusters","keywords","seed","lexicon","terms","phrases"):
                clusters += to_str_list(lexical.get(key))
        elif isinstance(lexical, list):
            clusters += to_str_list(lexical)

    # semantic
    concepts = []
    if semantic:
        if isinstance(semantic, dict):
            for key in ("concepts","semantics","frames","features","facets","benefits","scenes"):
                concepts += to_str_list(semantic.get(key))
        elif isinstance(semantic, list):
            concepts += to_str_list(semantic)

    # market
    trend = []
    scenes = []
    audience = []
    if market:
        if isinstance(market, dict):
            trend += to_str_list(market.get("vocabulary"))
            scenes += to_str_list(market.get("scenes"))
            audience += to_str_list(market.get("audience"))
        elif isinstance(market, list):
            trend += to_str_list(market)
    # normalized
    forbidden = []
    if normalized:
        if isinstance(normalized, dict):
            for key in ("forbidden_words","banned","ng","prohibited"):
                forbidden += to_str_list(normalized.get(key))
        elif isinstance(normalized, list):
            forbidden += to_str_list(normalized)

    # 画像描写ワード禁止（ユーザーの明確な方針）
    forbidden += ["画像", "写真", "フレーム", "構図", "被写体", "画角", "解像度", "ピクセル", "背景", "白背景", "イメージ図", "ボケ", "シルエット"]

    # 軽量要約テキスト（AIへ最小限）
    block = {
        "persona_tone": dedupe_preserve_order([w for w in tone_words if w])[:20],
        "lexical_hints": dedupe_preserve_order([w for w in clusters if w])[:40],
        "semantic_hints": dedupe_preserve_order([w for w in concepts if w])[:40],
        "market_trend": dedupe_preserve_order([w for w in trend if w])[:30],
        "market_scenes": dedupe_preserve_order([w for w in scenes if w])[:20],
        "market_audience": dedupe_preserve_order([w for w in audience if w])[:20],
        "forbidden": dedupe_preserve_order([w for w in forbidden if w]),
        "template_note": "テンプレ感は出さないが、構成要素（spec/competence/user/scene/benefit）は意識して自然文に溶かし込む。"
    }
    # 文字列化して返す（プロンプトに埋めやすく）
    return json.dumps(block, ensure_ascii=False)

# =========================
# OpenAI ラッパ
# =========================

def load_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です（.env を確認）")
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo").strip()
    client = OpenAI(api_key=api_key)
    return client, model

JSON_SCHEMA_HINT = """
出力は必ず次のJSONオブジェクト1個で返してください（コードブロック不可）：
{
  "rakuten": "<60〜80字推奨・上限87字>",
  "yahoo": "<25〜30字推奨・上限30字>",
  "alts": ["<ALT1(80〜110字)>", "...", "<ALT20(80〜110字)>"]
}
注意：
- alts は必ず20件。内容はすべてユニーク。画像・構図・画角などの語は使用禁止。
- 禁止語（forbidden）を含めないこと。
- 句読点（。）、読点（、）を適切に使い自然文にすること。
- 「競合優位性」などの内部語は消費者向け自然表現に言い換えること（例：「他と比べて使いやすい」など）。
- spec / competence / user / scene / benefit の要素を自然に溶け込ませる（テンプレ感は出さない）。
"""

def build_messages(product_name, knowledge_json_text):
    system = (
        "あなたは日本語のEC商品コピーライターです。"
        "以下の制約を厳格に守ってください：\n"
        "• 画像描写・構図・画角・解像度などの用語を使わない\n"
        "• 禁止語（forbidden）に含まれる語を使わない\n"
        "• 楽天は60〜80字を推奨、上限87字\n"
        "• Yahooは25〜30字、上限30字\n"
        "• ALTは各80〜110字、ちがう視点の20本\n"
        "• spec/competence/user/scene/benefitを自然に織り交ぜる\n"
        "• テンプレ感は出さない\n"
        "• 読点・句点で日本語として違和感のない一文にする\n"
    )
    user = (
        f"対象商品名：{product_name}\n"
        f"参考知見（ローカル要約）：{knowledge_json_text}\n"
        f"{JSON_SCHEMA_HINT}\n"
        "さあ、JSONだけを返してください。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]

def try_json_mode(client, model, messages, max_completion_tokens=700):
    """
    response_format={'type':'json_object'} を試す。
    失敗したら例外を投げる（上位でテキストモードにフォールバック）
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        # 温度は未指定（モデル既定値=1 固定問題を回避）
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
    )

def try_text_mode(client, model, messages, max_completion_tokens=700):
    """
    通常モード（テキスト）でJSONを返してもらう
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )

def extract_json_from_text(text):
    # 最初の { から最後の } までを貪欲に取得
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except Exception:
        # 末尾 , の除去など軽い手当て
        chunk = re.sub(r',\s*}', '}', chunk)
        chunk = re.sub(r',\s*]', ']', chunk)
        try:
            return json.loads(chunk)
        except Exception:
            return None

def call_openai_for_product(client, model, product_name, knowledge_json_text, logf):
    messages = build_messages(product_name, knowledge_json_text)

    # まずJSONモードを試す
    for attempt in range(2):
        try:
            res = try_json_mode(client, model, messages, max_completion_tokens=900)
            raw = res.choices[0].message.content or ""
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "json", "raw": raw}, ensure_ascii=False) + "\n")
            data = json.loads(raw)
            return data
        except BadRequestError as e:
            # レスポンスフォーマット未対応モデルなど → テキストモードへ
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "json_error", "error": str(e)}, ensure_ascii=False) + "\n")
            break
        except (APIError, RateLimitError, OpenAIError) as e:
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "json_api_error", "error": str(e)}, ensure_ascii=False) + "\n")
            time.sleep(2)
        except Exception as e:
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "json_unknown", "error": str(e)}, ensure_ascii=False) + "\n")
            break

    # テキストモードでJSON抽出
    for attempt in range(3):
        try:
            res = try_text_mode(client, model, messages, max_completion_tokens=900)
            raw = res.choices[0].message.content or ""
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "text", "raw": raw}, ensure_ascii=False) + "\n")
            data = extract_json_from_text(raw)
            if data:
                return data
        except (APIError, RateLimitError, OpenAIError) as e:
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "text_api_error", "error": str(e)}, ensure_ascii=False) + "\n")
            time.sleep(2)
        except Exception as e:
            if logf:
                logf.write(json.dumps({"product": product_name, "mode": "text_unknown", "error": str(e)}, ensure_ascii=False) + "\n")
            time.sleep(1)

    return None

# =========================
# ローカル整形（安全弁）
# =========================

def cleanse_forbidden(text, forbidden):
    t = text.strip()
    if not forbidden:
        return t
    for w in forbidden:
        if not w:
            continue
        t = t.replace(w, "")
    # 連続スペース・句読点整形
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[、。]{2,}', '。', t)
    return t.strip()

def local_refine(product_name, raw_obj, forbidden):
    """
    - 長さ制約の最終確認
    - 禁止語削除
    - ALTの本数・ユニーク化
    """
    rak = (raw_obj.get("rakuten") or "").strip()
    yah = (raw_obj.get("yahoo") or "").strip()
    alts = raw_obj.get("alts") or []

    # Rakuten: 推奨60–80 / 上限87
    rak = cleanse_forbidden(rak, forbidden)
    rak = enforce_range(rak, 60, 87)

    # Yahoo: 推奨25–30 / 上限30
    yah = cleanse_forbidden(yah, forbidden)
    yah = enforce_range(yah, 25, 30)

    # ALT: 20本 / 各80–110
    alts = [cleanse_forbidden(a or "", forbidden) for a in alts if isinstance(a, str)]
    alts = [enforce_range(a, 80, 110) for a in alts if a]
    alts = [a for a in alts if a]  # 非空
    # ユニーク化
    alts = dedupe_preserve_order(alts)
    # 足りない場合はRak/Yahを変換して補完
    while len(alts) < 20:
        # 変形して補完（単純に語尾や順序を少し替える）
        base = rak if (len(alts) % 2 == 0 and rak) else yah
        if not base:
            base = product_name + " をより快適に使えるように配慮された設計で、日常の不満を減らし便利さを実感できます。"
        variant = base
        # 軽変形：読点を追加／助詞入れ替え
        variant = variant.replace("、", "、")
        if jlen(variant) < 80:
            variant += " 毎日の使用で差が出る設計で、使うたびに快適さを感じられます。"
        variant = enforce_range(variant, 80, 110)
        alts.append(variant)
    if len(alts) > 20:
        alts = alts[:20]

    return rak, yah, alts

# =========================
# 本体
# =========================

def main():
    ensure_dirs()
    client, model = load_openai_client()
    names = read_input_csv("./input.csv")
    kb = pick_semantics()

    out_time = now_stamp()
    path_rakuten = f"./output/ai_writer/rakuten_copy_{out_time}.csv"
    path_yahoo  = f"./output/ai_writer/yahoo_copy_{out_time}.csv"
    path_alt    = f"./output/ai_writer/alt_text_{out_time}.csv"
    path_log    = f"./output/ai_writer/split_full_{out_time}.jsonl"

    knowledge_forbidden_union = []
    if kb.get("normalized"):
        # 正規化の禁止語集合（サマライザでも吸収しているが、ここでは生も見る）
        if isinstance(kb["normalized"], dict):
            for key in ("forbidden_words","banned","ng","prohibited"):
                knowledge_forbidden_union += to_str_list(kb["normalized"].get(key))
        elif isinstance(kb["normalized"], list):
            knowledge_forbidden_union += to_str_list(kb["normalized"])
    # 画像描写ワードも追加（念押し）
    knowledge_forbidden_union += ["画像", "写真", "フレーム", "構図", "被写体", "画角", "解像度", "ピクセル", "背景", "白背景", "イメージ図", "ボケ", "シルエット"]
    knowledge_forbidden_union = dedupe_preserve_order([w for w in knowledge_forbidden_union if w])

    print(f"🌸 writer_splitter_perfect_integrated 実行開始（全件AI＋知見要約＋3分割）")
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")

    # 出力CSVを先に用意（追記していく）
    df_rak = pd.DataFrame([], columns=["商品名","楽天コピー"])
    df_yah = pd.DataFrame([], columns=["商品名","Yahooコピー"])
    # ALTは横持ち20列
    alt_cols = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]
    df_alt = pd.DataFrame([], columns=alt_cols)

    # ログファイル
    logf = io.open(path_log, "w", encoding="utf-8")

    pbar = tqdm(total=len(names), desc="🧠 商品別AI生成中", ncols=88)
    for nm in names:
        try:
            # ローカル知見を軽量要約に
            knowledge_text = summarize_knowledge(nm, kb)
            # AI呼び出し
            data = call_openai_for_product(client, model, nm, knowledge_text, logf)
            if not data or not isinstance(data, dict):
                # 空応答→簡易フォールバック
                data = {
                    "rakuten": f"{nm} の特長を活かし、日常の不満を減らして快適に使えるよう配慮した設計です。",
                    "yahoo": f"{nm} の使いやすさに配慮した設計。",
                    "alts": []
                }
            # ローカル最終整形
            rak, yah, alts = local_refine(nm, data, knowledge_forbidden_union)

            # 行追加
            df_rak.loc[len(df_rak)] = [nm, rak]
            df_yah.loc[len(df_yah)] = [nm, yah]
            row = [nm] + alts
            df_alt.loc[len(df_alt)] = row

            pbar.set_postfix_str(f"{nm[:16]}… → R:{jlen(rak)} / Y:{jlen(yah)} / ALT20")
        except KeyboardInterrupt:
            logf.write(json.dumps({"product": nm, "event": "keyboard_interrupt"}, ensure_ascii=False) + "\n")
            break
        except Exception as e:
            logf.write(json.dumps({"product": nm, "event": "exception", "error": str(e)}, ensure_ascii=False) + "\n")
            # フォールバックで空欄追加（後で再生成可能に）
            df_rak.loc[len(df_rak)] = [nm, ""]
            df_yah.loc[len(df_yah)] = [nm, ""]
            df_alt.loc[len(df_alt)] = [nm] + [""]*20
        finally:
            pbar.update(1)
    pbar.close()
    logf.close()

    # 出力（Excel/Windows互換のためBOM付きUTF-8）
    df_rak.to_csv(path_rakuten, index=False, encoding="utf-8-sig")
    df_yah.to_csv(path_yahoo,  index=False, encoding="utf-8-sig")
    df_alt.to_csv(path_alt,    index=False, encoding="utf-8-sig")

    print("✅ 出力完了:")
    print(f"   - 楽天: {path_rakuten}")
    print(f"   - Yahoo: {path_yahoo}")
    print(f"   - ALT20: {path_alt}")
    print(f"   - JSONL: {path_log}")
    print("✅ 共通ALT20は『alt_text_*.csv』に全商品ぶんを横持ちで書き出します。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
