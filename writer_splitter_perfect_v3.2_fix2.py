# -*- coding: utf-8 -*-
"""
writer_splitter_perfect_v3.2_fix2.py
- 全件AI＋知見要約＋3分割（楽天は温存、Yahoo/ALTのみ再生成）
- 入力: ./input.csv（Shift-JIS, ヘッダ行あり, 「商品名」列を縦走査）
- 出力: ./output/ai_writer/{rakuten_copy_*.csv, yahoo_copy_*.csv, alt_text_*.csv, split_full_*.jsonl}
"""

import os, re, csv, glob, json, time, datetime
from pathlib import Path
from openai import OpenAI, OpenAIError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------- 基本設定 ----------
BASE = Path(".")
AI_OUT = BASE / "output" / "ai_writer"
AI_OUT.mkdir(parents=True, exist_ok=True)
SEM_OUT = BASE / "output" / "semantics"
INPUT_CSV = BASE / "input.csv"

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

FORBIDDEN = set([
    "画像", "写真", "フォト", "イメージ", "見た目", "映っている", "映像",
    "競合優位性", "他社優位性", "社外秘", "注意喚起", "※画像はイメージです",
    "クリック", "上の写真", "下の画像",
])

SENT_END = "。．！!？?；;"

# ---------- 共通ユーティリティ ----------
def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def load_csv_shiftjis(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"入力が見つかりません: {path}")
    with open(path, "r", encoding="cp932", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader]
    return rows

def find_header_and_column(rows, header_name="商品名"):
    header = rows[0]
    if header_name not in header:
        raise KeyError(f"ヘッダに「{header_name}」が見つかりません。")
    return header, header.index(header_name)

def extract_product_names(rows, col_idx):
    names = []
    for r in rows[1:]:
        if col_idx < len(r):
            nm = (r[col_idx] or "").strip()
            if nm:
                names.append(nm)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def sanitize(text: str) -> str:
    if not text:
        return ""
    t = text
    for ng in FORBIDDEN:
        t = t.replace(ng, "")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\/\|\-・\s]+$", "", t)
    return t

def limit_length_ja(s: str, max_chars: int) -> str:
    s = s.strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()

def alt_shorten_to_range(s: str, min_len=80, max_len=110):
    s = sanitize(s)
    if not s:
        return s
    if len(s) > max_len:
        cut_idx = None
        for m in re.finditer(r"[。．！？!?]", s):
            pos = m.end()
            if min_len <= pos <= max_len:
                cut_idx = pos
        if cut_idx:
            s = s[:cut_idx]
        else:
            s = s[:max_len]
    return s.strip(" 　、，.")

def find_latest_file(pattern: str):
    files = sorted(glob.glob(str(AI_OUT / pattern)))
    return Path(files[-1]) if files else None

def read_latest_rakuten_or_empty():
    latest = find_latest_file("rakuten_copy_*.csv")
    if not latest:
        return {}
    out = {}
    with open(latest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nm = (row.get("商品名") or "").strip()
            rk = (row.get("Rakuten_Copy") or row.get("楽天コピー") or "").strip()
            if nm:
                out[nm] = rk
    return out

# ---------- 知見要約 ----------
def summarize_knowledge():
    """ローカル知見JSON群を読み込み、プロンプト注入用テキストを要約生成"""
    try:
        def latest(pattern):
            files = sorted(glob.glob(str(SEM_OUT / pattern)))
            return Path(files[-1]) if files else None

        def safe_load_json(p):
            if not p or not p.exists():
                return {}
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

        def cap_list(xs, key=None, cap=10):
            out = []
            if isinstance(xs, dict):
                xs = list(xs.values())
            if not isinstance(xs, list):
                xs = [xs]
            for x in xs[:cap]:
                if isinstance(x, dict) and key:
                    v = x.get(key, "")
                else:
                    v = str(x)
                v = (v or "").strip()
                if v:
                    out.append(v)
            return "、".join(out)

        lex = safe_load_json(latest("lexical_clusters_*.json"))
        sem = safe_load_json(latest("structured_semantics_*.json"))
        per = safe_load_json(latest("styled_persona_*.json"))
        mar = safe_load_json(latest("market_vocab_*.json"))
        nor = safe_load_json(latest("normalized_*.json"))

        trend = cap_list(mar, key="vocabulary", cap=12)

        clusters = []
        if isinstance(lex, dict) and "clusters" in lex:
            clusters = [c.get("name", "") for c in lex["clusters"] if isinstance(c, dict)]
        elif isinstance(lex, list):
            clusters = [x.get("name","") if isinstance(x, dict) else str(x) for x in lex]
        clusters_str = "、".join([x for x in clusters[:12] if x])

        concepts = []
        if isinstance(sem, dict):
            for k,v in sem.items():
                if isinstance(v, (str, int, float)):
                    concepts.append(f"{k}:{v}")
                elif isinstance(v, list):
                    concepts.append(f"{k}:{'|'.join(map(str,v[:3]))}")
                elif isinstance(v, dict):
                    concepts.append(f"{k}:{'|'.join(list(v.keys())[:3])}")
        concepts_str = "、".join(concepts[:12])

        tone = ""
        if isinstance(per, dict):
            t = per.get("tone", {})
            if isinstance(t, dict):
                tone = "・".join([f"{k}:{v}" for k,v in t.items()])[:120]
            elif isinstance(t, list):
                tone = "・".join(map(str, t))[:120]

        forbidden_local = []
        if isinstance(nor, dict):
            fw = nor.get("forbidden_words", [])
            if isinstance(fw, list):
                forbidden_local = [str(x) for x in fw]

        knowledge = []
        if trend:
            knowledge.append(f"市場語彙/トレンド: {trend}")
        if clusters_str:
            knowledge.append(f"語彙クラスタ: {clusters_str}")
        if concepts_str:
            knowledge.append(f"概念/構造: {concepts_str}")
        if tone:
            knowledge.append(f"推奨トーン: {tone}")

        knowledge_txt = "\n".join(knowledge)
        return knowledge_txt, set(forbidden_local)

    except Exception as e:
        print(f"⚠️ summarize_knowledge() 内でエラー: {e}")
        return "", set()

# ---------- OpenAI呼び出し ----------
def build_client():
    return OpenAI()

def call_openai_json(client, model, messages, max_completion_tokens=800):
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
    )
    txt = (res.choices[0].message.content or "").strip()
    if not txt:
        raise ValueError("Empty content")
    return json.loads(txt)

def prompt_messages(product_name: str, knowledge_text: str):
    sys = (
        "あなたは日本語のECコピーライターです。出力は必ずJSONのみ。\n"
        "禁則：画像・写真などの描写語、競合優位性という語、機密・注意喚起表現は使わない。\n"
        "構成ヒント（テンプレ化はしない）：商品スペック→コアコンピタンス→どんな人→どんなシーン→使うと→課題解決/便利さ。\n"
        "Yahooコピーは自然な完結文を25〜30全角に厳守。\n"
        "ALTは画像描写禁止で100〜130全角、意味が完結した文で。"
    )
    usr = (
        f"商品名: {product_name}\n\n"
        f"ローカル知見:\n{knowledge_text}\n\n"
        "JSONフォーマット:\n"
        "{\n"
        '  \"yahoo_copy\": \"25〜30全角の完結コピー\",\n'
        '  \"alt\": \"100〜130全角のALT（画像描写禁止）\"\n'
        "}"
    )
    return [
        {"role":"system","content":sys},
        {"role":"user","content":usr}
    ]

def generate_for_product(client, model, product_name, knowledge_text, retries=3):
    last_err = None
    for attempt in range(1, retries+1):
        try:
            js = call_openai_json(client, model, prompt_messages(product_name, knowledge_text))
            yc = sanitize(js.get("yahoo_copy","").strip())
            alt = sanitize(js.get("alt","").strip())
            if not (25 <= len(yc) <= 30):
                if len(yc) > 30:
                    yc = limit_length_ja(yc, 30)
                elif len(yc) < 25:
                    raise ValueError("Yahoo copy too short")
            alt_final = alt_shorten_to_range(alt, 80, 110)
            if len(alt_final) < 80:
                raise ValueError("ALT too short after shorten")
            return yc, alt_final
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err if last_err else RuntimeError("unknown generation error")

# ---------- メイン ----------
def main():
    print("🌸 writer_splitter_perfect_v3.2_fix2 実行開始（Yahoo/ALT再生成＋知見要約＋禁則/長さ整形）")
    rows = load_csv_shiftjis(INPUT_CSV)
    header, col_idx = find_header_and_column(rows, "商品名")
    names = extract_product_names(rows, col_idx)
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")

    knowledge_text, forbidden_local = summarize_knowledge()
    if forbidden_local:
        FORBIDDEN.update(forbidden_local)

    rakuten_map = read_latest_rakuten_or_empty()

    it = tqdm(range(len(names)), desc="🧠 商品別AI生成中", ncols=80) if tqdm else range(len(names))
    client = build_client()

    yahoo_rows, alt_rows = [], []
    jsonl_path = AI_OUT / f"split_full_{nowstamp()}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        for i in it:
            nm = names[i]
            try:
                yc, alt = generate_for_product(client, MODEL, nm, knowledge_text)
            except Exception as e:
                yc, alt = "", ""
                if tqdm is None:
                    print(f"⚠️ 生成失敗: {nm[:20]}... => {e}")
            rec = {"product_name": nm, "yahoo_copy": yc, "alt": alt, "ts": nowstamp()}
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            yahoo_rows.append({"商品名": nm, "Yahoo_Copy": yc})
            alt_row = {"商品名": nm}
            for k in range(1, 21):
                alt_row[f"ALT{k}"] = alt
            alt_rows.append(alt_row)

    rk_rows = [{"商品名": nm, "Rakuten_Copy": rakuten_map.get(nm, "")} for nm in names]

    ts = nowstamp()
    yahoo_csv = AI_OUT / f"yahoo_copy_{ts}.csv"
    alt_csv   = AI_OUT / f"alt_text_{ts}.csv"
    rak_csv   = AI_OUT / f"rakuten_copy_{ts}.csv"

    with open(yahoo_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["商品名", "Yahoo_Copy"])
        w.writeheader()
        w.writerows(yahoo_rows)

    with open(alt_csv, "w", encoding="utf-8", newline="") as f:
        fields = ["商品名"] + [f"ALT{k}" for k in range(1,21)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(alt_rows)

    with open(rak_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["商品名", "Rakuten_Copy"])
        w.writeheader()
        w.writerows(rk_rows)

    print("✅ 出力完了:")
    print(f"   - 楽天: {rak_csv}")
    print(f"   - Yahoo: {yahoo_csv}")
    print(f"   - ALT20: {alt_csv}")
    print(f"   - JSONL: {jsonl_path}")
    print("✅ 共通ALT20は『alt_text_*.csv』にALT1〜ALT20へ横持ちで複写します。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
