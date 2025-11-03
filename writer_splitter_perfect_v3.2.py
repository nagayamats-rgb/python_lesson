# -*- coding: utf-8 -*-
"""
writer_splitter_perfect_v3.2.py
- 全件AI＋知見要約＋3分割（楽天は温存、Yahoo/ALTのみ再生成）
- 入力: ./input.csv（Shift-JIS, ヘッダ行あり, 「商品名」列を縦走査）
- 出力: ./output/ai_writer/{rakuten_copy_*.csv, yahoo_copy_*.csv, alt_text_*.csv, split_full_*.jsonl}
"""

import os, re, csv, glob, json, time, datetime
from pathlib import Path

# ---- OpenAI SDK ----
from openai import OpenAI, OpenAIError

# ---- 進捗バー（tqdmが無い環境でも動作）----
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------- 基本設定 ----------
BASE = Path(".")
AI_OUT = BASE / "output" / "ai_writer"
AI_OUT.mkdir(parents=True, exist_ok=True)

SEM_OUT = BASE / "output" / "semantics"  # 知見JSONがある想定のフォルダ
INPUT_CSV = BASE / "input.csv"

# モデル（GPT-4系で安定運用）
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# 禁則語（画像描写・不適切語・不要表現など）
FORBIDDEN = set([
    "画像", "写真", "フォト", "イメージ", "見た目", "映っている", "映像",
    "競合優位性", "他社優位性", "社外秘", "注意喚起", "※画像はイメージです",
    "クリック", "上の写真", "下の画像",
])

# 句読点セット（ALTの綺麗な短縮に使用）
SENT_END = "。．！!？?；;"

# ---------- ユーティリティ ----------

def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def load_csv_shiftjis(path: Path):
    """Shift-JISでCSV読み込みして全行返す"""
    if not path.exists():
        raise FileNotFoundError(f"入力が見つかりません: {path}")
    rows = []
    with open(path, "r", encoding="cp932", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader]
    return rows

def find_header_and_column(rows, header_name="商品名"):
    """ヘッダ行を見つけ、指定列のインデックスを返す"""
    if not rows:
        return None, None
    header = rows[0]
    if header_name not in header:
        raise KeyError(f"ヘッダに「{header_name}」が見つかりません。")
    return header, header.index(header_name)

def extract_product_names(rows, col_idx):
    """ヘッダの次行から商品名列を走査、空白無視で抽出"""
    names = []
    for r in rows[1:]:
        if col_idx >= len(r):
            continue
        nm = (r[col_idx] or "").strip()
        if nm:
            names.append(nm)
    # 重複除去（順序維持）
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def safe_json_load(path: Path, default=None):
    default = default if default is not None else {}
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def summarize_knowledge():
    """
    任意のローカル知見JSONを読み、プロンプト用に要約テキストを作成（存在しなければ空）
    対応ファイル（存在すれば使う）:
      - lexical_clusters_*.json（語彙/クラスタ）
      - structured_semantics_*.json（概念/構造）
      - styled_persona_*.json（調子/トーン）
      - market_vocab_*.json（市場語彙/流行）
      - normalized_*.json（禁則/正規化）
    """
    # 最新ファイルを拾うヘルパ
    def latest(pattern):
        files = sorted(glob.glob(str(SEM_OUT / pattern)))
        return Path(files[-1]) if files else None

    lex = safe_json_load(latest("lexical_clusters_*.json") or Path(""), default=[])
    sem = safe_json_load(latest("structured_semantics_*.json") or Path(""), default={})
    per = safe_json_load(latest("styled_persona_*.json") or Path(""), default={})
    mar = safe_json_load(latest("market_vocab_*.json") or Path(""), default=[])
    nor = safe_json_load(latest("normalized_*.json") or Path(""), default={})

    # ざっくり要約
def cap_list(xs, key=None, cap=10):
    out = []
    # ★ 修正：dictならvalues()を取ってリスト化
    if isinstance(xs, dict):
        xs = list(xs.values())
    for x in xs[:cap]:
        if isinstance(x, dict) and key:
            v = x.get(key, "")
        else:
            v = str(x)
        v = (v or "").strip()
        if v:
            out.append(v)
    return "、".join(out)

    trend = cap_list(mar, key="vocabulary", cap=12)
    clusters = []
    if isinstance(lex, dict) and "clusters" in lex and isinstance(lex["clusters"], list):
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

def sanitize(text: str) -> str:
    """禁則語の除去＆余計な括弧/連続空白の整理"""
    if not text:
        return ""
    t = text
    # 禁則語削除
    for ng in FORBIDDEN:
        t = t.replace(ng, "")
    # 連続空白の整形
    t = re.sub(r"\s+", " ", t).strip()
    # 不要な末尾記号
    t = re.sub(r"[\/\|\-・\s]+$", "", t)
    return t

def limit_length_ja(s: str, max_chars: int) -> str:
    """全角基準の文字数上限で丸め（ざっくり）"""
    s = s.strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()

def alt_shorten_to_range(s: str, min_len=80, max_len=110):
    """
    ALTの最終整形：
    - まず100〜130で生成 → 句点で文を落として80〜110に収める
    - 句読点での意味単位優先、無ければ安全に丸め
    """
    s = sanitize(s)
    if not s:
        return s
    # 句点で落とし込み
    if len(s) > max_len:
        # 文末マッチ（80〜110に収まる最後の句点）
        cut_idx = None
        for m in re.finditer(r"[。．！？!?]", s):
            pos = m.end()
            if min_len <= pos <= max_len:
                cut_idx = pos
        if cut_idx:
            s = s[:cut_idx]
        else:
            # 次善策：max_lenで丸める
            s = s[:max_len]
    # 下限を満たさない場合は、そのまま（短すぎるケースは再生成対象に回すのが本来）
    return s.strip(" 　、，.")

def find_latest_file(pattern: str):
    files = sorted(glob.glob(str(AI_OUT / pattern)))
    return Path(files[-1]) if files else None

def read_latest_rakuten_or_empty():
    """既存の楽天CSVを読み込み（最も新しいもの）。無ければ空を返す。"""
    latest = find_latest_file("rakuten_copy_*.csv")
    if not latest:
        return {}
    out = {}
    with open(latest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # 期待カラム: 商品名,Rakuten_Copy
        # 壊れていても落ちずに拾える範囲で拾う
        for row in reader:
            nm = (row.get("商品名") or "").strip()
            rk = (row.get("Rakuten_Copy") or row.get("楽天コピー") or "").strip()
            if nm:
                out[nm] = rk
    return out

# ---------- OpenAI呼び出し ----------

def build_client():
    # OPENAI_API_KEY は環境変数で
    return OpenAI()

def call_openai_json(client, model, messages, max_completion_tokens=800):
    """
    JSONモードで呼び出し、content[0].text をJSONとして返す。
    再試行は上位で行う。
    """
    try:
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
    except OpenAIError as e:
        raise
    except Exception as e:
        raise

def prompt_messages(product_name: str, knowledge_text: str):
    """
    Yahoo（25〜30文字）＆ ALT（100〜130文字）を同時JSONで出させるプロンプト。
    画像描写・『競合優位性』などは禁止。
    """
    sys = (
        "あなたは日本語のECコピーライターです。出力は必ずJSONのみ。\n"
        "禁則：画像・写真などの描写語、競合優位性という語、機密・注意喚起表現は使わない。\n"
        "構成ヒント（テンプレ化はしない）：商品スペック→コアコンピタンス→どんな人→どんなシーン→使うと→課題解決/便利さ。\n"
        "Yahooコピーは自然な完結文を**25〜30全角**に厳守。\n"
        "ALTは画像描写禁止で**100〜130全角**、意味が完結した文で。"
    )
    usr = (
        f"商品名: {product_name}\n\n"
        f"ローカル知見:\n{knowledge_text}\n\n"
        "JSONフォーマット:\n"
        "{\n"
        '  "yahoo_copy": "25〜30全角の完結コピー",\n'
        '  "alt": "100〜130全角のALT（画像描写禁止）"\n'
        "}"
    )
    return [
        {"role":"system","content":sys},
        {"role":"user","content":usr}
    ]

def generate_for_product(client, model, product_name, knowledge_text, retries=3):
    """
    単一商品について Yahoo & ALT を生成。
    再試行で空応答やAPIエラーを緩和。
    """
    last_err = None
    for attempt in range(1, retries+1):
        try:
            js = call_openai_json(client, model, prompt_messages(product_name, knowledge_text))
            yc = sanitize(js.get("yahoo_copy","").strip())
            alt = sanitize(js.get("alt","").strip())

            # Yahoo長さ（25〜30）を厳格化
            if not (25 <= len(yc) <= 30):
                # ズレたら安全に丸め or 再試行
                if len(yc) > 30:
                    yc = limit_length_ja(yc, 30)
                elif len(yc) < 25:
                    # 再試行
                    raise ValueError("Yahoo copy too short")

            # ALTは 80〜110 へローカル整形（まずは100〜130生成想定）
            alt_final = alt_shorten_to_range(alt, 80, 110)
            if len(alt_final) < 80:
                # 再試行
                raise ValueError("ALT too short after shorten")

            return yc, alt_final
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    # すべて失敗時
    raise last_err if last_err else RuntimeError("unknown generation error")

# ---------- メイン ----------

def main():
    print("🌸 writer_splitter_perfect_v3.2 実行開始（Yahoo/ALT再生成＋知見要約＋禁則/長さ整形）")

    # 入力読み込み
    rows = load_csv_shiftjis(INPUT_CSV)
    header, col_idx = find_header_and_column(rows, "商品名")
    names = extract_product_names(rows, col_idx)
    print(f"✅ 商品名抽出: {len(names)}件（重複除去済）")

    # 知見要約
    knowledge_text, forbidden_local = summarize_knowledge()
    # ローカル禁則も加える
    if forbidden_local:
        for w in forbidden_local:
            if w: FORBIDDEN.add(str(w))

    # 楽天コピーは既存を温存（無ければ空）
    rakuten_map = read_latest_rakuten_or_empty()

    # 進捗バー
    it = range(len(names))
    if tqdm:
        it = tqdm(it, desc="🧠 商品別AI生成中", ncols=80)

    client = build_client()

    # 出力用
    yahoo_rows = []
    alt_rows = []
    jsonl_path = AI_OUT / f"split_full_{nowstamp()}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        for i in it:
            nm = names[i]
            try:
                yc, alt = generate_for_product(client, MODEL, nm, knowledge_text, retries=3)
            except Exception as e:
                # 失敗時は空欄で落とさず続行（後工程で個別再生成可能）
                yc, alt = "", ""
                if tqdm is None:
                    print(f"⚠️ 生成失敗: {nm[:20]}... => {e}")

            # JSONL記録
            rec = {
                "product_name": nm,
                "yahoo_copy": yc,
                "alt": alt,
                "ts": nowstamp()
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

            yahoo_rows.append({"商品名": nm, "Yahoo_Copy": yc})
            # ALTは横持ち20本 → 今回は ALT共通1枠に格納（共通ALT20を想定する場合は20列に複写も可）
            # 仕様通り「共通ALT20」を1列ではなく “ALT1..ALT20” に展開する場合はここで複写する。
            # ここでは 1商品につき1行で ALT1..ALT20 を同文で埋める（要件：共通ALT20）
            alt_row = {"商品名": nm}
            for k in range(1, 21):
                alt_row[f"ALT{k}"] = alt
            alt_rows.append(alt_row)

    # 既存楽天CSVの最新をコピー or 新規作成
    # → 出力は最新タイムスタンプで必ず作る（中身は最新ファイルがあれば転記、無ければ空）
    rk_rows = []
    for nm in names:
        rk_rows.append({"商品名": nm, "Rakuten_Copy": rakuten_map.get(nm, "")})

    ts = nowstamp()
    yahoo_csv = AI_OUT / f"yahoo_copy_{ts}.csv"
    alt_csv   = AI_OUT / f"alt_text_{ts}.csv"
    rak_csv   = AI_OUT / f"rakuten_copy_{ts}.csv"

    # 書き出し
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
