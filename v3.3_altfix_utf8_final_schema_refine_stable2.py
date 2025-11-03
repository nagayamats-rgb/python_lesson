# -*- coding: utf-8 -*-
"""
v3.3_altfix_utf8_final_schema_refine_stable2.py
ALT（楽天共通ALT20）を“長文→ローカル整形”で安定生成する専用スクリプト。
- 入力: /Users/tsuyoshi/Desktop/python_lesson/rakuten.csv（UTF-8, 先頭行ヘッダ,「商品名」列）
- 出力: /Users/tsuyoshi/Desktop/python_lesson/output/ai_writer/alt_text_refined_final_stable.csv
- OpenAI: .env を自動読込（OPENAI_API_KEY / OPENAI_MODEL 任意）
- 応答形式: response_format="text"（JSONは使わない）
- 生成: 100〜130字をAIに書かせ、ローカルで80〜110字に自然整形（句点だけ禁止）
- 20件/商品を保証（少ない場合はローカル変換で補完）
"""

import os
import re
import csv
import json
import time
import glob
from typing import List, Tuple, Dict, Any

# ========= 0) .env ローダ（依存なし） =========
def load_env_from_local_env():
    """
    シェルの新しいウィンドウで OPENAI_API_KEY が消える問題に備え、
    カレント配下 or プロジェクト直下の .env を手動で読む。
    """
    candidates = [
        ".env",
        "/Users/tsuyoshi/Desktop/python_lesson/.env",
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if "=" in line and not line.strip().startswith("#"):
                            k, v = line.strip().split("=", 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
        except Exception:
            pass

load_env_from_local_env()

# ========= 1) OpenAI クライアント =========
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("openai ライブラリが見つかりません。`pip install openai` を実行してください。") from e

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")

client = OpenAI()  # APIキーは環境変数から

# ========= 2) 入出力パス =========
INPUT_RAKUTEN = "/Users/tsuyoshi/Desktop/python_lesson/rakuten.csv"  # UTF-8
OUT_DIR = "/Users/tsuyoshi/Desktop/python_lesson/output/ai_writer"
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUT_DIR, "alt_text_refined_final_stable.csv")

# ========= 3) モデル・パラメタ =========
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# ※ 一部モデルは非対応パラメタがあるため、温度などは“未指定”で呼ぶ（安全）
MAX_COMPLETION_TOKENS = 1000  # 長文許容
RETRY = 3
RETRY_WAIT = 3

# ========= 4) ローカル知見の読み込み（任意ファイルがあれば吸い上げ） =========
SEMANTICS_DIR = "/Users/tsuyoshi/Desktop/python_lesson/output/semantics"

def safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_local_knowledge() -> Tuple[str, List[str], Dict[str, str]]:
    """
    可能な限り /output/semantics 配下から知見を要約。
    - forbidden_words: 禁止語集約
    - synonyms_map: 置換して差分バリエーションを作る簡易語彙
    - knowledge_text: AIプロンプトに埋め込むダイジェスト
    """
    forbidden: List[str] = []
    synonyms_map: Dict[str, str] = {}

    if os.path.isdir(SEMANTICS_DIR):
        for fp in glob.glob(os.path.join(SEMANTICS_DIR, "*.json")):
            data = safe_load_json(fp)
            if not data:
                continue
            # 代表的な構造を想定して抽出
            if isinstance(data, dict):
                if "forbidden_words" in data and isinstance(data["forbidden_words"], list):
                    forbidden.extend([str(x) for x in data["forbidden_words"] if isinstance(x, (str, int))])
                if "synonyms" in data and isinstance(data["synonyms"], dict):
                    for k, v in data["synonyms"].items():
                        if isinstance(k, str) and isinstance(v, str):
                            synonyms_map[k] = v
            # リスト形式も許容（各要素が dict の想定）
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        if "forbidden_words" in row and isinstance(row["forbidden_words"], list):
                            forbidden.extend([str(x) for x in row["forbidden_words"] if isinstance(x, (str, int))])

    # 画像描写NG・メタ表現NGなど最低限
    base_forbidden = [
        "画像", "写真", "イメージ", "見た目", "こちら", "当店", "競合", "競合優位性",
        "売上No.1", "No.1", "ランキング上位", "クリック", "リンク"
    ]
    forbidden.extend(base_forbidden)
    # 正規化・一意化
    forb = sorted(set([str(x).strip() for x in forbidden if str(x).strip()]))

    # 知見テキスト（簡潔に）
    knowledge_text = (
        "・画像描写は禁止。商品スペック／コアコンピタンス／想定ユーザー／使用シーン／ベネフィットを無理なく含める。"
        "・競合比較や“競合優位性”などのメタ表現は禁止。"
        "・句点や読点を正しく用い、自然な文で100〜130文字を目安に。"
    )

    return knowledge_text, forb, synonyms_map

KNOWLEDGE_TEXT, FORBIDDEN_WORDS, SYN_MAP = collect_local_knowledge()

# ========= 5) テキスト整形ユーティリティ =========
JP_SPACES = re.compile(r"[ \t\u3000]+")
MULTI_PUNCTS = re.compile(r"[。\.]{2,}")
TRAILING_QUOTES = re.compile(r"[\"'’”）)\]]+$")
LEADING_QUOTES = re.compile(r"^[\"'‘“（(\[]+")

def normalize_text(s: str) -> str:
    s = s.replace("\r", "").replace("\n", " ").strip()
    s = JP_SPACES.sub(" ", s)
    s = LEADING_QUOTES.sub("", s)
    s = TRAILING_QUOTES.sub("", s)
    s = s.replace("..", "。")
    s = MULTI_PUNCTS.sub("。", s)
    return s.strip()

def is_punct_only(s: str) -> bool:
    if not s:
        return True
    t = s.strip()
    return all(ch in "。、.，,！!？?・" for ch in t)

def ends_with_terminal(s: str) -> bool:
    return s.endswith(("。","！","？","!","?"))

def finalize_sentence(s: str) -> str:
    s = normalize_text(s)
    if not s or is_punct_only(s):
        return ""  # 句点だけは禁止：無効化
    if not ends_with_terminal(s):
        s = s + "。"
    return s

def remove_forbidden(s: str, forbidden: List[str]) -> str:
    out = s
    for w in forbidden:
        if w and w in out:
            out = out.replace(w, "")  # 単純除去（意図せぬ語尾欠損は後段で整える）
    return normalize_text(out)

def natural_trim(s: str, min_len=80, target_max=110, hard_max=130) -> str:
    """
    - まず hard_max（130）で粗く抑える
    - できる限り句点（。）、読点（、）、スペースで自然カット
    - 最後に 80〜110 に収める努力をする
    """
    s = normalize_text(s)
    if len(s) > hard_max:
        s = s[:hard_max]
    # 末尾を自然に落とす
    cut_points = [i for i, ch in enumerate(s) if ch in ("。", "、", " ")]
    if cut_points:
        # target_max を超える場合は、target_max以下で一番後ろの区切りで切る
        if len(s) > target_max:
            candidates = [i for i in cut_points if i <= target_max]
            if candidates:
                s = s[:candidates[-1]+1]
    s = s.strip()
    s = finalize_sentence(s)
    # まだ長いならもう一段落とす
    if len(s) > target_max:
        # 最後の "。" で切る
        last = s.rfind("。")
        if last != -1 and last+1 >= min_len:
            s = s[:last+1]
        elif len(s) > target_max:
            s = s[:target_max].rstrip("、，,")
            s = finalize_sentence(s)
    return s

def clean_sentences(lines: List[str], forbidden: List[str]) -> List[str]:
    out = []
    seen = set()
    for raw in lines:
        s = normalize_text(raw)
        if not s or is_punct_only(s):
            continue
        s = remove_forbidden(s, forbidden)
        s = finalize_sentence(s)
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def diversify(sent: str, syn_map: Dict[str,str]) -> str:
    """
    同一文の軽微差分（ALT不足時の補完用）
    """
    s = sent
    for k, v in syn_map.items():
        if k in s:
            s = s.replace(k, v)
            break
    if s == sent:
        # 置換がなければ軽い言い回し変更
        s = s.replace("便利", "使いやすい").replace("最適", "ちょうど良い")
    return finalize_sentence(s)

# ========= 6) OpenAI 呼び出し（text 応答） =========
def call_openai_text(product_name: str, knowledge: str) -> str:
    """
    可能な限り“テキスト長文”で20件以上の候補を1レスポンスで取得。
    解析はローカルで安定化させる。
    """
    sys = (
        "あなたは日本語のプロのライターです。"
        "楽天向けALTテキストを作成します。"
        "画像描写やメタ表現は禁止。1文＝自然な日本語で出力。"
        "見出しや番号は不要。行ごとに1候補。"
    )
    usr = (
        f"商品名：{product_name}\n"
        f"{knowledge}\n"
        "出力要件：\n"
        "・1行につき1文のALT候補。\n"
        "・各文はおおよそ100〜130字で、文末は句点で自然に終える。\n"
        "・20件以上（25件程度）書き出す。\n"
        "・画像・写真の記述は入れない。\n"
        "・競合比較や“競合優位性”の語は使わない。\n"
        "・改行で区切って列挙。"
    )

    last_err = None
    for _ in range(RETRY):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                response_format="text",          # ← JSONは使わない
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                # temperature は一部モデル非対応があり得るので未指定（再現性優先なら 0.3 指定可）
                stream=False,
            )
            txt = res.choices[0].message.content if res.choices else ""
            if txt and txt.strip():
                return txt
        except Exception as e:
            last_err = e
            time.sleep(RETRY_WAIT)
    # すべて失敗時
    raise RuntimeError(f"OpenAI呼び出しに失敗しました: {last_err}")

def parse_lines_from_text(block: str) -> List[str]:
    """
    箇条書きでもプレーンでも受け取れるように、行単位に剥がす。
    """
    if not block:
        return []
    # ハイフン/番号箇条書きを想定して分解
    raw_lines = re.split(r"[\r\n]+", block)
    out = []
    for ln in raw_lines:
        ln = ln.strip()
        if not ln:
            continue
        # 箇条書き記号を剥ぐ
        ln = re.sub(r"^[-・*●○\d\.\)）]+\s*", "", ln)
        out.append(ln)
    return out

# ========= 7) ALT生成 + リファイン =========
def ai_generate_alt(product_name: str) -> List[str]:
    """
    1) AIで25文程度取得（100〜130字想定）
    2) ローカルでクレンジング → 80〜110字へ自然整形
    3) 20件に整える（不足は軽微差分で補完）
    """
    raw_text = call_openai_text(product_name, KNOWLEDGE_TEXT)
    lines = parse_lines_from_text(raw_text)

    # 1st pass: クレンジング
    cleaned = clean_sentences(lines, FORBIDDEN_WORDS)

    # 2nd pass: 長さ整形（自然トリム）
    shaped = [natural_trim(s, min_len=80, target_max=110, hard_max=130) for s in cleaned]
    shaped = clean_sentences(shaped, FORBIDDEN_WORDS)  # もう一度整列（空を除外）

    # 20件に満たない場合は補完
    while len(shaped) < 20 and shaped:
        shaped.append(diversify(shaped[len(shaped) % max(1, len(shaped)) - 1], SYN_MAP))

    # まだ足りない（極端に空）の場合の保険
    if not shaped:
        shaped = [finalize_sentence(f"{product_name} の特長を活かし、日常の不便を減らす実用的な設計。持ち運びやすさと耐久性に配慮し、毎日のシーンで安心して使える。")] * 20

    # 20件に揃える／超過は上位20件へ
    shaped = shaped[:20]
    return shaped

# ========= 8) 入力CSV 読込（UTF-8） =========
def read_products_from_csv(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"入力CSVが見つかりません: {path}")
    names = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "商品名" not in (reader.fieldnames or []):
            raise ValueError("CSVに『商品名』列がありません。")
        for r in reader:
            nm = (r.get("商品名") or "").strip()
            if nm:
                names.append(nm)
    # 一意化
    uniq = []
    seen = set()
    for nm in names:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq

# ========= 9) CSV出力 =========
def write_alt_csv(path: str, items: List[Tuple[str, List[str]]]):
    fieldnames = ["商品名"] + [f"ALT{i}" for i in range(1, 21)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, alts in items:
            row = {"商品名": name}
            for i in range(20):
                row[f"ALT{i+1}"] = alts[i] if i < len(alts) else ""
            writer.writerow(row)

# ========= 10) 進捗（tqdm任意） =========
def progress_iter(seq, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(seq, desc=desc)
    except Exception:
        # フォールバック
        total = len(seq)
        for i, x in enumerate(seq, 1):
            if i == 1 or i == total or i % max(1, total // 10) == 0:
                print(f"🧠 ALT生成中: {i}/{total} ({int(i/total*100)}%)")
            yield x

# ========= 11) main =========
def main():
    print("🌸 v3.3_altfix_utf8_final_schema_refine_stable2 実行開始（ALT長文→ローカル整形・禁則適用）")
    products = read_products_from_csv(INPUT_RAKUTEN)
    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")

    results: List[Tuple[str, List[str]]] = []
    for name in progress_iter(products, desc="🧠 ALT生成中"):
        try:
            alts = ai_generate_alt(name)
        except Exception as e:
            # フェイルセーフ：最低限1件から複製
            fallback = finalize_sentence(f"{name} は、日常の不便を減らす実用的な設計。携帯性と耐久性に配慮し、幅広いシーンで安心して使える。")
            alts = [fallback] * 20
            print(f"⚠️ 生成エラーを回避しフェイルセーフを適用: {e}")
        # 文字数統計
        lens = [len(x) for x in alts if x]
        avg_len = sum(lens)/len(lens) if lens else 0
        print(f"   ├ avg_len: {avg_len:.1f}字 / 有効: {len(lens)}件")
        results.append((name, alts))

    write_alt_csv(OUTPUT_CSV, results)
    print(f"✅ 出力完了: {OUTPUT_CSV}")
    print("✅ 仕様: ALTはAIで100〜130字→ローカルで80〜110字に整形。句点だけ行は除外・補完済。画像描写語・メタ語は禁止。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
