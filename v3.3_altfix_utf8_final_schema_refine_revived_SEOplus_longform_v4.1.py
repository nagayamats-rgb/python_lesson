# -*- coding: utf-8 -*-
"""
ALT長文（SEO＋自然文）生成 v4.1r3（構造知識＋gpt-5）
"""

import os, re, csv, glob, json, time
from dotenv import load_dotenv
from collections import defaultdict

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from openai import OpenAI
except Exception:
    raise SystemExit("openai SDK が見つかりません。pip install openai python-dotenv")

# === 定数 ===
INPUT_CSV = "./rakuten.csv"
OUT_DIR   = "./output/ai_writer"
RAW_PATH  = os.path.join(OUT_DIR, "alt_text_ai_raw_longform_v4.1.csv")
REF_PATH  = os.path.join(OUT_DIR, "alt_text_refined_final_longform_v4.1.csv")
DIFF_PATH = os.path.join(OUT_DIR, "alt_text_diff_longform_v4.1.csv")
SEMANTICS_DIR = "./output/semantics"

FORBIDDEN = [
    "画像","写真","見た目","上の画像","下の写真","当店","当社","レビュー","ランキング",
    "クリック","こちら","競合","優位性","業界最高","最安","No.1","ナンバーワン","リンク",
    "ページ","カート","購入はこちら","送料無料","返金保証"
]

RAW_MIN, RAW_MAX = 100, 130
FINAL_MIN, FINAL_MAX = 80, 110

LEADING_ENUM_RE = re.compile(r"^\s*[\d①②③④⑤⑥⑦⑧⑨⑩\-\*\・\u2022]\s*[\.．、]?\s*")
WHITESPACE_RE   = re.compile(r"\s+")
MULTI_COMMA_RE  = re.compile(r"、{3,}")

# === 初期化 ===
def init_client():
    load_dotenv(override=True)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY が見つかりません。.env を確認してください。")
    model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"
    client = OpenAI(api_key=key)
    return client, model

# === 商品読み込み ===
def load_products(path):
    if not os.path.exists(path):
        raise SystemExit(f"CSVが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "商品名" not in reader.fieldnames:
            raise SystemExit("ヘッダに『商品名』列が必要です。")
        items = [r["商品名"].strip() for r in reader if r.get("商品名")]
    uniq, seen = [], set()
    for nm in items:
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)
    return uniq

# === 知見構造化 ===
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge_structured():
    if not os.path.isdir(SEMANTICS_DIR):
        payload = {
            "keywords": [], "market_terms": [], "scenes": [], "targets": [],
            "benefits": [], "tone": ["自然で読みやすく", "SEO効果を意識"], "forbidden": FORBIDDEN
        }
        return payload

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    payload = defaultdict(list)
    forbid = []

    for p in files:
        data = safe_load_json(p)
        if not data: continue
        name = os.path.basename(p).lower()

        if "lexical" in name or "cluster" in name:
            arr = data.get("clusters") if isinstance(data, dict) else data
            if isinstance(arr, list):
                for c in arr:
                    if isinstance(c, dict):
                        payload["keywords"] += c.get("terms", [])

        if "market" in name:
            v = data.get("vocabulary") or data.get("vocab") or []
            if isinstance(v, list):
                payload["market_terms"] += [x for x in v if isinstance(x, str)]

        if "semantic" in name:
            # ✅ dict / list 両対応
            if isinstance(data, dict):
                for k in ["concepts", "scenes", "targets", "benefits"]:
                    payload[k] += [x for x in (data.get(k) or []) if isinstance(x, str)]
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for k in ["concepts", "scenes", "targets", "benefits"]:
                            payload[k] += [x for x in (item.get(k) or []) if isinstance(x, str)]

        if "persona" in name:
            # ✅ dict / list 両対応
            if isinstance(data, list):
                payload["tone"] += [v for v in data if isinstance(v, str)]
            elif isinstance(data, dict):
                tone = data.get("tone") or {}
                if isinstance(tone, dict):
                    payload["tone"] += [v for v in tone.values() if isinstance(v, str)]

        if "forbid" in name or "normalize" in name:
            # ✅ dict / list 両対応
            if isinstance(data, list):
                forbid += [w for w in data if isinstance(w, str)]
            elif isinstance(data, dict):
                fw = data.get("forbidden_words") or data.get("forbid") or []
                if isinstance(fw, list):
                    forbid += [w for w in fw if isinstance(w, str)]

    payload["forbidden"] = list({*FORBIDDEN, *forbid})
    for k, v in payload.items():
        payload[k] = list(dict.fromkeys(v))
    return payload

# === プロンプト ===
SYSTEM_PROMPT = (
    "あなたは日本語ECサイトのSEOライティング専門家です。"
    "楽天用の商品画像ALTテキストを20本生成します。"
    "以下の構造化知識を参考に、自然で魅力的な文を作成してください。\n"
    "【必須ルール】\n"
    "・画像・写真などの描写語は禁止。\n"
    "・メタ表現（競合優位性・No.1 等）は禁止。\n"
    "・全角約100〜130字、1〜2文で自然に句点で終える。\n"
    "・商品名・対応機種・機能・用途・ベネフィットを自然に含める。\n"
    "・20行のテキストのみを返してください。"
)

def build_user_prompt(product, knowledge):
    payload = json.dumps(knowledge, ensure_ascii=False, indent=2)
    return (
        f"商品名: {product}\n"
        "次の構造化知識を参考にして、楽天SEOに最適化された自然なALT文を20件生成してください。\n"
        f"{payload}\n"
        "各行は独立した自然文で、句点「。」で終わること。"
    )

# === OpenAI 呼び出し ===
def call_openai_lines(client, model, product, knowledge, retry=3, wait=5):
    user_prompt = build_user_prompt(product, knowledge)
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "text"},
                max_completion_tokens=1800,
                temperature=1
            )
            txt = (res.choices[0].message.content or "").strip()
            if txt:
                lines = [LEADING_ENUM_RE.sub("", ln).strip("・-—●　") for ln in txt.split("\n") if ln.strip()]
                return lines[:60]
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI応答なし: {last_err}")

# === 整形 ===
def soft_clip(t):
    t = t.strip()
    if not t.endswith("。"):
        t += "。"
    t = WHITESPACE_RE.sub(" ", t)
    t = MULTI_COMMA_RE.sub("、、", t)
    if len(t) > 120:
        cut = t[:120]
        p = cut.rfind("。")
        t = cut[:p+1] if p != -1 else cut
    for ng in FORBIDDEN:
        t = t.replace(ng, "")
    return t.strip()

def refine_lines(raw):
    valid = []
    for ln in raw:
        if not ln:
            continue
        s = soft_clip(ln)
        if len(s) < 15:
            continue
        valid.append(s)
    uniq = list(dict.fromkeys(valid))
    refined = [soft_clip(x) for x in uniq][:20]
    while len(refined) < 20 and refined:
        refined.append(refined[len(refined) % len(refined)])
    return refined[:20]

# === 出力 ===
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# === メイン ===
def main():
    print("🌸 ALT長文生成 v4.1r3（構造知識＋gpt-5 完全安定版）")
    client, model = init_client()
    ensure_outdir()
    products = load_products(INPUT_CSV)
    print(f"✅ 商品数: {len(products)}件")

    knowledge = summarize_knowledge_structured()
    raws, refs = [], []

    for p in tqdm(products, desc="🧠 生成中", total=len(products)):
        try:
            raw = call_openai_lines(client, model, p, knowledge)
        except Exception as e:
            raw = [f"{p} の特徴を活かした設計で日常の利便性を高めます。"] * 20
        ref = refine_lines(raw)
        raws.append(raw[:20])
        refs.append(ref)
        time.sleep(0.2)

    write_csv(RAW_PATH, ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)],
              [[p] + r + [""] * (20 - len(r)) for p, r in zip(products, raws)])
    write_csv(REF_PATH, ["商品名"] + [f"ALT_{i+1}" for i in range(20)],
              [[p] + r + [""] * (20 - len(r)) for p, r in zip(products, refs)])
    write_csv(DIFF_PATH,
              ["商品名"] + [f"ALT_raw_{i+1}" for i in range(20)] + [f"ALT_refined_{i+1}" for i in range(20)],
              [[p] + raws[i] + refs[i] for i, p in enumerate(products)])

    avg = lambda xs: sum(len(x) for l in xs for x in l if x) / max(1, sum(len(l) for l in xs))
    print("✅ 出力完了:")
    print(f"   raw={RAW_PATH}\n   refined={REF_PATH}\n   diff={DIFF_PATH}")
    print(f"📏 平均文字数 raw={avg(raws):.1f}, refined={avg(refs):.1f}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
