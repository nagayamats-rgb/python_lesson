# -*- coding: utf-8 -*-
"""
KOTOHA ENGINE — Hybrid AI Writer v4.2 (CSV基点・完全版)
- 入力: ./input.csv（cp932 / Shift-JIS相当）
  必須列: 「商品名」「ジャンルID」  ※R, X..CC(ALT)は最終CSVで使用するが本モジュールはJSON出力
- 参照: ./output/semantics/structured_semantics_*.json（任意）
        ./output/lexical_clusters_*.json（推奨：クラスタ語彙）
- 出力: ./output/ai_writer/hybrid_writer_full_YYYYMMDD_HHMM.json
- 仕様:
  * 全商品（例: ~763件）に対し Copy=1, ALT=20 を必ず生成
  * 生成方式はハイブリッド：
      - 約30%: OpenAIで生成（OPENAI_ENABLE=true の時）
      - 約70%: ローカル・テンプレ展開（高品質・長さ遵守）
  * 長さ制約: Copy=40–60（全角換算） / ALT=80–110（全角換算）
  * ユニーク性: ALTは20本すべて異なる（重複除去＋微差生成）
  * 禁則: 「存在しないスペック」推定を緩和（最終CSV直前のルールで強制）→ここでは“露骨な誇張語”のみ抑制
  * 進捗・統計: tqdm＋要約ログ
"""

import os
import re
import json
import glob
import math
import time
import random
import string
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ------- 設定 -------
SEED = 42
random.seed(SEED)

COPY_MIN, COPY_MAX = 40, 60           # 全角換算
ALT_MIN, ALT_MAX   = 80, 110          # 全角換算
ALT_COUNT_PER_ITEM = 20

INPUT_CSV = "./input.csv"
SEMANTICS_DIR = "./output/semantics"
CLUSTERS_DIR  = "./output"

OUT_DIR = "./output/ai_writer"
os.makedirs(OUT_DIR, exist_ok=True)

# ------- ユーティリティ -------
def zlen(s: str) -> int:
    """全角換算長（ざっくり：ASCIIは0.5, 非ASCIIは1.0としてカウント）"""
    if not s:
        return 0
    half = sum(ch in string.printable and ch not in "　" for ch in s)
    full = len(s) - half
    # 半角2つで全角1換算
    return full + math.ceil(half / 2)

def clamp_len(s: str, lo: int, hi: int) -> str:
    """全角長が範囲外なら微調整（短い→付け足し、長い→安全な形でカット）。"""
    txt = s.strip()
    L = zlen(txt)
    if L < lo:
        # 足し具（語尾を崩さず自然な追記）
        suffix_bank = [" — 詳細は商品ページへ", "。選ばれる定番仕様", "。日常を快適に", "。使うほど便利", "。シンプルに心地よく"]
        while zlen(txt) < lo:
            txt += random.choice(suffix_bank)
    elif L > hi:
        # 句点・読点・記号で安全カット
        cut_points = [m.start() for m in re.finditer(r"[。．、,・/|｜\-・!！\s]", txt)]
        if cut_points:
            # もっとも近い手前のカット点
            pos = None
            for p in cut_points:
                if zlen(txt[:p]) <= hi:
                    pos = p
                else:
                    break
            if pos:
                txt = txt[:pos].rstrip("、。,.!！・-/|｜")
            else:
                # 緊急カット
                txt = txt[:max(1, min(len(txt), hi))]
        else:
            txt = txt[:max(1, min(len(txt), hi))]
    return txt

def uniqueify(lines):
    """完全一致・前後空白差の重複を除去し順序保持。"""
    seen = set()
    out = []
    for s in lines:
        key = s.strip()
        if key not in seen and key:
            seen.add(key)
            out.append(s)
    return out

def safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ------- データ読込 -------
def load_csv_items(path=INPUT_CSV):
    df = pd.read_csv(path, encoding="cp932", dtype=str).fillna("")
    items = []
    for _, row in df.iterrows():
        name  = row.get("商品名", "").strip()
        genre = row.get("ジャンルID", "").strip()
        if name:
            items.append({"name": name, "genre": genre})
    return items

def latest_json(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_support():
    sem_path = latest_json(os.path.join(SEMANTICS_DIR, "structured_semantics_*.json"))
    clu_path = latest_json(os.path.join(CLUSTERS_DIR,  "lexical_clusters_*.json"))
    semantics = safe_json_load(sem_path) or {}
    clusters  = safe_json_load(clu_path) or {}
    # クラスタJSONは {cluster_name: {keywords:[], patterns:[]}} or [ {name, keywords} ] の双方に対応
    cluster_list = []
    if isinstance(clusters, dict):
        for k, v in clusters.items():
            kws = v.get("keywords") or v.get("words") or []
            cluster_list.append({"name": k, "keywords": kws})
    elif isinstance(clusters, list):
        for c in clusters:
            nm = c.get("name") or c.get("cluster") or "cluster"
            kws = c.get("keywords") or c.get("words") or []
            cluster_list.append({"name": nm, "keywords": kws})
    return semantics, cluster_list

# ------- スコアリング（AIコール選定） -------
def tokenize(s: str):
    s = re.sub(r"[【】\[\]（）()!！/｜|,、.\-＋+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [w for w in s.split(" ") if w]

def novelty_score(name_tokens, cluster_kws):
    # 商品名とクラスタ語彙の非重複率をスコア化（高いほど新規性＝AI向き）
    if not cluster_kws:
        return 0.5
    overlap = len(set(name_tokens) & set(cluster_kws))
    base = 1.0 - (overlap / (len(set(name_tokens)) + 1e-6))
    return max(0.0, min(1.0, base))

def density_score(name_tokens):
    # トークン数が多いほど構文が複雑→AI向き
    n = len(name_tokens)
    return max(0.0, min(1.0, (n - 4) / 12))  # 4語で0、16語でほぼ1

def rarity_score(genre, genre_freq):
    # 現在までのジャンル頻度が少ないほどAI向き（多様性確保）
    total = sum(genre_freq.values()) + 1e-6
    freq = genre_freq.get(genre or "NA", 0) + 1e-6
    rarity = 1.0 - (freq / (total))
    return max(0.0, min(1.0, rarity))

def choose_ai_indices(items, clusters, target_ratio=0.30):
    # 各商品に対しスコア計算 → 上位30%をAI
    scores = []
    genre_freq = Counter()
    for idx, it in enumerate(items):
        tokens = tokenize(it["name"])
        clu = clusters[idx % max(1, len(clusters))] if clusters else {"keywords": []}
        base = 0.5 * novelty_score(tokens, clu.get("keywords", [])) \
             + 0.35 * density_score(tokens) \
             + 0.15 * rarity_score(it.get("genre",""), genre_freq)
        scores.append((idx, base))
        # 頻度更新（後続の rarity に効く）
        genre_freq[it.get("genre","")] += 1

    scores.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(round(len(items) * target_ratio)))
    ai_idx = set(i for i, _ in scores[:k])
    return ai_idx

# ------- ローカルテンプレ（高品質） -------
COPY_PATTERNS = [
    "{core}、{benefit}。{scene}",
    "{core} — {benefit}で、{scene} を快適に。",
    "{core}。{spec} を備え、{scene}にちょうどいい。"
]

ALT_PATTERNS = [
    "{core} / {spec} / {benefit} / {scene}",
    "{scene}に最適な{core}（{spec}）— {benefit}",
    "{brand}{category}{core}：{benefit}、{scene}、{spec}",
    "{core} | {scene} | {benefit} | {spec}"
]

BENEFITS = [
    "毎日を軽やかにする使い心地", "ストレスのない操作感", "長く使える安心設計",
    "無駄のない美しさ", "忙しい日常にフィット", "はじめてでも迷わない"
]
SCENES = [
    "自宅でも外出先でも", "オフィスとプライベートに", "旅行や出張に",
    "通勤・通学の相棒に", "ギフトにも最適", "ワークと趣味の両立に"
]
SPECS_HINT = [
    "軽量設計", "耐久性の高い素材", "持ち運びしやすいサイズ", "日常使いに十分な性能",
    "お手入れ簡単", "使い勝手を優先した設計"
]
BRANDS_HINT = ["", "", ""]

def sample_words(name: str, cluster_kws):
    # 商品名から核語（core）・カテゴリ推定
    tokens = tokenize(name)
    # 核（目立つ語）を適当に抽出（簡易）
    core = " ".join(tokens[:3]) if tokens else name[:12]
    category = ""
    for w in ["ケース","フィルム","充電器","ケーブル","バンド","バッグ","リュック","ドレス","水着"]:
        if w in name:
            category = w
            break
    # specはクラスタ語彙とヒントから
    spec = random.choice(SPECS_HINT + (cluster_kws or []) or ["使いやすさ重視"])
    benefit = random.choice(BENEFITS)
    scene = random.choice(SCENES)
    brand = random.choice(BRANDS_HINT)

    return dict(core=core, category=category, spec=spec, benefit=benefit, scene=scene, brand=brand)

def local_copy(name, cluster_kws):
    w = sample_words(name, cluster_kws)
    s = random.choice(COPY_PATTERNS).format(**w)
    s = re.sub(r"\s{2,}", " ", s).strip(" ・/|｜")
    s = clamp_len(s, COPY_MIN, COPY_MAX)
    s = s.rstrip("!！")  # 感嘆符は避ける方針
    return s

def local_alts(name, cluster_kws, need=ALT_COUNT_PER_ITEM):
    outs = []
    used = set()
    # バリエーション源
    kws = list(set((cluster_kws or []) + tokenize(name)))
    random.shuffle(kws)
    i = 0
    while len(outs) < need and i < need * 5:
        i += 1
        w = sample_words(name, kws[:6])
        tmpl = random.choice(ALT_PATTERNS)
        s = tmpl.format(**w)
        s = re.sub(r"\s{2,}", " ", s).strip(" ・/|｜")
        s = clamp_len(s, ALT_MIN, ALT_MAX)
        if s not in used:
            outs.append(s); used.add(s)
    # 不足分は微差で補う
    while len(outs) < need and outs:
        base = random.choice(outs)
        tweak = random.choice(["— 詳しくは商品ページへ", " / 使いやすさを追求", " / 日常にちょうどいい"])
        s = clamp_len(base + tweak, ALT_MIN, ALT_MAX)
        if s not in used:
            outs.append(s); used.add(s)
    return outs[:need]

# ------- OpenAI（任意） -------
def openai_client():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None

OPENAI_PROMPT = """あなたは優秀なECコピーライター兼SEOアナリストです。
与えられた「商品名」「文脈語」（任意）をもとに、
1) 購買意欲を喚起するキャッチコピー（全角40〜60字）
2) SEOに最適化された画像ALTテキスト（全角80〜110字）×20本
を日本語で生成してください。

制約:
- 句読点・助詞の自然さを最優先。
- 感嘆符は避ける。
- 事実に過度な推測を加えない（曖昧な場合は一般的価値に寄せる）。
- ALTはすべて異なる内容にする。
- 出力はstrictなJSONで:
{{
  "copy": "<string>",
  "alts": ["<string>", ... 20本]
}}
"""

def call_openai_copy_alts(client, name, context_words):
    try:
        ctx = "、".join(context_words[:8]) if context_words else ""
        prompt = f"商品名: {name}\n文脈語: {ctx}\n\n" + OPENAI_PROMPT
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.6,
            max_tokens=800
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        copy = clamp_len(data.get("copy",""), COPY_MIN, COPY_MAX)
        alts = [clamp_len(a, ALT_MIN, ALT_MAX) for a in (data.get("alts") or [])]
        alts = uniqueify(alts)[:ALT_COUNT_PER_ITEM]
        # 不足補完
        if len(alts) < ALT_COUNT_PER_ITEM:
            alts += local_alts(name, context_words, ALT_COUNT_PER_ITEM - len(alts))
        return copy, alts
    except Exception:
        # フォールバック
        copy = local_copy(name, context_words)
        alts = local_alts(name, context_words, ALT_COUNT_PER_ITEM)
        return copy, alts

# ------- メイン処理 -------
def main():
    load_dotenv()  # .env の OPENAI_ENABLE を参照
    openai_enable = (os.getenv("OPENAI_ENABLE","false").lower() == "true")
    client = openai_client() if openai_enable else None

    # 入力
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"input.csv が見つかりません: {INPUT_CSV}")
    items = load_csv_items(INPUT_CSV)
    semantics, clusters = load_support()

    total = len(items)
    if total == 0:
        raise RuntimeError("CSVに有効な商品行がありません（商品名が空）")

    # AIコール選定（約30%）
    ai_indices = choose_ai_indices(items, clusters, target_ratio=0.30)

    results = []
    ai_count = 0
    tpl_count = 0

    print("🌸 Hybrid AI Writer v4.2 実行開始")
    pbar = tqdm(range(total), desc="🪄 商品生成中", total=total)
    for i in pbar:
        prod = items[i]
        name = prod["name"]
        clu = clusters[i % max(1, len(clusters))] if clusters else {"keywords": []}
        ctx_words = clu.get("keywords", [])

        use_ai = (i in ai_indices) and (client is not None)
        if use_ai:
            copy, alts = call_openai_copy_alts(client, name, ctx_words)
            ai_count += 1
        else:
            copy = local_copy(name, ctx_words)
            alts = local_alts(name, ctx_words, ALT_COUNT_PER_ITEM)
            tpl_count += 1

        # 最終ガード（長さ・件数・重複）
        copy = clamp_len(copy, COPY_MIN, COPY_MAX)
        alts = uniqueify([clamp_len(a, ALT_MIN, ALT_MAX) for a in alts])[:ALT_COUNT_PER_ITEM]
        while len(alts) < ALT_COUNT_PER_ITEM:
            # 追加微差
            base = local_alts(name, ctx_words, 1)[0]
            if base not in alts:
                alts.append(base)

        results.append({
            "index": i,
            "product_name": name,
            "genre": prod.get("genre",""),
            "copy": copy,
            "alts": alts
        })
        if (i+1) % 50 == 0 or i == total-1:
            pbar.set_postfix({"AI": ai_count, "TPL": tpl_count})

    # 出力
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"hybrid_writer_full_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"items": results}, f, ensure_ascii=False, indent=2)

    # 検証
    miss_copy = sum(1 for r in results if not r["copy"])
    miss_alts = sum(1 for r in results if len(r["alts"]) != ALT_COUNT_PER_ITEM)
    avg_copy = sum(zlen(r["copy"]) for r in results)/total
    avg_alts = sum(sum(zlen(a) for a in r["alts"])/ALT_COUNT_PER_ITEM for r in results)/total

    print("✅ 出力完了:", out_path)
    print(f"📊 件数: {total}（AI:{ai_count} / TPL:{tpl_count}）")
    print(f"📏 Copy平均長: {avg_copy:.1f} / ALT平均長: {avg_alts:.1f}")
    print(f"🔎 欠落確認: Copy欠落={miss_copy}, ALT件数不正={miss_alts}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
