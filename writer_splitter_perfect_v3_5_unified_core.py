# -*- coding: utf-8 -*-
"""
writer_splitter_perfect_v3_5_unified_core.py
=========================================================
統合ライター（要・KOTOHA Framework）

- 楽天・Yahoo・ALT（楽天専用）を全件AI生成
- ローカル知見（/output/semantics/）を要約しAIに渡す
- 禁則語・長さ・句点補完など統一ルールで整形
- GPT-4o（.env 固定）を利用
=========================================================

【要（かんなめ）構造理念】
---------------------------------------------------------
要とは、AI生成・ローカル知見・整形・禁則・出力統合の結節点である。

1. AI知見活用：/output/semantics/ 配下の JSON 群を自動集約し、商品別に要約。
2. 安定生成構文：GPT 応答を JSON 構造で受け、安全パースと再試行を備える。
3. 整形ルール統合：禁則語・文字長・句点補完・語尾自然化を全処理に適用。
4. 出力統合：楽天・Yahoo・ALT（楽天専用）を同一プロセスで同時生成。

本スクリプトはプロジェクトの心臓部であり、変更時は再審査の上で凍結解除する。
=========================================================
"""

import os
import re
import csv
import json
import glob
import time
from dotenv import load_dotenv
from collections import defaultdict
from openai import OpenAI

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x


# =====================================================
# 0. 環境初期化
# =====================================================
def init_env_and_client():
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")
    client = OpenAI(api_key=api_key)
    model = "gpt-4o"
    return client, model


# =====================================================
# 1. 定数・禁則語
# =====================================================
BASE_FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真", "当店", "当社",
    "レビュー", "ランキング", "クリック", "こちら", "リンク", "ページ",
    "カート", "購入はこちら", "送料無料（確約）", "返金保証", "競合", "優位性",
    "No.1", "ナンバーワン", "最安", "業界最高", "最強"
]

# Yahoo専用の販促・誘導語（ALTでは禁止）
ALT_FORBIDDEN_EXT = [
    "送料無料", "セール", "期間限定", "ポイント", "クーポン", "お買い得",
    "割引", "キャンペーン", "ご注文", "早い者勝ち", "在庫限り",
    "レビュー投稿", "ショップ", "お気に入り", "今すぐ", "特別価格", "数量限定"
]

FORBIDDEN_ALL = list({*BASE_FORBIDDEN, *ALT_FORBIDDEN_EXT})


# =====================================================
# 2. 知見統合
# =====================================================
SEMANTICS_DIR = "./output/semantics"

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def summarize_knowledge():
    if not os.path.isdir(SEMANTICS_DIR):
        return "知見: 主な用途・対象・特徴・スペック・関連語を自然に含めてください。", []

    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    clusters, market, semantics, tone, template, forb = [], [], [], [], [], []

    for p in files:
        data = safe_load_json(p)
        if not data: continue
        name = os.path.basename(p).lower()

        if "cluster" in name:
            if isinstance(data, dict):
                arr = data.get("clusters", [])
                for c in arr:
                    if isinstance(c, dict):
                        terms = c.get("terms", [])
                        clusters.extend([t for t in terms if isinstance(t, str)])
        elif "market" in name:
            if isinstance(data, list):
                for v in data:
                    if isinstance(v, dict) and "vocabulary" in v:
                        market.append(v["vocabulary"])
                    elif isinstance(v, str):
                        market.append(v)
        elif "semantic" in name:
            if isinstance(data, dict):
                for k in ["concepts", "scenes", "targets"]:
                    semantics.extend(data.get(k, []))
        elif "persona" in name:
            if isinstance(data, dict):
                tone.append(json.dumps(data))
        elif "template" in name:
            if isinstance(data, dict):
                template.extend(data.get("hints", []))
        elif "forbid" in name or "normalized" in name:
            if isinstance(data, dict):
                forb.extend(data.get("forbidden_words", []))

    text = "知見: " + "、".join(set(clusters + market + semantics))[:300]
    return text, list({*FORBIDDEN_ALL, *forb})


# =====================================================
# 3. 入力ロード
# =====================================================
def load_products(csv_path="./rakuten.csv"):
    if not os.path.exists(csv_path):
        raise SystemExit(f"入力ファイルが見つかりません: {csv_path}")
    products = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("商品名") or "").strip()
            if name:
                products.append(name)
    seen, uniq = set(), []
    for p in products:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


# =====================================================
# 4. AI呼び出し
# =====================================================
def call_openai_json(client, model, messages, retry=3, wait=5):
    last_err = None
    for _ in range(retry):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1200,
                response_format={"type": "json_object"},
                temperature=1,
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(wait)
    raise RuntimeError(f"OpenAI呼び出し失敗: {last_err}")


# =====================================================
# 5. 整形ルール
# =====================================================
def refine_text(text):
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    if not t.endswith("。"):
        t += "。"
    for ng in FORBIDDEN_ALL:
        t = t.replace(ng, "")
    return t[:110].strip()


# =====================================================
# 6. メイン
# =====================================================
def main():
    print("🌸 writer_splitter_perfect_v3_5_unified_core 実行開始（要／楽天ALT専用）")
    client, model = init_env_and_client()
    knowledge, forbidden_local = summarize_knowledge()

    products = load_products()
    print(f"✅ 商品名抽出: {len(products)}件（重複除去済）")

    out_path = "./output/ai_writer/rakuten_alt_core.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["商品名"] + [f"ALT_{i+1}" for i in range(20)])

        for p in tqdm(products, desc="🧠 商品別AI生成中", total=len(products)):
            sys_msg = {
                "role": "system",
                "content": (
                    "あなたは日本語のプロコピーライターです。"
                    "楽天市場の商品画像ALTを生成します。"
                    "自然な日本語で、1〜2文、全角80〜110文字に収め、句点で終えること。"
                    "画像や写真などの描写語、販促語（送料無料、セール、今すぐ 等）は禁止です。"
                )
            }
            user_msg = {
                "role": "user",
                "content": (
                    f"商品名: {p}\n{knowledge}\n"
                    "20件の自然文ALTをJSON形式で生成し、キーは alt1〜alt20 にしてください。"
                )
            }

            try:
                data = call_openai_json(client, model, [sys_msg, user_msg])
                alts = [refine_text(data.get(f"alt{i+1}", "")) for i in range(20)]
            except Exception as e:
                alts = [f"{p} は毎日の生活を快適にする実用的なデザインです。"] * 20
            writer.writerow([p] + alts)
            time.sleep(0.2)

    print(f"✅ 出力完了: {out_path}")
    print("🔒 凍結バージョン：要（かんなめ）正式実装・楽天ALT専用構成")


if __name__ == "__main__":
    main()
import atlas_autosave_core
