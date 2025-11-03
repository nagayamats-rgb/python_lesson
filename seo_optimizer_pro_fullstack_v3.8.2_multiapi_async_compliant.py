#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEO Optimizer Pro v3.8.2 (multiapi_async_compliant)
Author: ChatGPT + [Your Name]
Date: 2025-10-30

- 楽天 / Yahoo 商品検索API 正式準拠
- 進捗可視化 (tqdm_asyncio)
- API別エラー詳細表示
- ALT / キャッチコピー自動生成
"""

import os, sys, json, asyncio, aiohttp, random, re, logging, urllib.parse
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from janome.tokenizer import Tokenizer
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# ------------------------------
# ロギング設定
# ------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("seo-optimizer")

# ------------------------------
# 環境変数
# ------------------------------
load_dotenv()

RAKUTEN_API_BASE_URL = os.getenv("RAKUTEN_API_BASE_URL")
RAKUTEN_APP_ID = os.getenv("RAKUTEN_APP_ID")

YAHOO_API_BASE_URL = os.getenv("YAHOO_API_BASE_URL")
YAHOO_APP_ID = os.getenv("YAHOO_APP_ID")

OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ENABLE = os.getenv("OPENAI_ENABLE", "false").lower() == "true"
OPENAI_USE_BATCH = os.getenv("OPENAI_USE_BATCH", "true").lower() == "true"

CONCURRENCY = int(os.getenv("SEO_CONCURRENCY", "6"))

# ------------------------------
# CSV 読込（Shift_JIS対応）
# ------------------------------
def load_input_csv(path="input.csv"):
    path = Path(path)
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"✅ CSV読み込み成功: encoding={enc}, shape={df.shape}")
            break
        except Exception as e:
            logger.warning(f"⚠️ {enc} 読み込み失敗: {e}")
    else:
        raise RuntimeError("❌ CSV読み込みに失敗しました。")

    name_col = next((c for c in df.columns if "商品" in c or "name" in c.lower()), df.columns[2])
    df = df[df[name_col].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df, name_col

# ------------------------------
# APIクライアント（準拠＋詳細エラー）
# ------------------------------
class MarketAPIClient:
    def __init__(self, base_url, appid, service_name):
        self.base_url = base_url
        self.appid = appid
        self.name = service_name

    async def _fetch(self, session, params):
        async with session.get(self.base_url, params=params) as resp:
            txt = await resp.text()
            if resp.status == 200:
                return json.loads(txt)
            else:
                raise RuntimeError(f"{self.name} {resp.status} {txt[:180]}")

    async def fetch_with_retry(self, keyword, max_retries=3):
        # パラメータ分岐（公式仕様準拠）
        encoded_kw = urllib.parse.quote(keyword)
        if "rakuten" in self.base_url:
            params = {
                "applicationId": self.appid,
                "keyword": encoded_kw,
                "hits": 30,
                "format": "json",
                "formatVersion": 2,
                "sort": "-reviewCount"
            }
        elif "yahooapis" in self.base_url:
            params = {
                "appid": self.appid,
                "query": encoded_kw,
                "results": 30,
                "sort": "-review_count"
            }
        else:
            raise ValueError(f"未対応APIエンドポイント: {self.base_url}")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for attempt in range(1, max_retries+1):
                try:
                    data = await self._fetch(session, params)
                    if data:
                        return data
                except Exception as e:
                    logger.error(f"❌ {self.name} API失敗({attempt}/{max_retries}) [{keyword}]: {e}")
                    await asyncio.sleep(2 * attempt)
            logger.error(f"🚨 {self.name} リトライ上限超過 [{keyword}] — このクエリはスキップされます。")
            return {}

# ------------------------------
# カテゴリ推定
# ------------------------------
def infer_category(name, tokenizer):
    tokens = [t.surface for t in tokenizer.tokenize(name) if t.part_of_speech.startswith("名詞")]
    if not tokens:
        return "未分類"
    hints = {
        "ギフト": ["ギフト","贈り物","プレゼント"],
        "健康": ["健康","オーガニック","無添加"],
        "日用品": ["雑貨","収納","掃除"],
        "食品": ["食品","スイーツ","調味料"],
    }
    for cat, words in hints.items():
        if any(w in tokens for w in words):
            return cat
    return "".join(tokens[:2]) + "カテゴリ"

# ------------------------------
# 語彙辞書構築（7〜15位）
# ------------------------------
async def build_vocab_dictionary(client, df, name_col, tokenizer):
    vocab_map = defaultdict(lambda: defaultdict(list))
    async def process_item(name):
        cat = infer_category(name, tokenizer)
        data = await client.fetch_with_retry(name)
        items = []
        if "Items" in data:
            items = data.get("Items", [])
        elif "hits" in data:
            items = data.get("hits", [])
        if not items:
            return cat, []
        titles = []
        for item in items[6:15]:
            if isinstance(item, dict):
                title = (
                    item.get("Item", {}).get("itemName") or
                    item.get("name") or
                    item.get("Title") or ""
                )
                titles.append(title)
        words = []
        for t in titles:
            for token in tokenizer.tokenize(t):
                if token.part_of_speech.startswith("名詞") and len(token.surface) > 1:
                    words.append(token.surface)
        freq = Counter(words)
        return cat, [w for w, _ in freq.most_common(20)]

    tasks = [process_item(str(row[name_col])) for _, row in df.iterrows()]
    results = []
    for f in tqdm_asyncio.as_completed(tasks, desc=f"📊 {client.name} 語彙辞書生成中"):
        try:
            res = await f
            results.append(res)
        except Exception as e:
            logger.error(f"{client.name} 語彙処理中エラー: {e}")

    for cat, words in results:
        vocab_map[cat]["vocab"].extend(words)
    return vocab_map

# ------------------------------
# semantic block構造
# ------------------------------
SEMANTIC_TEMPLATES = {
    "spec": ["{name} {keyword}仕様", "人気の{keyword}搭載", "{keyword}デザイン {name}"],
    "feature": ["{keyword}が特長", "{keyword}で好評", "{keyword}が魅力"],
    "scene": ["{keyword}におすすめ", "{keyword}で活躍", "{keyword}用途に最適"],
    "benefit": ["{keyword}がうれしいポイント", "{keyword}で毎日を快適に", "{keyword}だから選ばれています"]
}

def inject_market_vocabulary(local_templates, market_vocab):
    for cat, data in market_vocab.items():
        words = data.get("vocab", [])
        if not words:
            continue
        for block in SEMANTIC_TEMPLATES.keys():
            merged = local_templates.setdefault(cat, {}).setdefault(block, [])
            for w in words:
                merged.append(random.choice(SEMANTIC_TEMPLATES[block]).format(keyword=w))
    return local_templates

# ------------------------------
# 文生成
# ------------------------------
def _clean(s): return re.sub(r"[ 　]+"," ",s).replace("、、","、").strip()

def compose_alt(name, genre, vocab, templates, n=20):
    results = []
    base_words = vocab.get(genre, {}).get("vocab", [genre])
    for _ in range(n):
        parts = []
        for block in ["spec","feature","scene","benefit"]:
            tpl = random.choice(templates.get(genre, {}).get(block, ["{keyword}"]))
            kw = random.choice(base_words)
            parts.append(tpl.format(name=name, keyword=kw))
        text = _clean(f"{name} {'、'.join(parts)}")
        results.append(text[:110])
    return list(dict.fromkeys(results))

def compose_catchcopy(name, brand, genre):
    base = f"{genre}なら、{brand}の「{name}」"
    pads = ["毎日にちょうどいい", "ギフトにも人気", "レビュー高評価"]
    while len(base) < 30:
        base += "、" + random.choice(pads)
    return base[:60]

# ------------------------------
# メイン処理
# ------------------------------
async def main_async(input_csv="input.csv", output_csv="output_alts.csv"):
    df, name_col = load_input_csv(input_csv)
    tokenizer = Tokenizer()
    rakuten = MarketAPIClient(RAKUTEN_API_BASE_URL, RAKUTEN_APP_ID, "Rakuten")
    yahoo = MarketAPIClient(YAHOO_API_BASE_URL, YAHOO_APP_ID, "Yahoo")

    logger.info("🪄 市場語彙辞書構築開始")
    rakuten_vocab = await build_vocab_dictionary(rakuten, df, name_col, tokenizer)
    yahoo_vocab = await build_vocab_dictionary(yahoo, df, name_col, tokenizer)

    market_vocab = defaultdict(lambda: defaultdict(list))
    for src in [rakuten_vocab, yahoo_vocab]:
        for cat, data in src.items():
            market_vocab[cat]["vocab"].extend(data.get("vocab", []))

    templates = inject_market_vocabulary(defaultdict(lambda: defaultdict(list)), market_vocab)

    outputs = []
    for _, row in tqdm_asyncio.tqdm(df.iterrows(), total=len(df), desc="🧠 ALT/コピー生成中"):
        name = str(row[name_col]).strip()
        brand = str(row.get("ブランド名",""))
        genre = infer_category(name, tokenizer)
        alts = compose_alt(name, genre, market_vocab, templates)
        catch = compose_catchcopy(name, brand, genre)
        outputs.append({"name": name, "category": genre, "catchcopy": catch, "alts": " | ".join(alts)})

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"💾 出力完了: {output_csv} ({len(out_df)}件)")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    logger.info("🚀 SEO Optimizer Pro v3.8.2 起動")
    main()
import atlas_autosave_core
