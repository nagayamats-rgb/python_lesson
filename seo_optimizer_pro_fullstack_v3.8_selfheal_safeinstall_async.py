#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEO Optimizer Pro v3.8 (multiapi_async, semantic blocks, persona-ready)
Author: ChatGPT + [あなたの名]
Date: 2025-10-30

目的:
  - Shift_JIS対応CSVから商品データを読み込み
  - .envに記載のRakuten / Yahoo / OpenAI APIを安全に呼び出す
  - 各APIから検索結果7〜15位の商品情報を収集して語彙辞書を鍛える
  - 感情トーン対応キャッチコピー（30〜60字）
  - 意図構造ALT文（90〜110字）
  - 完全非同期・API自己修復対応
"""

# ===============================
# Imports & Auto-install
# ===============================
import os, sys, time, json, asyncio, aiohttp, random, re, logging, pickle
from pathlib import Path
from collections import defaultdict, Counter

REQUIRED = ["aiohttp", "pandas", "tqdm", "janome", "python-dotenv"]
for mod in REQUIRED:
    try:
        __import__(mod)
    except ImportError:
        print(f"⚙️ Installing missing module: {mod}")
        os.system(f"{sys.executable} -m pip install -U {mod}")

import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from janome.tokenizer import Tokenizer
from dotenv import load_dotenv

# ===============================
# 設定・ロギング
# ===============================
load_dotenv()
LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("seo-optimizer")

# ===============================
# 環境変数（.envから読込）
# ===============================
RAKUTEN_API_BASE_URL = os.getenv("RAKUTEN_API_BASE_URL")
RAKUTEN_APP_ID = os.getenv("RAKUTEN_APP_ID")

YAHOO_API_BASE_URL = os.getenv("YAHOO_API_BASE_URL")
YAHOO_APP_ID = os.getenv("YAHOO_APP_ID")

OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ENABLE = os.getenv("OPENAI_ENABLE", "false").lower() == "true"
OPENAI_USE_BATCH = os.getenv("OPENAI_USE_BATCH", "true").lower() == "true"

CONCURRENCY = int(os.getenv("SEO_CONCURRENCY", "6"))
CHECKPOINT_FILE = "checkpoint.pkl"
ERROR_LOG_FILE = "api_error.jsonl"

# ===============================
# 入力読み込み（Shift_JIS対応）
# ===============================
def load_input_csv(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {path}")
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path_obj, encoding=enc)
            logger.info(f"✅ CSV読み込み成功: encoding={enc}, shape={df.shape}")
            break
        except Exception as e:
            logger.warning(f"⚠️ 読み込み失敗（encoding={enc}）: {e}")
    else:
        raise UnicodeError("CSVエンコーディングを判定できませんでした。")

    # 商品名空欄行は生成対象外（カラバリ）
    if "商品名" in df.columns:
        name_col = "商品名"
    else:
        name_col = df.columns[2]  # Fallback
    df = df[df[name_col].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df, name_col

# ===============================
# API自己修復クライアント
# ===============================
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
            elif resp.status in (401,403):
                raise PermissionError(f"{self.name}: 認証エラー {resp.status}")
            elif resp.status == 429:
                raise ConnectionRefusedError(f"{self.name}: レート制限 {resp.status}")
            elif resp.status >= 500:
                raise ConnectionError(f"{self.name}: サーバーエラー {resp.status}")
            else:
                raise RuntimeError(f"{self.name}: 予期しない応答 {resp.status}: {txt[:150]}")

    async def fetch_with_retry(self, keyword, max_retries=3):
        params = {"applicationId": self.appid, "appid": self.appid, "keyword": keyword, "query": keyword, "hits": 30, "results": 30}
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    data = await self._fetch(session, params)
                    return data
                except PermissionError as e:
                    logger.error(f"🔑 {self.name}: APIキー無効。{e}")
                    await asyncio.sleep(2)
                except ConnectionRefusedError:
                    logger.warning(f"⏳ {self.name}: 制限中。リトライ待機({attempt+1}/3)")
                    await asyncio.sleep(30)
                except ConnectionError:
                    logger.warning(f"🔁 {self.name}: サーバー再試行({attempt+1}/3)")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"❌ {self.name}: 不明エラー {e}")
                    await asyncio.sleep(2)
        raise RuntimeError(f"{self.name}: API再試行失敗 ({keyword})")

# ===============================
# カテゴリ生成（検索意図に基づく）
# ===============================
def infer_category(name: str, tokenizer=None) -> str:
    """商品名から検索意図カテゴリを類推する"""
    if not name:
        return "未分類"
    if tokenizer is None:
        tokenizer = Tokenizer()

    tokens = [t.surface for t in tokenizer.tokenize(name)
              if t.part_of_speech.startswith("名詞")]
    if not tokens:
        return "その他"

    # 意図クラスタ辞書（初期ヒント）
    CATEGORY_HINTS = {
        "ギフト": ["ギフト", "贈り物", "プレゼント", "お祝い"],
        "健康": ["オーガニック", "無添加", "健康", "ナチュラル"],
        "日用品": ["キッチン", "掃除", "収納", "雑貨"],
        "ファッション": ["バッグ", "アクセサリー", "シャツ", "靴"],
        "食品": ["スイーツ", "食品", "調味料", "ご飯", "ドリンク"],
    }

    for cat, hints in CATEGORY_HINTS.items():
        if any(h in tokens for h in hints):
            return cat

    # 未定義カテゴリ → トップ2名詞連結
    return "".join(tokens[:2]) + "カテゴリ"

# ===============================
# 市場語彙抽出（7〜15位）→ 共起語辞書生成
# ===============================
async def build_vocab_dictionary(client: MarketAPIClient, df, name_col, tokenizer):
    """APIから7〜15位の商品タイトルを抽出し、カテゴリ別語彙辞書を構築"""
    vocab_map = defaultdict(lambda: defaultdict(list))

    async def fetch_and_extract(name):
        cat = infer_category(name, tokenizer)
        data = await client.fetch_with_retry(name)
        items = []
        if "Items" in data:  # Rakuten形式
            items = data["Items"]
        elif "hits" in data:  # Yahoo形式
            items = data["hits"]
        if not items:
            return cat, []

        # 7〜15位のタイトルから名詞抽出
        titles = []
        for item in items[6:15]:
            title = ""
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
        common_words = [w for w, _ in freq.most_common(20)]
        return cat, common_words

    tasks = [fetch_and_extract(str(n)) for n in df[name_col].head(50)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, tuple):
            cat, words = res
            for w in words:
                vocab_map[cat]["vocab"].append(w)
    return vocab_map

# ===============================
# semantic block 構造（spec, feature, scene, benefit）
# ===============================
SEMANTIC_TEMPLATES = {
    "spec": [
        "{name} {keyword}仕様", "人気の{keyword}搭載", "{keyword}デザイン {name}"
    ],
    "feature": [
        "{keyword}が特長", "{keyword}で好評", "{keyword}が魅力"
    ],
    "scene": [
        "{keyword}におすすめ", "{keyword}シーンに最適", "{keyword}で活躍"
    ],
    "benefit": [
        "{keyword}で毎日を快適に", "{keyword}がうれしいポイント", "{keyword}だから選ばれています"
    ],
}

def inject_market_vocabulary(local_templates, market_vocab):
    """市場語彙辞書をテンプレートに注入"""
    for cat, data in market_vocab.items():
        words = data.get("vocab", [])
        if not words:
            continue
        for block in SEMANTIC_TEMPLATES.keys():
            merged = local_templates.setdefault(cat, {}).setdefault(block, [])
            for w in words:
                merged.append(random.choice(SEMANTIC_TEMPLATES[block]).format(keyword=w))
    return local_templates

# ===============================
# ローカル初期語彙＋テンプレート生成
# ===============================
def bootstrap_local_vocab_and_templates(df, name_col):
    tokenizer = Tokenizer()
    vocab = defaultdict(lambda: defaultdict(list))
    templates = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        cat = infer_category(name, tokenizer)
        words = [t.surface for t in tokenizer.tokenize(name)
                 if t.part_of_speech.startswith("名詞") and len(t.surface) > 1]
        vocab[cat]["vocab"].extend(words)
        for block in SEMANTIC_TEMPLATES.keys():
            templates[cat][block] = SEMANTIC_TEMPLATES[block].copy()
    return vocab, templates
# ===============================
# Utility：句読点整形・トリム
# ===============================
def _clean_text(s: str) -> str:
    s = re.sub(r"[ 　]+", " ", s)
    s = s.replace("、、", "、").replace("。。", "。")
    s = s.replace("〜", "").replace("..", "。")
    return s.strip()

# ===============================
# ALT生成（記者＋SEOアナリスト視点）
# ===============================
def compose_alt_variations(name, brand, genre, vocab, templates, n=20):
    results = []
    base_words = vocab.get(genre, {}).get("vocab", [])
    if not base_words:
        base_words = [genre, name]
    for _ in range(n):
        parts = []
        for block in ["spec", "feature", "scene", "benefit"]:
            block_tpls = templates.get(genre, {}).get(block, [])
            if block_tpls:
                tpl = random.choice(block_tpls)
                kw = random.choice(base_words)
                parts.append(tpl.format(name=name, keyword=kw))
        text = "、".join(parts)
        alt = f"{name} {text}"
        alt = _clean_text(alt)
        # 句読点密度を制御し、110文字に丸め
        if len(alt) > 110:
            alt = alt[:110].rsplit("、", 1)[0]
        results.append(alt)
    return list(dict.fromkeys(results))  # 重複削除

# ===============================
# キャッチコピー生成（編集者視点・30〜60文字）
# ===============================
def compose_catchcopy(name, brand, genre, emotion, benefits):
    JP_MAX, JP_MIN = 60, 30
    base = f"{emotion}{genre}なら、{brand}の「{name}」"
    base = _clean_text(base)
    if len(base) < JP_MIN:
        for p in benefits:
            if len(base) >= JP_MIN:
                break
            base += "、" + p
    if len(base) > JP_MAX:
        base = base[:JP_MAX]
    if len(base) < JP_MIN:
        base += "。"  # 安全弁
    return base

# ===============================
# OpenAI補完（オプション）
# ===============================
async def enhance_with_openai(batch_texts):
    if not OPENAI_ENABLE or not OPENAI_API_KEY:
        return batch_texts  # 使わない場合はそのまま返す
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{OPENAI_API_BASE_URL}chat/completions"
    prompt = (
        "次のALTまたはキャッチコピーを自然で流暢な日本語に整えてください。\n"
        "句読点の過不足を直し、意味を保ったまま110文字以内にまとめます。"
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": "\n".join(batch_texts)}],
        "temperature": 0.5,
        "max_tokens": 3000
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.error(f"OpenAI補完失敗: {resp.status} {txt[:200]}")
                return batch_texts
            data = await resp.json()
            out = data["choices"][0]["message"]["content"].splitlines()
            return [o.strip() for o in out if o.strip()]

# ===============================
# メイン処理
# ===============================
async def main_async(input_csv="input.csv", output_csv="output_alts.csv"):
    df, name_col = load_input_csv(input_csv)
    tokenizer = Tokenizer()

    # 初期語彙・テンプレ生成
    local_vocab, local_templates = bootstrap_local_vocab_and_templates(df, name_col)

    # APIクライアント
    rakuten_client = MarketAPIClient(RAKUTEN_API_BASE_URL, RAKUTEN_APP_ID, "Rakuten")
    yahoo_client = MarketAPIClient(YAHOO_API_BASE_URL, YAHOO_APP_ID, "Yahoo")

    logger.info("🪄 市場語彙辞書構築開始（楽天）")
    rakuten_vocab = await build_vocab_dictionary(rakuten_client, df, name_col, tokenizer)
    logger.info("🪄 市場語彙辞書構築開始（Yahoo）")
    yahoo_vocab = await build_vocab_dictionary(yahoo_client, df, name_col, tokenizer)

    # 語彙統合
    market_vocab = defaultdict(lambda: defaultdict(list))
    for src in [rakuten_vocab, yahoo_vocab, local_vocab]:
        for cat, data in src.items():
            market_vocab[cat]["vocab"].extend(data.get("vocab", []))

    templates = inject_market_vocabulary(local_templates, market_vocab)

    # 生成実行
    outputs = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        brand = str(row.get("ブランド名", "")) if "ブランド名" in df.columns else ""
        genre = infer_category(name, tokenizer)
        vocab = market_vocab
        alts = compose_alt_variations(name, brand, genre, vocab, templates, n=20)
        catch = compose_catchcopy(name, brand, genre, "", ["ギフトにも人気", "毎日にちょうどいい", "レビュー高評価"])
        if OPENAI_ENABLE and OPENAI_USE_BATCH:
            alts = await enhance_with_openai(alts)
        outputs.append({
            "name": name,
            "category": genre,
            "catchcopy": catch,
            "alts": " | ".join(alts)
        })

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 出力完了: {output_csv} ({len(out_df)}件)")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
import atlas_autosave_core
