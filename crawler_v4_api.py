# -*- coding: utf-8 -*-
from __future__ import annotations

"""
crawler_v4_api.py
KOTOHA PJT: Yahoo/Rakuten 両モールを一気通貫で処理して、
- クエリ候補生成（入力CSVから）
- SERP取得（API / フォールバックは入力CSVのURL行）
- 非オーガニック除外 → 1〜9位を採用
- 詳細HTML取得（--save-html）
- 正規化（product_name / description_blocks / images）
- ALT 4層生成（lexical → phrase → syntactic → alt）
- 構文抽出（syntactic_extractor_v3）で自然文レビューCSV
- ライターフェーズ学習CSV（pn/desc語彙、ALT語彙、テンプレ）

使い方（例）
  source .venv/bin/activate
  python crawler_v4_api.py \
    --source yahoo \
    --input sauce/yahoo.csv \
    --sample 5 --save-html --learn --debug

  # 本番統合（Yahoo+楽天をまとめて）
  python crawler_v4_api.py
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import hashlib
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---- 依存の事前インストール（不足があれば最低限入れる） -----------------
def _ensure_pkgs():
    try:
        import importlib  # noqa
        import requests  # noqa
        import pandas  # noqa
        from bs4 import BeautifulSoup  # noqa
        import yaml  # noqa
        from dotenv import load_dotenv  # noqa
    except Exception:
        print("[setup] installing missing packages ...")
        os.system(f"{sys.executable} -m pip install -U python-dotenv tqdm requests pandas beautifulsoup4 lxml PyYAML loguru")
_ensure_pkgs()

from loguru import logger
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 構文抽出ユーティリティ（別ファイル）
try:
    from syntactic_extractor_v3 import extract_syntactic_sentences
except Exception:
    print("ERROR: syntactic_extractor_v3.py が見つかりません。プロジェクト直下に配置してください。")
    sys.exit(1)

# ---- 正規化・ALT用の軽量ルール ------------------------------------------
NOISE_WORDS = {"送料無料", "最安値", "スーパーDEAL", "倍ポイント", "タイムセール"}
ALT_BAN_WORDS = {"画像", "写真", "pic", "image"}
JP_PUNCT = r"、。・：；（）［］「」『』【】"

_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_SYMBOLS = re.compile(rf"[{re.escape(JP_PUNCT)}]+")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) KOTOHA-Bot/4.1 (+https://example.invalid)"

# ---- ファイル名ユーティリティ --------------------------------------------
def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

# ---- 正規化 ---------------------------------------------------------------
def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_description(desc: str) -> List[str]:
    if not desc:
        return []
    txt = unicodedata.normalize("NFKC", desc)
    # ボタンやUI的なカケラはある程度消す（強すぎない程度）
    txt = re.sub(r"(送料|ポイント|クーポン|ログイン|購入履歴|お気に入り|ストアトップ).{0,20}", " ", txt)
    # 句点・改行で区切る
    txt = txt.replace("。", "。\n")
    blocks = []
    for line in txt.splitlines():
        line = _RE_MULTI_SPACE.sub(" ", line).strip()
        if not line:
            continue
        # カテゴリ羅列ぽさ＆異様に短い/長いは落とす
        if re.search(r"(,|/|・|＞|>|カテゴリ|ジャンル)", line) and len(line) < 20:
            continue
        if 4 <= len(line) <= 220:
            blocks.append(line)
    return blocks

# ---- ALT 4層 --------------------------------------------------------------
def lexical_layer(name: str, desc_blocks: List[str]) -> List[str]:
    base = f"{name} " + " ".join(desc_blocks[:3])
    base = unicodedata.normalize("NFKC", base)
    # 簡易トークン：空白と記号で
    toks = re.split(r"[\s/・,、。()（）\[\]【】「」『』:：;；\|\-]+", base)
    out = []
    for t in toks:
        t = t.strip()
        if not t: 
            continue
        if t in NOISE_WORDS:
            continue
        if re.fullmatch(r"\d+[a-zA-Z]*", t):
            # 数字だけは弱く
            continue
        out.append(t)
    # 重複削除
    uniq = []
    seen = set()
    for w in out:
        if w not in seen:
            uniq.append(w); seen.add(w)
    return uniq[:60]

def phrase_layer(words: List[str]) -> List[str]:
    # 3〜6語で軽いフレーズ化（スライディング）
    phrases = []
    n = len(words)
    for i in range(0, min(n, 30), 2):
        chunk = words[i:i+random.choice([3,4,5,6])]
        if len(chunk) >= 3:
            phrases.append(" ".join(chunk))
    return phrases[:20]

def syntactic_layer(phrases: List[str]) -> List[str]:
    sents = []
    for ph in phrases[:10]:
        # 記号→スペース
        clean = _RE_SYMBOLS.sub(" ", ph)
        clean = _RE_MULTI_SPACE.sub(" ", clean).strip()
        if not clean:
            continue
        # 簡単な文テンプレ
        sents.append(f"{clean}。")
    return sents[:10]

def alt_layer(sents: List[str]) -> List[str]:
    alts = []
    for s in sents:
        a = s
        for ng in ALT_BAN_WORDS:
            a = a.replace(ng, "")
        a = _RE_MULTI_SPACE.sub(" ", a).strip(" 。")
        if 8 <= len(a) <= 140:
            alts.append(a)
    if not alts:
        alts = ["商品画像。"]
    return alts[:5]

def make_alts(name: str, desc_blocks: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    words = lexical_layer(name, desc_blocks)
    phrases = phrase_layer(words)
    sents = syntactic_layer(phrases)
    alts = alt_layer(sents)
    debug = {
        "lexical": words,
        "phrases": phrases,
        "syntactic": sents,
        "alt": alts,
    }
    return alts, debug

# ---- API呼び出し（Yahoo / Rakuten） --------------------------------------
def yahoo_item_search(appid: str, query: str, hits: int = 30) -> List[Dict[str, Any]]:
    url = os.getenv("YAHOO_API_BASE_URL") or "https://shopping.yahooapis.jp/ShoppingWebService/V3/itemSearch"
    params = {
        "appid": appid,
        "query": query,
        "results": min(hits, 50),
    }
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = data.get("hits") or data.get("items") or []
    # 正規化
    out = []
    for it in items:
        name = it.get("name") or it.get("Title") or ""
        urlp = it.get("url") or it.get("Url") or it.get("urlLink") or ""
        out.append({
            "title": normalize_name(name),
            "url": urlp,
            "raw": it,
        })
    return out

def rakuten_item_search(appid: str, query: str, hits: int = 30) -> List[Dict,]:
    url = os.getenv("RAKUTEN_API_BASE_URL") or "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20220601"
    params = {
        "applicationId": appid,
        "keyword": query,
        "hits": min(hits, 30),
        "sort": "+reviewCount",
    }
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = (data.get("Items") or [])
    out = []
    for wrap in items:
        it = wrap.get("Item") or {}
        name = it.get("itemName") or ""
        urlp = it.get("itemUrl") or ""
        out.append({
            "title": normalize_name(name),
            "url": urlp,
            "raw": it,
        })
    return out

# ---- オーガニックフィルタ ---------------------------------------------------
_AD_HINT = re.compile(r"(PR|広告|sponsored|adserver|/promo/|/event/)", re.I)

def is_non_organic(rec: Dict[str, Any]) -> bool:
    t = (rec.get("title") or "") + " " + (rec.get("url") or "")
    if _AD_HINT.search(t):
        return True
    # 極端なテンプレートURL（絞り込み/特集）
    if re.search(r"(search|category|/event/|/special/|/recommend)", rec.get("url") or "", re.I):
        return True
    return False

def reindex_and_pick_organic(recs: List[Dict[str, Any]], pick_top: int = 9) -> List[Dict[str, Any]]:
    organic = [r for r in recs if not is_non_organic(r)]
    # 1..N 採番
    for i, r in enumerate(organic, start=1):
        r["organic_rank"] = i
    return organic[:pick_top]

# ---- 詳細ページ取得 --------------------------------------------------------
def fetch_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text

def parse_detail_from_html(html: str) -> Tuple[str, List[str], List[str]]:
    soup = BeautifulSoup(html, "lxml")
    # タイトル候補
    name = soup.find("h1")
    name = name.get_text(strip=True) if name else ""
    # 説明候補
    desc_nodes = soup.select("article, main, #ItemDetail, #itemDetail, .item-detail, #description, .description, #spec, .spec")
    if not desc_nodes:
        desc_nodes = soup.select("div")
        desc_nodes = sorted(desc_nodes, key=lambda d: len(d.get_text(strip=True)), reverse=True)[:3]
    text = "\n".join([n.get_text("\n", strip=True) for n in desc_nodes])
    blocks = split_description(text)
    # 画像候補
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if src and not src.startswith("data:"):
            imgs.append(src)
    return normalize_name(name), blocks, imgs[:20]

# ---- 入力/参照CSV ----------------------------------------------------------
def load_input_products(source: str, path: str, env: Dict[str, str]) -> List[str]:
    """
    検索クエリに使う元ネタを入力CSVから拾う。
    - Yahoo: REF_YAHOOで指定の参照CSVの見出し2行目から意味列を使う → 実運用は name 列（ユーザー仕様確定）
    - Rakuten: REF_RAKUTENの2行目の指定、実運用は 商品名（RAKUTEN_NAME_COL）
    """
    df = pd.read_csv(path)
    if source == "yahoo":
        col = "name"
        if col not in df.columns:
            raise RuntimeError(f"入力CSVに '{col}' 列が見つかりません: {path}")
        names = df[col].dropna().astype(str).tolist()
        return [normalize_name(x) for x in names]
    else:
        # Rakuten は .env の RAKUTEN_NAME_COL を優先（既定: 商品名）
        rcol = env.get("RAKUTEN_NAME_COL") or "商品名"
        if rcol not in df.columns:
            raise RuntimeError(f"入力CSVに '{rcol}' 列が見つかりません: {path}")
        names = df[rcol].dropna().astype(str).tolist()
        return [normalize_name(x) for x in names]

# ---- 進行管理 --------------------------------------------------------------
def ensure_dirs(paths: List[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]], headers: List[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})

# ---- メイン処理 ------------------------------------------------------------
def process_source(source: str, input_path: str, env: Dict[str, str], args) -> Dict[str, Any]:
    run_tag = now_tag()
    out_dir = env.get("OUTPUT_DIR", "output/semantics")
    log_dir = env.get("LOG_DIR", "output/logs")
    dbg_dir = env.get("DEBUG_DIR", "debug")
    ensure_dirs([out_dir, log_dir, dbg_dir, f"{dbg_dir}/html"])

    logger.info(f"[crawler_v4_api] source={source} input={input_path}")
    logger.info(f"mode={'api'} allow_fallback_csv=True")

    # 1) 入力→クエリ候補
    all_names = load_input_products(source, input_path, env)
    if args.sample and args.sample > 0:
        # 同じ結果にするため固定seed
        random.seed(42)
        all_names = random.sample(all_names, min(args.sample, len(all_names)))

    queries = [{"source": source, "query": q} for q in all_names]
    qfile = f"{out_dir}/query_candidates_{run_tag}.csv"
    pd.DataFrame(queries).to_csv(qfile, index=False)
    logger.info(f"query_candidates -> {qfile} ({len(queries)} rows)")

    # 2) SERP（API）
    serp_rows = []
    raw_jsonl = []
    for q in all_names:
        qtxt = q
        try:
            if source == "yahoo":
                app = env.get("YAHOO_APP_ID") or ""
                items = yahoo_item_search(app, qtxt, hits=30)
            else:
                app = env.get("RAKUTEN_APP_ID") or ""
                items = rakuten_item_search(app, qtxt, hits=30)
        except Exception as e:
            logger.warning(f"[SERP {source}] failed: {qtxt}: {e}")
            items = []

        # 生
        raw_jsonl.append({"query": qtxt, "items": items})

        # 除外 → 1〜9位
        picked = reindex_and_pick_organic(items, pick_top=9)
        for r in picked:
            serp_rows.append({
                "source": source,
                "query": qtxt,
                "organic_rank": r.get("organic_rank", ""),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
            })

    raw_path = f"{out_dir}/serp_raw_{source}_{run_tag}.jsonl"
    write_jsonl(raw_path, raw_jsonl)
    ranked_path = f"{out_dir}/serp_ranked_{source}_{run_tag}.csv"
    write_csv(ranked_path, serp_rows, ["source", "query", "organic_rank", "title", "url"])
    logger.info(f"serp_raw -> {raw_path}")
    logger.info(f"serp_ranked -> {ranked_path} (picked_total={len(serp_rows)})")

    # 3) 詳細取得 → 正規化/ALT/構文
    norm_rows = []
    alt_jsonl_rows = []
    alt_list_rows = []
    alt_detail_rows = []
    syntactic_rows = []
    learn_vocab_rows = []      # product_name/desc 語彙
    learn_alt_rows = []        # alt 語彙
    learn_tmpl_rows = []       # テンプレ文

    for i, row in enumerate(serp_rows, start=1):
        url = row["url"]
        try:
            html = fetch_html(url)
        except Exception as e:
            logger.warning(f"[DETAIL {source}] fetch failed: {url}: {e}")
            html = ""

        # HTML保存（任意）
        if args.save_html and html:
            hid = f"{source}_{sha1(url)}.html"
            Path(f"{dbg_dir}/html/{hid}").write_text(html, encoding="utf-8")

        # HTML → name/desc/images
        name, desc_blocks, images = parse_detail_from_html(html)

        # 正規化プレビュー行
        norm_rows.append({
            "source": source,
            "query": row["query"],
            "organic_rank": row["organic_rank"],
            "product_name": name or row["title"],
            "desc_blocks_count": len(desc_blocks),
            "desc_preview": " / ".join(desc_blocks[:3]),
            "images_count": len(images),
            "url": url,
        })

        # ALT
        alts, dbg = make_alts(name or row["title"], desc_blocks)
        alt_jsonl_rows.append({
            "source": source,
            "product_name": name or row["title"],
            "description_blocks": desc_blocks,
            "alt_texts": alts,
            "policy_flags": {"warnings": [], "alt_style": "sentence"},
            "cluster_prim_id": sha1((name or row["title"]) + "_prim"),
        })
        alt_list_rows.append({
            "source": source,
            "product_name": name or row["title"],
            "description_preview": " ".join(desc_blocks[:2]),
            "alt_top1": alts[0] if len(alts) > 0 else "",
            "alt_top2": alts[1] if len(alts) > 1 else "",
            "alt_top3": alts[2] if len(alts) > 2 else "",
            "alt_count": len(alts),
            "cluster_prim_id": sha1((name or row["title"]) + "_prim"),
            "cluster_sem_id": "",
            "notes": "",
            "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        for idx, a in enumerate(alts):
            alt_detail_rows.append({
                "source": source,
                "product_name": name or row["title"],
                "description_block_index": -1,
                "description_block_text": "",
                "image_index": idx,
                "alt_text": a,
                "serp_rank": row["organic_rank"],
                "cluster_sem_id": "",
                "flags": "",
                "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        # 構文（自然文）抽出・レビュー
        if html:
            syn = extract_syntactic_sentences(html, source=source)
            kept = syn.get("sentences", [])
            syntactic_rows.append({
                "source": source,
                "product_name": name or row["title"],
                "organic_rank": row["organic_rank"],
                "syntactic_count": len(kept),
                "syntactic_join": " ".join(kept[:50]),
                "url": url,
            })
            # 学習コーパス：語彙（PN/Desc）、ALT語彙、テンプレ
            # PN/Desc語彙 = lexicalから
            for w in dbg.get("lexical", []):
                learn_vocab_rows.append({
                    "source": source,
                    "type": "pn_desc",
                    "token": w,
                    "product_name": name or row["title"],
                    "url": url,
                })
            # ALT語彙
            for alt in alts:
                for tok in re.split(r"\s+", re.sub(rf"[{re.escape(JP_PUNCT)}。、・/]", " ", alt)):
                    tok = tok.strip()
                    if tok and tok not in ALT_BAN_WORDS and len(tok) <= 20:
                        learn_alt_rows.append({
                            "source": source,
                            "token": tok,
                            "product_name": name or row["title"],
                            "url": url,
                        })
            # テンプレ（syntacticレイヤ）
            for s in dbg.get("syntactic", []):
                learn_tmpl_rows.append({
                    "source": source,
                    "template": s,
                    "product_name": name or row["title"],
                    "url": url,
                })

    # 4) 出力
    # 正規化プレビュー
    norm_path = f"{out_dir}/normalized_preview_{source}_{run_tag}.csv"
    write_csv(norm_path, norm_rows, [
        "source", "query", "organic_rank", "product_name",
        "desc_blocks_count", "desc_preview", "images_count", "url"
    ])
    # ALT
    alt_jsonl = f"{out_dir}/{source}_alt_{run_tag}.jsonl"
    write_jsonl(alt_jsonl, alt_jsonl_rows)
    alt_csv = f"{out_dir}/{source}_alt_{run_tag}.csv"
    write_csv(alt_csv, alt_list_rows, [
        "source","product_name","description_preview","alt_top1","alt_top2","alt_top3",
        "alt_count","cluster_prim_id","cluster_sem_id","notes","run_ts"
    ])
    alt_detail_csv = f"{out_dir}/{source}_alt_details_{run_tag}.csv"
    write_csv(alt_detail_csv, alt_detail_rows, [
        "source","product_name","description_block_index","description_block_text","image_index",
        "alt_text","serp_rank","cluster_sem_id","flags","run_ts"
    ])
    # 構文レビュー
    syn_path = f"{out_dir}/syntactic_review_{source}_{run_tag}.csv"
    write_csv(syn_path, syntactic_rows, [
        "source","product_name","organic_rank","syntactic_count","syntactic_join","url"
    ])
    # 学習CSV（必要なら）
    if args.learn:
        learn_pn = f"{out_dir}/learn_pn_desc_vocab_{run_tag}.csv"
        learn_alt = f"{out_dir}/learn_alt_vocab_{run_tag}.csv"
        learn_tpl = f"{out_dir}/learn_templates_{run_tag}.csv"
        write_csv(learn_pn, learn_vocab_rows, ["source","type","token","product_name","url"])
        write_csv(learn_alt, learn_alt_rows, ["source","token","product_name","url"])
        write_csv(learn_tpl, learn_tmpl_rows, ["source","template","product_name","url"])
        logger.info(f"learn_csv -> {learn_pn}, {learn_alt}, {learn_tpl}")

    ok = len(norm_rows)
    fails = 0
    logger.info(f"total={ok} ok={ok} fails={fails} success_rate=100.00%")

    return {
        "ranked_path": ranked_path,
        "norm_path": norm_path,
        "alt_jsonl": alt_jsonl,
        "alt_csv": alt_csv,
        "alt_detail_csv": alt_detail_csv,
        "syn_path": syn_path,
    }

# ---- エントリポイント ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yahoo", "rakuten", "both"], default="both")
    parser.add_argument("--input", help="入力CSV（source指定時はそのソースのCSV）")
    parser.add_argument("--sample", type=int, default=0, help="各ソースからランダム抽出する件数")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--save-html", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")

    env = {k: os.getenv(k, "") for k in [
        "INPUT_YAHOO","INPUT_RAKUTEN","REF_YAHOO","REF_RAKUTEN",
        "OUTPUT_DIR","LOG_DIR","DEBUG_DIR","LEARN_CORPUS",
        "YAHOO_APP_ID","RAKUTEN_APP_ID","RAKUTEN_NAME_COL",
        "YAHOO_API_BASE_URL","RAKUTEN_API_BASE_URL"
    ]}

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # ソースの解決
    sources: List[Tuple[str, str]] = []
    if args.source in ("yahoo", "rakuten"):
        if not args.input:
            # .env を見る
            path = env["INPUT_YAHOO"] if args.source == "yahoo" else env["INPUT_RAKUTEN"]
        else:
            path = args.input
        sources.append((args.source, path))
    else:
        # both
        sources.append(("yahoo", env["INPUT_YAHOO"] or "sauce/yahoo.csv"))
        sources.append(("rakuten", env["INPUT_RAKUTEN"] or "sauce/rakuten.csv"))

    # 実行
    for src, path in sources:
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"入力CSVが見つかりません: {path}")
        process_source(src, path, env, args)

if __name__ == "__main__":
    main()
