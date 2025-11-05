# -*- coding: utf-8 -*-
"""
syntactic_extractor_v3.py
Yahoo/Rakutenの商品ページHTMLから「構文学習に使える自然文」を抽出する小さなユーティリティ。

設計ポイント
- 強めのUIノイズ除去（価格/送料/ポイント/注意書き/レビュー/カテゴリ列挙/ボタン文言等）
- 本文と思しき主要領域を優先的に走査（article, main, #ItemDetail 等）
- 句点ベース + 記号除去の日本語センテンス分割（簡易）
- 過度に短い/長い/単語羅列っぽい行を除外
- 抽出理由デバッグ（どのノイズ規則に引っかかったか等）を返す

公開関数
- extract_syntactic_sentences(html: str, source: str) -> dict
  戻り値: {
    "sentences": [str, ...],      # 学習に使える自然文
    "dropped":   [ (text, reason), ... ], # 捨てたテキストと理由
  }
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup, NavigableString

# ====== 記号・デコレーションのノイズ ======
RE_SPACES = re.compile(r'\s+')
RE_URL = re.compile(r'https?://\S+')
RE_ASCII_NOISE = re.compile(r'[\[\]\(\){}<>|\\^~`=_+※]+')
RE_DECOR = re.compile(r'[↓→▶★◆◇■□●◎○▲△▼▽☆♪♬✦✧→⇒⇨・※　\s]{2,}')
RE_MULTISYM = re.compile(r'[,、。]{3,}')

# 文らしさ判定の閾値
MIN_LEN = 8
MAX_LEN = 180
MIN_KANJI_KANA_RATIO = 0.25   # 日本語っぽさ
MAX_TOKENS_PER_SENT = 50

# 楽天/Yahoo 共通で弾く断片（価格・送料・ポイント・レビューなど）
COMMON_DROP_PATTERNS = [
    r'送料無料', r'最安値', r'タイムセール', r'ポイント[0-9]+倍', r'倍ポイント',
    r'在庫処分', r'期間限定', r'クーポン', r'キャンペーン', r'セール', r'割引',
    r'レビュー\([0-9,]+\)', r'[0-9,]+件のレビュー', r'評価', r'星[0-9.]+',
    r'税込', r'税抜', r'送料', r'全国一律', r'条件により送料が異なる',
    r'不適切な商品を報告', r'カートに追加', r'お気に入り', r'注文履歴',
    r'ログイン', r'新着情報', r'マイページ', r'ヘルプ',
    r'JavaScriptが無効', r'ブラウザ.*有効', r'詳細はこちら', r'会社概要', r'問い合わせ',
    r'カテゴリ', r'特集', r'ストアトップ', r'ショップ', r'店舗', r'在庫', r'納期',
    r'価格', r'円\b', r'PayPay', r'ポイント', r'付与', r'特典', r'上限',
    r'お知らせ', r'閲覧履歴', r'購入履歴',
]
RE_COMMON_DROP = re.compile('|'.join(COMMON_DROP_PATTERNS))

# Yahoo特有のUI/ガワ文言
YAHOO_UI_NOISE = [
    r'Yahoo!? JAPAN', r'無料でお店を開こう', r'World Select', r'JAPAN\b',
    r'ストアをお気に入り', r'ストアトップを見る', r'ペイトク', r'アプリ.*クーポン',
]
RE_YAHOO_UI = re.compile('|'.join(YAHOO_UI_NOISE))

# 楽天特有のUI/ガワ文言
RAKUTEN_UI_NOISE = [
    r'楽天', r'カテゴリトップ', r'商品情報', r'商品詳細', r'ジャンル',
    r'この部分は iframe 対応のブラウザで見てください', r'ショップ', r'店舗',
]
RE_RAKUTEN_UI = re.compile('|'.join(RAKUTEN_UI_NOISE))

# 単語羅列っぽい（名詞連打/カテゴリだらけ等）を雑に検知
RE_LISTY = re.compile(r'(?:[,、・/＞>\s]|>|\s){1,}')  # 区切り記号が密集

# ---- 主要コンテンツらしい領域のセレクタ（優先度順） ----
MAIN_CANDIDATE_SELECTORS = [
    'article', 'main', '#main', '#contents', '#ItemDetail', '#itemDetail', '.itemDetail',
    '#productDetail', '.product-detail', '.productDetail', '#spec', '.spec', '#description', '.description',
    '#content', '.content', '.item_desc', '.item-description',
]


def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\u3000', ' ')
    s = RE_URL.sub('', s)
    s = RE_ASCII_NOISE.sub(' ', s)
    s = RE_DECOR.sub(' ', s)
    s = RE_SPACES.sub(' ', s).strip()
    return s


def looks_like_sentence(s: str) -> bool:
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    # 日本語文字の比率（かな・カナ・漢字）
    jp_chars = sum(1 for ch in s if ('\u3040' <= ch <= '\u30ff') or ('\u4e00' <= ch <= '\u9fff'))
    ratio = jp_chars / max(1, len(s))
    if ratio < MIN_KANJI_KANA_RATIO:
        return False
    # 記号連打やリストっぽさ
    if RE_MULTISYM.search(s):
        return False
    # 「,・/」の密度が高すぎる＝羅列臭
    if len(RE_LISTY.findall(s)) >= 5:
        return False
    # 語数の上限（雑に）
    if len(s.split()) > MAX_TOKENS_PER_SENT:
        return False
    return True


def drop_reason(s: str, source: str) -> str | None:
    if RE_COMMON_DROP.search(s):
        return 'common_ui_or_price'
    if source == 'yahoo' and RE_YAHOO_UI.search(s):
        return 'yahoo_ui'
    if source == 'rakuten' and RE_RAKUTEN_UI.search(s):
        return 'rakuten_ui'
    return None


def sentence_split_jp(text: str) -> List[str]:
    """
    超簡易：句点/改行で分割 → 正規化。
    """
    # 句点で分けつつ改行も分割子扱い
    text = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n')
    parts = []
    for line in text.splitlines():
        line = normalize_text(line)
        if line:
            parts.append(line)
    # 行内に句点がまだ複数あれば、保守的にそのまま残す（分解しすぎない）
    return parts


def pick_main_nodes(soup: BeautifulSoup) -> List:
    # 明示領域を優先
    for sel in MAIN_CANDIDATE_SELECTORS:
        nodes = soup.select(sel)
        if nodes:
            return nodes
    # fallback：本文っぽい div をざっくり
    large_divs = sorted(
        (div for div in soup.find_all('div')),
        key=lambda d: len(d.get_text(strip=True)),
        reverse=True
    )
    return large_divs[:5]


def extract_text_blocks(nodes) -> List[str]:
    blocks: List[str] = []
    for node in nodes:
        # 子要素のテキストをまとめて取得
        text = node.get_text(separator='\n', strip=True)
        text = normalize_text(text)
        if text:
            blocks.append(text)
    return blocks


def extract_syntactic_sentences(html: str, source: str) -> Dict[str, List]:
    """
    メイン抽出関数。
    :param html: 商品ページHTML
    :param source: 'yahoo' or 'rakuten'
    :return: {"sentences": [...], "dropped": [(text, reason), ...]}
    """
    source = (source or '').lower()
    if source not in ('yahoo', 'rakuten'):
        source = 'yahoo'

    soup = BeautifulSoup(html, 'lxml')

    # 主要候補ノード
    nodes = pick_main_nodes(soup)
    raw_blocks = extract_text_blocks(nodes)

    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []

    for block in raw_blocks:
        for sent in sentence_split_jp(block):
            if not sent:
                continue
            reason = drop_reason(sent, source)
            if reason:
                dropped.append((sent, reason))
                continue
            if looks_like_sentence(sent):
                kept.append(sent)
            else:
                dropped.append((sent, 'not_sentence_like'))

    # 重複を削る（順序保持）
    seen = set()
    uniq_kept = []
    for s in kept:
        if s not in seen:
            uniq_kept.append(s)
            seen.add(s)

    return {"sentences": uniq_kept, "dropped": dropped}


# 直接叩いたときの簡易テスト
if __name__ == '__main__':
    import sys, json, pathlib
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    src = sys.argv[2] if len(sys.argv) > 2 else 'yahoo'
    if not p or not p.exists():
        print("usage: python syntactic_extractor_v3.py <html_file> [yahoo|rakuten]")
        sys.exit(1)
    html = p.read_text(encoding='utf-8', errors='ignore')
    out = extract_syntactic_sentences(html, src)
    print(json.dumps({
        "preview_sentences": out["sentences"][:10],
        "dropped_examples": out["dropped"][:10]
    }, ensure_ascii=False, indent=2))
