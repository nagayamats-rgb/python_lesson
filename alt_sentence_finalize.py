# alt_sentence_finalize.py
# -*- coding: utf-8 -*-
"""
ALT自然文仕上げポストプロセッサ
- *_alt_details_*.csv（現状フレーズALT）と syntactic_review_*.csv（自然文束）を参照
- ALTを「句点で終わる一文」に整形し、禁則語を除去、140字上限を強制
- 既存ファイルは残しつつ、*_alt_details_*_sentence.csv を追生成（人間目視しやすい）

使い方:
  source .venv/bin/activate
  python alt_sentence_finalize.py --source yahoo   --run-tag 20251105_1313
  python alt_sentence_finalize.py --source rakuten --run-tag 20251105_1315

run-tag はファイル名に含まれる日時タグ。省略すると最新の *_alt_details_{source}_*.csv を自動選択。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "semantics"

# 禁則・装飾・UIノイズ
RE_BANNED = re.compile(r"(画像|写真|pic|image)", re.IGNORECASE)
RE_DECOR = re.compile(r"[↓→▶★※•◆◇■□●◎○▲△▼▽☆♪♬✦✧【】＜＞<>（）\(\)「」『』\[\]]+")
RE_WS = re.compile(r"\s+")
RE_MULTI_PUNCT = re.compile(r"[、,]{2,}")
RE_UI_NOISE = re.compile(
    r"(カート|お気に入り|不適切な商品を報告|ログイン|クーポン|アプリ|特典|ペイ|JavaScript|iframe|ヘルプ|"
    r"ストア|友だち追加|カテゴリ|特集|在庫|注文履歴|マイページ)"
)

RE_SENT_END = re.compile(r"[。．！？!?]$")

# 140字上限
ALT_MAXLEN = 140


def _shorten(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _clean_phrase(s: str) -> str:
    s = s or ""
    s = RE_DECOR.sub("", s)
    s = RE_WS.sub(" ", s).strip()
    s = RE_MULTI_PUNCT.sub("、", s)
    return s


def _pick_hint_from_syntactic(join_text: str) -> Optional[str]:
    """
    構文束から一文ヒントをもらう。
    - 最初の1文をざっくり抽出（読点や文点で分割）
    - UIノイズ/装飾を除去し、短文化（〜30字程度）
    """
    if not join_text:
        return None
    # 粗く文分割
    parts = re.split(r"[。．!?！？]\s*", join_text)
    for p in parts:
        p = _clean_phrase(p)
        if not p:
            continue
        if RE_UI_NOISE.search(p):
            continue
        # あまり長いとALTに不向きなので短文化
        if len(p) > 32:
            p = _shorten(p, 32)
        if p:
            return p
    return None


def _finalize_sentence(phrase: str,
                       product_name: str,
                       syn_hint: Optional[str]) -> str:
    """
    フレーズALT + 構文ヒント + 商品名 から「短い一文」を作る。
    ルール:
      - 禁則語を削り、装飾除去
      - 短すぎる/空になったら product_name の短縮を足す
      - 最後は句点で終わらせる
      - 140字以内
    """
    p0 = _clean_phrase(phrase)
    p0 = RE_BANNED.sub("", p0).strip()

    # まっさらなら product_name ベースに
    if not p0:
        base = _clean_phrase(product_name)
    else:
        base = p0

    # 構文ヒントを（あれば）後ろに足す。ただし重複・ノイズ回避
    segs: List[str] = []
    if base:
        segs.append(base)
    if syn_hint and syn_hint not in base and not RE_BANNED.search(syn_hint):
        segs.append(syn_hint)

    sent = " ".join(segs).strip()
    sent = RE_WS.sub(" ", sent)

    # 末尾句点
    if sent and not RE_SENT_END.search(sent):
        sent = sent + "。"

    # 140字上限
    if len(sent) > ALT_MAXLEN:
        # 句点を保ちながら切る
        cut = ALT_MAXLEN - 1
        sent = sent[:cut].rstrip("、, ・/　 ") + "。"

    # 万が一空
    if not sent:
        pn = _clean_phrase(product_name) or "アイテム"
        sent = f"{pn}の外観。"

    return sent


def _read_latest_csv(pattern: str) -> Tuple[pd.DataFrame, str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    latest = files[-1]
    return pd.read_csv(latest), latest


def _load_inputs(source: str, run_tag: Optional[str]) -> Dict[str, Tuple[pd.DataFrame, str]]:
    d: Dict[str, Tuple[pd.DataFrame, str]] = {}

    if run_tag:
        alt_pat = str(OUT_DIR / f"{source}_alt_details_{run_tag}.csv")
        syn_pat = str(OUT_DIR / f"syntactic_review_{source}_{run_tag}.csv")
        norm_pat = str(OUT_DIR / f"normalized_preview_{source}_{run_tag}.csv")
        d["alt"], _ = _read_latest_csv(alt_pat), alt_pat
        d["syn"], _ = _read_latest_csv(syn_pat), syn_pat
        d["norm"], _ = _read_latest_csv(norm_pat), norm_pat
    else:
        d["alt"] = _read_latest_csv(str(OUT_DIR / f"{source}_alt_details_*.csv"))
        d["syn"] = _read_latest_csv(str(OUT_DIR / f"syntactic_review_{source}_*.csv"))
        d["norm"] = _read_latest_csv(str(OUT_DIR / f"normalized_preview_{source}_*.csv"))

    return d


def _build_syn_map(df_syn: pd.DataFrame) -> Dict[str, str]:
    # product_name をキーに syntactic_join を引く単純マップ
    # 列名ゆれに耐える
    name_col = None
    join_col = None
    for c in df_syn.columns:
        if re.fullmatch("product_name", c, re.IGNORECASE):
            name_col = c
        if re.search(r"(syntactic_join|syntactic|text)", c, re.IGNORECASE):
            join_col = c
    if not name_col or not join_col:
        raise ValueError(f"syntactic_review: columns not found (name={name_col}, join={join_col})")
    mp: Dict[str, str] = {}
    for _, row in df_syn.iterrows():
        pn = str(row[name_col])
        jt = str(row[join_col]) if pd.notna(row[join_col]) else ""
        mp[pn] = jt
    return mp


def _choose_name_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c == "product_name":
            return c
    # fallback
    for c in df.columns:
        if "name" in c:
            return c
    return df.columns[0]


def main():
    ap = argparse.ArgumentParser(description="Finalize ALT as natural sentences.")
    ap.add_argument("--source", choices=["yahoo", "rakuten"], required=True)
    ap.add_argument("--run-tag", default=None, help="例: 20251105_1313（省略で最新自動）")
    ap.add_argument("--overwrite", action="store_true", help="既存alt_detailsを上書き（通常は新規*_sentence.csvを出力）")
    args = ap.parse_args()

    inputs = _load_inputs(args.source, args.run_tag)
    (df_alt, alt_path) = inputs["alt"]
    (df_syn, syn_path) = inputs["syn"]
    (df_norm, norm_path) = inputs["norm"]

    syn_map = _build_syn_map(df_syn)

    # 列名把握
    alt_col = None
    name_col = None
    for c in df_alt.columns:
        if c == "alt_text":
            alt_col = c
        if c == "product_name":
            name_col = c
    if alt_col is None:
        # altっぽい列
        for c in df_alt.columns:
            if re.search(r"alt", c, re.IGNORECASE):
                alt_col = c
                break
    if name_col is None:
        name_col = _choose_name_col(df_alt)

    if alt_col is None:
        raise ValueError("alt_details: alt_text 列が見つかりません")

    new_alts: List[str] = []
    for _, row in df_alt.iterrows():
        alt_phrase = str(row[alt_col]) if pd.notna(row[alt_col]) else ""
        pn = str(row[name_col]) if pd.notna(row[name_col]) else ""
        syn_join = syn_map.get(pn, "")

        syn_hint = _pick_hint_from_syntactic(syn_join)
        sent = _finalize_sentence(alt_phrase, pn, syn_hint)
        new_alts.append(sent)

    df_out = df_alt.copy()
    df_out["alt_text"] = new_alts

    # 出力
    if args.overwrite:
        out_path = alt_path  # 上書き
    else:
        # *_sentence.csv に
        p = Path(alt_path)
        out_path = str(p.with_name(p.stem + "_sentence" + p.suffix))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[alt_sentence_finalize] wrote -> {out_path}")

    # 品質要約
    empty = int(df_out["alt_text"].fillna("").str.strip().eq("").sum())
    over = int(df_out["alt_text"].astype(str).str.len().gt(ALT_MAXLEN).sum())
    banned = int(df_out["alt_text"].astype(str).str.contains(RE_BANNED).sum())
    sentence_like = float(df_out["alt_text"].astype(str).str.endswith(tuple("。．!?！？")).mean())
    print(f"empty={empty}, >140={over}, banned={banned}, sentence_rate={sentence_like:.2f}")


if __name__ == "__main__":
    main()
