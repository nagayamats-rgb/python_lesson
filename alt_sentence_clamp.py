#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT 自然文(_sentence.csv)を 140 文字以内に“意味をできるだけ保ったまま”クランプする後処理。
- 句点・終止記号で区切って詰める
- なお入らない残りは「…」で丸める
- 軽いノイズ除去（装飾記号/短い括弧注釈）も任意でON
"""
import re
import sys
import argparse
from pathlib import Path

import pandas as pd

DECOR_RE = re.compile(r'[※★◆▶→⇒⇨●◎○☆★•◇■□♪♬✦✧]+')
PAREN_SHORT_RE = re.compile(r'（[^）]{0,40}）')

def light_clean(s: str) -> str:
    s = DECOR_RE.sub('', s)
    s = PAREN_SHORT_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def smart_trim(s: str, limit: int = 140, clean: bool = True) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if clean:
        s = light_clean(s)
    if len(s) <= limit:
        return s
    # 句点等で分割して、収まるまで詰める
    parts = re.split(r'(?<=[。．!?！？])', s)
    out = ""
    for seg in parts:
        if not seg:
            continue
        if len(out) + len(seg) <= limit:
            out += seg
        else:
            # それでも入らないときは「…」で丸める
            remain = limit - len(out)
            if remain > 3:
                out += seg[:remain-1] + "…"
            break
    if not out:
        out = s[:limit-1] + "…"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="*_sentence.csv のパス")
    ap.add_argument("--limit", type=int, default=140)
    ap.add_argument("--no-clean", action="store_true", help="軽いノイズ除去を無効化")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[clamp] file not found: {p}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(p)
    alt_col = "alt_text" if "alt_text" in df.columns else next(c for c in df.columns if "alt" in c.lower())
    df[alt_col] = df[alt_col].map(lambda x: smart_trim(x, args.limit, clean=(not args.no_clean)))

    out = p.with_name(p.stem + "_clamped.csv")
    df.to_csv(out, index=False)

    alts = df[alt_col].astype(str)
    print(f"[clamp] rows={len(df)}  empty={(alts.str.strip()== '').sum()}  >{args.limit}={(alts.str.len()>args.limit).sum()}")
    print(f"[clamp] sentence_like={(alts.str.match(r'.*[。．!?！？…]$')).sum()}  wrote={out}")

if __name__ == "__main__":
    main()
