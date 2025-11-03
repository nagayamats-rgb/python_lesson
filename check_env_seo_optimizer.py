#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# check_env_seo_optimizer.py
# v1.0 — 環境変数検査スクリプト

import os
import sys
from pathlib import Path

def load_env_file(path=".env"):
    """最小限の.envローダー（dotenv未使用）"""
    env_path = Path(path)
    if not env_path.exists():
        print(f"⚠️ .env ファイルが見つかりません ({env_path.resolve()})")
        return {}
    env = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def color(txt, ok=True):
    return f"\033[92m{txt}\033[0m" if ok else f"\033[91m{txt}\033[0m"

def main():
    env = os.environ.copy()
    env_file = load_env_file()
    env.update(env_file)  # .env優先

    required = ["SEO_MARKET_API_URL", "SEO_MARKET_API_KEY"]
    optional = ["SEO_CONCURRENCY", "SEO_API_MAX_RETRIES", "SEO_API_RL_SLEEP"]
    error = False

    print("\n🔍 SEO Optimizer 用 .env チェック\n")

    for key in required:
        val = env.get(key, "").strip()
        if not val:
            print(color(f"❌ 必須キー {key} が設定されていません。", False))
            error = True
        else:
            print(color(f"✅ {key} = {val[:60] + ('...' if len(val) > 60 else '')}"))

    for key in optional:
        val = env.get(key)
        if val:
            print(color(f"ℹ️ {key} = {val}"))
        else:
            print(color(f"⚠️ {key} は未設定（デフォルト値を使用）", False))

    # 値の妥当性チェック
    try:
        if "SEO_CONCURRENCY" in env:
            c = int(env["SEO_CONCURRENCY"])
            if not (1 <= c <= 32):
                raise ValueError
        if "SEO_API_MAX_RETRIES" in env:
            r = int(env["SEO_API_MAX_RETRIES"])
            if not (1 <= r <= 10):
                raise ValueError
        if "SEO_API_RL_SLEEP" in env:
            s = int(env["SEO_API_RL_SLEEP"])
            if not (1 <= s <= 600):
                raise ValueError
    except ValueError:
        print(color("❌ 数値型環境変数に不正な値があります。", False))
        error = True

    print("\n✅ チェック完了。" if not error else "\n🚫 エラーが見つかりました。")
    sys.exit(0 if not error else 1)

if __name__ == "__main__":
    main()
import atlas_autosave_core
