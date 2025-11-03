# query_generator.py
"""
🌸 KOTOHA ENGINE v1.0 - query_generator.py
-------------------------------------------
目的:
- 商品名/ジャンルID（+あればテンプレ出力）から検索クエリ候補を生成
- ルールベースで確実に動作（AIなし）をデフォルト
- OPENAI_ENABLE=true の場合、低コストで自然なクエリを追加生成
- 中間生産物を段階保存して結合テスト/再実行を容易に

入出力:
- 入力: structured_preview.csv または最新の output_templates_*.csv
- 出力: query_seeds_*.csv / query_candidates_*.csv / query_batches_*.jsonl / logs/
"""

import os
import re
import csv
import json
import glob
import logging
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# OpenAI (任意)
OPENAI_OK = False
try:
    from openai import OpenAI  # v1.x
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# ----------------------------
# ロガー
# ----------------------------
logger = logging.getLogger("KOTOHA_QUERY")
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/query_generator_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
logger.setLevel(logging.INFO)

# ----------------------------
# 設定ロード
# ----------------------------
def load_configs():
    # 機密
    load_dotenv(".env.txt")
    openai_enable = os.getenv("OPENAI_ENABLE", "false").lower() == "true"
    openai_key = os.getenv("OPENAI_API_KEY")

    # プロジェクト設定
    if not os.path.exists("kotoha_config.json"):
        raise FileNotFoundError("kotoha_config.json がありません。init_config.py を実行してください。")
    with open("kotoha_config.json", "r", encoding="utf-8") as f:
        global_cfg = json.load(f)

    # モジュール設定（なければデフォルトで作成）
    mod_path = "config/modules/query_generator.json"
    if not os.path.exists(mod_path):
        os.makedirs("config/modules", exist_ok=True)
        default_cfg = {
            "description": "検索クエリ生成（ルール＋任意でAI）",
            "seed_max": 5,
            "candidates_per_item": 16,
            "include_longtail": True,
            "persona_mods": ["初心者向け", "ビジネス用", "学生向け", "ギフト", "高耐久", "軽量", "静音", "急速", "省スペース"],
            "scene_mods": ["在宅", "出張", "通勤", "旅行", "オフィス", "寝室", "リビング"],
            "generic_mods": ["比較", "おすすめ", "人気", "ランキング", "使い方", "口コミ", "レビュー", "選び方"]
        }
        with open(mod_path, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2, ensure_ascii=False)
        logger.info("🛠️ query_generator.json を新規作成しました（デフォルト設定）")

    with open(mod_path, "r", encoding="utf-8") as f:
        module_cfg = json.load(f)

    output_dir = global_cfg.get("OUTPUT_DIR", "./")
    return openai_enable, openai_key, output_dir, module_cfg

# ----------------------------
# 入力ファイル決定
# ----------------------------
def pick_input_file(output_dir="./"):
    tpl = sorted(glob.glob(os.path.join(output_dir, "output_templates_*.csv")))
    if tpl:
        logger.info(f"📄 入力: 最新テンプレートを使用します → {tpl[-1]}")
        return tpl[-1]
    sp = os.path.join(output_dir, "structured_preview.csv")
    if os.path.exists(sp):
        logger.info(f"📄 入力: structured_preview.csv を使用します")
        return sp
    raise FileNotFoundError("input が見つかりません（structured_preview.csv / output_templates_*.csv）")

# ----------------------------
# キーワード抽出（商品名→種語）
# ----------------------------
def extract_seed_keywords(name: str, seed_max=5):
    # ノイズ除去
    t = re.sub(r"[【】\[\]\(\)（）]", " ", str(name))
    t = re.sub(r"[0-9A-Za-z\-_/|:＋+＊*]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 日本語語片抽出（2文字以上）
    words = re.findall(r"[一-龥ぁ-んァ-ンー]{2,}", t)
    # 順番を保ちつつ重複除去
    seen, seeds = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            seeds.append(w)
        if len(seeds) >= seed_max:
            break
    return seeds or ([t[:6]] if t else [])

# ----------------------------
# ルールベースのクエリ展開
# ----------------------------
def expand_queries_rule(seeds, genre, cfg):
    base = list(seeds)
    mods = cfg.get("persona_mods", []) + cfg.get("scene_mods", []) + cfg.get("generic_mods", [])
    genre_part = f" {genre}" if genre else ""

    cand = set()

    # 1語/2語/修飾語の組み合わせ
    for s in base:
        cand.add(s + genre_part)
        for m in mods:
            cand.add(f"{s} {m}")
            cand.add(f"{s}{genre_part} {m}")

    # ロングテール（省エネで少しだけ）
    if cfg.get("include_longtail", True):
        heads = ["買い方", "比較", "おすすめ", "人気", "安い", "高品質", "最新", "型番", "純正", "互換"]
        for s in base:
            for h in heads:
                cand.add(f"{s} {h}")
                if genre:
                    cand.add(f"{s} {genre} {h}")

    # 簡易スコア（長さと重複控除）で並び替え
    scored = sorted(list(cand), key=lambda x: (len(x), x))
    # トップN
    N = max(10, cfg.get("candidates_per_item", 16))
    return scored[:N]

# ----------------------------
# AIで自然なクエリを追加（任意）
# ----------------------------
def expand_queries_ai(client, name, genre, seeds, want=8):
    prompt = (
        "以下の情報から、日本人ユーザーが実際に検索しそうな自然な検索クエリを箇条書きで出力してください。\n"
        "・商品名の特徴語\n"
        f"・ジャンル: {genre or '不明'}\n"
        f"・抽出済みキーワード: {', '.join(seeds)}\n"
        f"条件: 7〜12個、日本語、検索に実際に使いそうな短い語句、記号なし、重複しない\n"
        "出力: 各行1クエリのみ\n"
    )
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "あなたは日本語SEOの専門家です。"}, {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=180
        )
        text = res.choices[0].message.content.strip()
        lines = [re.sub(r"^[\-\d\.\s、・]+", "", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and len(ln) <= 30]
        # 上限調整
        uniq, seen = [], set()
        for q in lines:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
            if len(uniq) >= want:
                break
        return uniq
    except Exception as e:
        logger.warning(f"⚠️ OpenAI生成をスキップ（{e}）")
        return []

# ----------------------------
# メイン
# ----------------------------
def main():
    try:
        openai_enable, openai_key, output_dir, cfg = load_configs()
    except Exception as e:
        logger.error(f"設定読込エラー: {e}")
        return

    try:
        input_path = pick_input_file(output_dir)
        df = pd.read_csv(input_path, dtype=str).fillna("")
    except Exception as e:
        logger.error(f"入力読込エラー: {e}")
        return

    # 列名（あなたのCSV準拠）
    NAME, GENRE = "商品名", "ジャンルID"

    # 出力タイムスタンプ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds_path = os.path.join(output_dir, f"query_seeds_{ts}.csv")
    cands_path = os.path.join(output_dir, f"query_candidates_{ts}.csv")
    batch_path = os.path.join(output_dir, f"query_batches_{ts}.jsonl")

    # OpenAI クライアント（任意）
    client = None
    if openai_enable and OPENAI_OK and openai_key:
        client = OpenAI(api_key=openai_key)
        logger.info("🤝 OpenAIクエリ展開を有効化（低コスト）")
    else:
        if openai_enable and not OPENAI_OK:
            logger.warning("⚠️ openai SDK(v1) が見つかりません。AI展開は無効化します。")
        logger.info("🪄 ルールベースのみでクエリ生成します（AIコスト0）")

    # 生成結果格納
    seeds_rows = []
    cands_rows = []
    batch_items = []

    for _, row in df.iterrows():
        name = str(row.get(NAME, "")).strip()
        genre = str(row.get(GENRE, "")).strip()
        if not name:
            continue

        seeds = extract_seed_keywords(name, seed_max=cfg.get("seed_max", 5))
        seeds_rows.append({
            "商品名": name,
            "ジャンルID": genre,
            "seed_keywords": "|".join(seeds)
        })

        # ルールベース候補
        rule_qs = expand_queries_rule(seeds, genre, cfg)

        # AI追加候補
        ai_qs = expand_queries_ai(client, name, genre, seeds, want=max(4, cfg.get("candidates_per_item", 16)//3)) if client else []

        # マージ & 重複排除
        merged, seen = [], set()
        for q in rule_qs + ai_qs:
            if q not in seen:
                seen.add(q)
                merged.append(q)

        # cands row（固定列 1..20）
        record = {"商品名": name, "ジャンルID": genre}
        for i in range(1, 21):
            record[f"Q{i}"] = merged[i-1] if i-1 < len(merged) else ""
        cands_rows.append(record)

        # バッチ（API呼び出し用）
        batch_items.append({
            "item_name": name,
            "genre_id": genre,
            "queries": merged
        })

    # -------- 保存（中間生産物） --------
    pd.DataFrame(seeds_rows).to_csv(seeds_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(cands_rows).to_csv(cands_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    with open(batch_path, "w", encoding="utf-8") as f:
        for item in batch_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"💾 種語を出力: {seeds_path}")
    logger.info(f"💾 候補クエリを出力: {cands_path}")
    logger.info(f"💾 バッチ（JSONL）を出力: {batch_path}")
    logger.info(f"✅ 完了: {len(cands_rows)} 件")
    logger.info("🧭 次は market_enricher.py で外部APIへ注入（楽天/Yahoo）します。")

if __name__ == "__main__":
    logger.info("🌸 KOTOHA ENGINE — Query Generator 起動")
    main()
import atlas_autosave_core
