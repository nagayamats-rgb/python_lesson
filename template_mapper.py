import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# ==============================================
# 🌸 KOTOHA ENGINE — Template Mapper v1.2 Smart Accessory Optimized
# ==============================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

load_dotenv()
OUTPUT_DIR = "./output/semantics"
INPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_latest_file(directory=INPUT_DIR, prefix="lexical_clusters_", ext=".json"):
    """最新の lexical_clusters ファイルを検出"""
    try:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
        if not files:
            logging.error("❌ lexical_clusters_*.json が見つかりません。")
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        latest = os.path.join(directory, files[0])
        logging.info(f"📄 最新ファイルを使用: {latest}")
        return latest
    except Exception as e:
        logging.error(f"⚠️ ファイル探索エラー: {e}")
        return None


def load_templates():
    """業種特化テンプレート（スマホアクセ）を読み込み"""
    templates = {
        "specs": [
            "最新の {keyword} を採用し、{feature} を実現。",
            "{material} 素材で高耐久・軽量設計。"
        ],
        "usability": [
            "持ち運びやすく、{use_scene} に最適。",
            "{device} にフィットし、日常使いも快適。"
        ],
        "differentiation": [
            "他社製品にはない {unique_point} が魅力。",
            "口コミ評価の高い {highlight} を搭載。"
        ],
        "emotion": [
            "毎日の使用がもっと楽しくなる {emotion_word} 体験。",
            "見た目もスマートに、あなたらしさを引き立てます。"
        ]
    }
    logging.info("📚 テンプレートをロードしました（Smart Accessory Optimized）")
    return templates


def map_clusters_to_templates(clusters, templates):
    """クラスタ語彙にテンプレートを割り当て"""
    mapped_data = []
    for c in tqdm(clusters, desc="🧩 クラスタマッピング進行中", unit="cluster"):
        cid = c.get("cluster_id")
        terms = c.get("terms", [])
        keywords = terms[:5]  # 上位5件を代表語彙として使用

        mapped_entry = {
            "cluster_id": cid,
            "keywords": keywords,
            "templates": {
                "specs": templates["specs"],
                "usability": templates["usability"],
                "differentiation": templates["differentiation"],
                "emotion": templates["emotion"]
            }
        }
        mapped_data.append(mapped_entry)
    return mapped_data


def main():
    start_time = time.time()
    logging.info("🌸 KOTOHA ENGINE — Template Mapper 起動")

    input_file = find_latest_file()
    if not input_file:
        return

    with open(input_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    templates = load_templates()
    mapped = map_clusters_to_templates(clusters, templates)

    output_path = os.path.join(
        OUTPUT_DIR, f"structured_semantics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapped, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n✅ 完了! Template Mapper 実行結果:")
    print(f"🕒 実行時間: {elapsed:.1f}秒")
    print(f"💾 出力ファイル: {output_path}")
    print(f"🧩 クラスタ構文数: {len(mapped)}")


if __name__ == "__main__":
    main()
import atlas_autosave_core
