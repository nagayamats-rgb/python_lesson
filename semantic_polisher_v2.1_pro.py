import os
import re
import json
import unicodedata
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ===============================
# 🌸 KOTOHA ENGINE — Semantic Polisher v2.1 Pro
# ===============================

def count_zenkaku(s: str) -> int:
    """全角文字数カウント"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FWA" else 1 for ch in s) // 2

def normalize_text(text: str) -> str:
    """句読点統一・禁則処理"""
    text = re.sub(r"[！!]", "", text)
    text = re.sub(r"[。]+", "。", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("。"):
        text += "。"
    return text

def generate_alt_variants(keywords):
    """ALTテンプレート生成（80〜110字）"""
    base_phrases = [
        "{core} {feature} {scene} {value}",
        "{scene} {core} {feature} {value}",
        "{core} の {feature} {scene} {value}",
        "{core} {scene} に最適な {feature} {value}",
    ]

    # サブ語彙候補
    value_terms = ["便利", "快適", "高品質", "人気", "多機能", "高評価", "耐久性抜群", "デザイン性が高い", "持ち運びやすい", "ギフトにも最適"]
    scene_terms = ["自宅", "オフィス", "外出先", "旅行", "ビジネス", "日常", "通勤", "勉強中", "就寝前", "家族時間"]
    feature_terms = ["軽量設計", "急速充電", "防水仕様", "スリムボディ", "衝撃吸収", "滑り止め加工", "放熱設計", "柔らか素材", "安定感抜群", "長持ちバッテリー"]

    alts = []
    for i in range(20):
        core = keywords[0] if keywords else "商品"
        feature = feature_terms[i % len(feature_terms)]
        scene = scene_terms[i % len(scene_terms)]
        value = value_terms[i % len(value_terms)]
        template = base_phrases[i % len(base_phrases)]
        alt = template.format(core=core, feature=feature, scene=scene, value=value)
        alt = normalize_text(alt)
        # 文字数補正
        while count_zenkaku(alt) < 80:
            alt += " " + value
        if count_zenkaku(alt) > 110:
            alt = alt[:110]
            if not alt.endswith("。"):
                alt += "。"
        alts.append(alt)
    return alts

def adjust_copy_length(copy_text: str) -> str:
    """キャッチコピーの長さ調整"""
    copy_text = normalize_text(copy_text)
    l = count_zenkaku(copy_text)
    if l < 40:
        copy_text = copy_text + " " + "快適に使える高品質モデル。"
    elif l > 60:
        copy_text = copy_text[:60]
        if not copy_text.endswith("。"):
            copy_text += "。"
    return normalize_text(copy_text)

def main():
    print("🌸 KOTOHA ENGINE — Semantic Polisher v2.1 Pro 起動")

        # 入力ファイル自動検出
    input_root = "./output"
    candidates = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.startswith("polished_") and f.endswith(".json"):
                candidates.append(os.path.join(root, f))

    if not candidates:
        print("🚫 polished_*.json が見つかりません。")
        return

    input_path = sorted(candidates)[-1]  # 最新ファイルを選択
    print(f"📄 入力: {input_path}")


    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = "./output/polished_v2"
    os.makedirs(out_dir, exist_ok=True)
    output_json = os.path.join(out_dir, f"polished_pro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_csv = os.path.join(out_dir, "polished_final_preview.csv")
    report_path = os.path.join(out_dir, "report_semantic_polisher.txt")

    summary = []
    records = []

    for cluster in tqdm(data, desc="🪞 Polishing"):
        cid = cluster.get("cluster_id")
        copy_text = adjust_copy_length(cluster.get("catch_copy", ""))
        alt_texts = generate_alt_variants(cluster.get("keywords", []))

        records.append({
            "cluster_id": cid,
            "catch_copy": copy_text,
            "alt_texts": alt_texts
        })

        summary.append({
            "cluster": cid,
            "copy_len": count_zenkaku(copy_text),
            "alts": len(alt_texts),
            "alt_avg_len": round(sum(count_zenkaku(a) for a in alt_texts) / len(alt_texts), 1)
        })

    # JSON保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # CSVプレビュー出力
    flat_rows = []
    for r in records:
        row = {"cluster_id": r["cluster_id"], "catch_copy": r["catch_copy"]}
        for i, a in enumerate(r["alt_texts"], start=1):
            row[f"alt_{i}"] = a
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    # レポート出力
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== KOTOHA ENGINE Semantic Polisher Report ===\n")
        f.write(f"総クラスタ数: {len(summary)}\n")
        f.write(f"平均キャッチ文字数: {sum(s['copy_len'] for s in summary)/len(summary):.1f}\n")
        f.write(f"ALT平均文字数: {sum(s['alt_avg_len'] for s in summary)/len(summary):.1f}\n")
        f.write(f"ALT平均数: {sum(s['alts'] for s in summary)/len(summary):.1f}\n\n")
        for s in summary:
            f.write(f"cluster {s['cluster']}: copy {s['copy_len']}字 / ALT {s['alts']}本 (平均{s['alt_avg_len']}字)\n")

    print(f"\n✅ 完了! Polished出力: {output_json}")
    print(f"📊 CSVプレビュー: {output_csv}")
    print(f"🧾 レポート: {report_path}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
