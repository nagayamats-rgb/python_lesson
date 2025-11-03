import json, os, re
import pandas as pd
from datetime import datetime

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_field(text, keywords):
    """簡易的なタグ抽出: 各構成要素(spec, competence等)を検出"""
    if not text: return ""
    text = re.sub(r"[「」『』]", "", text)
    for kw in keywords:
        if kw in text:
            return text
    return ""

def classify_sentence(sentence):
    """文を構造的カテゴリに分類"""
    sentence = sentence.strip()
    spec_kw = ["仕様", "素材", "機能", "性能", "タイプ", "モデル", "サイズ"]
    competence_kw = ["特長", "特徴", "強み", "魅力", "こだわり", "設計", "工夫"]
    user_kw = ["あなた", "方", "人", "子供", "女性", "男性", "学生", "ビジネス"]
    scene_kw = ["自宅", "外出", "旅行", "オフィス", "通勤", "通学", "運動"]
    benefit_kw = ["便利", "解決", "快適", "使いやすい", "役立つ", "安心", "満足"]

    if extract_field(sentence, spec_kw): return "spec"
    if extract_field(sentence, competence_kw): return "competence"
    if extract_field(sentence, user_kw): return "user"
    if extract_field(sentence, scene_kw): return "scene"
    if extract_field(sentence, benefit_kw): return "benefit"
    return "misc"

def fill_missing(tags):
    """欠落要素を補完"""
    defaults = {
        "spec": "高品質な設計",
        "competence": "細部まで丁寧に作られた構造",
        "user": "幅広い年代の方",
        "scene": "毎日の生活シーン",
        "benefit": "快適で便利に使える"
    }
    for k, v in defaults.items():
        if not tags.get(k):
            tags[k] = v
    return tags

def reorder_and_refine(tags):
    """指定順序で自然な文構成に再構成"""
    order = ["spec", "competence", "user", "scene", "benefit"]
    structured = "。".join([tags[k] for k in order if k in tags])
    return re.sub(r"。。+", "。", structured.strip("。") + "。")

def adjust_copy_length(copy_text, platform):
    """媒体別文字数調整"""
    copy_text = copy_text.strip()
    if platform == "rakuten":
        max_len, ideal = 87, (60, 80)
    elif platform == "yahoo":
        max_len, ideal = 30, (25, 30)
    else:
        return copy_text

    # 文字数調整
    if len(copy_text) > max_len:
        copy_text = copy_text[:max_len]
    elif len(copy_text) < ideal[0]:
        copy_text += "。使いやすさも魅力です"

    return copy_text

def refine_text(copy_raw, alts_raw):
    """構文分類→補完→整形→媒体別コピー作成"""
    sentences = re.split(r"[。！？]", copy_raw)
    tags = {cat: "" for cat in ["spec", "competence", "user", "scene", "benefit"]}
    for s in sentences:
        cat = classify_sentence(s)
        if cat in tags and not tags[cat]:
            tags[cat] = s

    tags = fill_missing(tags)
    refined_core = reorder_and_refine(tags)

    rakuten_copy = adjust_copy_length(refined_core, "rakuten")
    yahoo_copy = adjust_copy_length(refined_core, "yahoo")

    refined_alts = []
    for a in alts_raw[:20]:
        alt_text = reorder_and_refine(fill_missing(tags))
        if len(alt_text) > 110:
            alt_text = alt_text[:110]
        refined_alts.append(alt_text)

    return rakuten_copy, yahoo_copy, refined_alts

def main():
    input_dir = "./output/ai_writer"
    output_dir = "./output/refined"
    os.makedirs(output_dir, exist_ok=True)

    latest_file = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])[-1]
    data = load_json(os.path.join(input_dir, latest_file))

    results = []
    for item in data:
        name = item.get("商品名") or item.get("product_name")
        copy_raw = item.get("copy") or item.get("キャッチコピー") or ""
        alts_raw = item.get("alts") or [""]
        rakuten_copy, yahoo_copy, refined_alts = refine_text(copy_raw, alts_raw)

        results.append({
            "商品名": name,
            "楽天用コピー": rakuten_copy,
            "Yahoo用コピー": yahoo_copy,
            "ALT数": len(refined_alts),
            "ALT例": refined_alts[0] if refined_alts else ""
        })

    now = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join(output_dir, f"refined_copy_alt_{now}.json")
    csv_path = os.path.join(output_dir, f"refined_copy_alt_{now}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 出力完了: {json_path}")
    print(f"✅ CSV出力: {csv_path}")

if __name__ == "__main__":
    print("🌸 KOTOHA Local Refiner v1.0 起動")
    main()
import atlas_autosave_core
