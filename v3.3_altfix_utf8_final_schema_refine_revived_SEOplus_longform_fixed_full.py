import csv, os, re, json, time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# === 設定 ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

INPUT_PATH = "./rakuten.csv"  # 対象ファイル
OUTPUT_PATH = "./output/ai_writer/alt_text_refined_final_longform.csv"

FORBIDDEN = [
    "画像", "写真", "見た目", "映っている", "図", "絵",
    "この商品", "こちらの商品", "上記", "下記", "イメージ", "画面", "図解", "イラスト"
]

# === 補助関数 ===
def clean_text(text):
    """禁則語削除＋句点補正＋文字数調整"""
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    for w in FORBIDDEN:
        text = text.replace(w, "")
    # 文末補正
    if text.endswith("。"):
        text = text[:-1]
    text = text.strip("。、 ")
    # 長すぎる場合は文単位でカット
    sents = re.split("(?<=。)", text)
    result = ""
    for s in sents:
        if len(result + s) > 110:
            break
        result += s
    # 足りない場合は自然に補完
    if len(result) < 80 and len(text) > 80:
        result = text[:100]
        result = result[:result.rfind("。")+1] if "。" in result else result
    return result.strip("。、 ")

def ai_generate_alts(product_name):
    """商品名をもとにALT 20件生成（5件×4回に分割）"""
    all_alts = []
    for block in range(4):
        prompt = f"""
あなたはSEO最適化ライターです。
以下の商品に関して、画像説明文として自然な日本語を5件生成してください。
・1件あたり120〜130文字前後で自然に。
・絵文字、特殊記号、HTMLは禁止。
・句読点は適切に配置。
・画像や写真の描写語は禁止。
・構成ヒント：「商品スペック→機能→どんな人→どんなシーン→利点」の自然文。
・各文は「。」で終えること。
商品名: {product_name}
"""
        try:
            res = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "あなたは日本語SEO最適化のプロフェッショナルコピーライターです。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
    max_completion_tokens=1000,
    response_format={"type": "text"}  # ✅ 新仕様対応
)
            
            text = res.choices[0].message.content
            # 文分割して5文抽出
            alts = re.findall(r"[^。]+。", text)
            alts = [clean_text(a) for a in alts[:5] if len(a) > 20]
            all_alts.extend(alts)
        except Exception as e:
            print(f"⚠️ 生成エラー({block+1}/4): {e}")
            time.sleep(2)
    return all_alts[:20]

# === メイン ===
def main():
    print("🌸 ALT生成開始（SEO強化＋長文モード）")

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = [r["商品名"] for r in reader if r.get("商品名")]
    print(f"✅ 対象商品数: {len(products)}件")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        header = ["商品名"] + [f"ALT{i+1}" for i in range(20)]
        writer.writerow(header)

        for nm in tqdm(products, desc="🧠 生成中"):
            alts = ai_generate_alts(nm)
            row = [nm] + alts + [""] * (20 - len(alts))
            writer.writerow(row)
            time.sleep(1.5)

    print(f"✅ 出力完了: {OUTPUT_PATH}")
    print("✅ ALTは120〜130字生成→自然な文末で80〜110字に整形。禁則・句点補正済。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
