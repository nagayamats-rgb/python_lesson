import csv, json, os, re, random
from statistics import mean

# === パス設定 ===
BASE_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INPUT_CSV  = f"{BASE_DIR}/alt_text_20251101_test.csv"
SEM_DIR    = f"{BASE_DIR}/output/semantics"
OUTPUT_DIR = f"{BASE_DIR}/output/refined"
OUTPUT_CSV = f"{OUTPUT_DIR}/alt_text_refined_test_v3.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === JSON知見 ===
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

knowledge = {}
for fn in os.listdir(SEM_DIR):
    if fn.endswith(".json"):
        knowledge[fn.split(".")[0]] = load_json(os.path.join(SEM_DIR, fn))

# === ローカル語彙抽出 ===
market_vocab = []
for v in knowledge.get("market_vocab_20251030_201906", []):
    if isinstance(v, dict) and "vocabulary" in v:
        market_vocab.append(v["vocabulary"])
core_vocab = set(market_vocab[:80])  # 上位語を優先保持

# === 正規化関数 ===
def normalize_text(s):
    if not s: return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("、。", "。").replace("。。", "。")
    s = re.sub(r"([。、])\1+", r"\1", s)
    return s

def fix_sentence_end(s):
    """語尾修正"""
    rules = {
        "しです": "します", "ますです": "ます", "いです": "い",
        "するです": "します", "れるです": "れます",
        "できです": "できます", "できるです": "できます"
    }
    for k,v in rules.items():
        s = s.replace(k,v)
    return s

def compress_particles(s):
    """助詞の重複削除"""
    s = re.sub(r"(で){2,}", "で", s)
    s = re.sub(r"(に){2,}", "に", s)
    s = re.sub(r"(を){2,}", "を", s)
    s = re.sub(r"(して){2,}", "して", s)
    return s

def simplify_verbs(s):
    """動詞冗長構文削除（軽量）"""
    s = re.sub(r"できる使いやすい設計", "使いやすい設計", s)
    s = re.sub(r"できる簡単操作", "簡単操作", s)
    return s

def noun_stop_transform(s):
    """15%確率で名詞終止化"""
    if random.random() < 0.15:
        s = re.sub(r"(ます|です)。$", "。", s)
        if not re.search(r"[。]$", s):
            s += "。"
    return s

def seo_filter(s):
    """低頻度語削除"""
    words = re.split(r"(?<=。)|(?<=、)", s)
    filtered = []
    for w in words:
        if not any(k in w for k in core_vocab):
            # 頻度低い文節でも削除しすぎない（安全率）
            if len(w) > 8:
                filtered.append(w)
        else:
            filtered.append(w)
    return "".join(filtered)

def cleanse_text(text):
    t = normalize_text(text)
    t = fix_sentence_end(t)
    t = compress_particles(t)
    t = simplify_verbs(t)
    t = seo_filter(t)
    t = noun_stop_transform(t)
    if not t.endswith("。"):
        t += "。"
    return t

# === メイン処理 ===
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        new_row = row.copy()
        for i in range(1, 21):
            col = f"ALT{i}"
            if col in new_row and new_row[col]:
                new_row[col] = cleanse_text(new_row[col])
        rows.append(new_row)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# === 評価ログ ===
lengths = [len(r[c]) for r in rows for c in r if c.startswith("ALT") and r[c]]
bad = [r[c] for r in rows for c in r if c.startswith("ALT") and "しです" in r[c]]
avg_len = round(mean(lengths), 1) if lengths else 0

print("🌸 ALTローカルリファイン v3 完了")
print(f"✅ 出力: {OUTPUT_CSV}")
print(f"📦 商品数: {len(rows)}")
print(f"📏 平均文字数: {avg_len}（{min(lengths)}〜{max(lengths)}）")
print(f"💬 語尾崩れ修正検出: {len(bad)}件")
print(f"🏷️ 参照語彙: {len(core_vocab)}件（market_vocabベース）")
import atlas_autosave_core
