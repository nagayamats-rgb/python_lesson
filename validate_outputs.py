# save as: validate_outputs.py
import json, re, sys, csv
from collections import Counter

# === 設定 ===
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else "./output/ai_writer/hybrid_writer_full_latest.json"
FORBIDDEN_PATHS = [
    "./output/semantics/normalized_20251031_0039.json",  # あれば読み込み
]
ALT_IMAGE_WORDS = r"(画像|写真|図|写っている|スクリーンショット|スクショ|イメージ図)"
BAN_PAT = re.compile(ALT_IMAGE_WORDS)

def jp_len(s): return len(s or "")

def load_forbidden():
    words = set()
    for p in FORBIDDEN_PATHS:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 構造揺れに対応
            if isinstance(data, dict):
                cand = data.get("forbidden_words") or data.get("forbidden") or []
            elif isinstance(data, list):
                cand = data
            else:
                cand = []
            for w in cand:
                if isinstance(w, str): words.add(w)
                elif isinstance(w, dict):
                    for k in ("word","ng","term","value","pattern"):
                        if k in w and isinstance(w[k], str): words.add(w[k])
        except FileNotFoundError:
            pass
    return words

def similar(a,b):
    # ザックリ重複判定（Jaccard）
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def validate(record):
    errors, warns = [], []

    pname = record.get("product_name") or record.get("name") or ""
    copy  = record.get("copy") or ""
    alts  = record.get("alts") if isinstance(record.get("alts"), list) else []

    # スキーマ
    if not pname: errors.append("product_name欠落")
    # copy
    if not copy: errors.append("copy空")
    if copy and not (40 <= jp_len(copy) <= 60):
        errors.append(f"copy長さ不正({jp_len(copy)})")
    # ただの転記・型番断片の疑い
    if copy and pname and (copy.strip() in pname or pname.strip() in copy):
        warns.append("copyが商品名転記っぽい")
    # 禁則
    forbidden = validate.forbidden
    for ng in forbidden:
        if ng and ng in copy:
            errors.append(f"copy禁則語: {ng}")

    # alts
    if len(alts) != 20: errors.append(f"ALT件数不正({len(alts)})")
    alt_dup_pairs = 0
    seen = []
    for i,a in enumerate(alts):
        if not isinstance(a, str) or not a.strip():
            errors.append(f"ALT[{i}]空/非文字列")
            continue
        n = jp_len(a)
        if not (80 <= n <= 110):
            errors.append(f"ALT[{i}]長さ不正({n})")
        # 画像描写禁止
        if BAN_PAT.search(a):
            errors.append(f"ALT[{i}]画像描写ワード検出")
        # 禁則
        for ng in forbidden:
            if ng and ng in a:
                errors.append(f"ALT[{i}]禁則語: {ng}")
        # 重複（緩め）
        for j,prev in enumerate(seen):
            if similar(a, prev) >= 0.9:
                alt_dup_pairs += 1
                break
        seen.append(a)

    if alt_dup_pairs > 0:
        warns.append(f"ALT高類似（重複疑い）{alt_dup_pairs}件")

    return errors, warns

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    validate.forbidden = load_forbidden()

    total = len(items)
    bad = 0
    with open("./output/ai_writer/validation_report.csv", "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(["#","product_name","errors","warnings","copy_len","alts_len"])
        for idx, rec in enumerate(items, 1):
            errs, warns = validate(rec)
            if errs: bad += 1
            w.writerow([
                idx,
                (rec.get("product_name") or rec.get("name") or "")[:40],
                " / ".join(errs),
                " / ".join(warns),
                jp_len(rec.get("copy") or ""),
                len(rec.get("alts") or []),
            ])

    print(f"✅ 総件数: {total} / ❌ 不合格: {bad}")
    print("レポート: ./output/ai_writer/validation_report.csv を確認してください。")

if __name__ == "__main__":
    main()
import atlas_autosave_core
