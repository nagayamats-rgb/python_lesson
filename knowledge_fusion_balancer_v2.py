# -*- coding: utf-8 -*-
"""
KOTOHA 知見バランサー v2（自然文構造版）
- 目的: /output/semantics 配下の JSON 群を“自然文の知見”に再構成し、
        ALT/コピー生成プロンプトに直接投入できる形へ正規化する。
- 入力: ./output/semantics/
    lexical_clusters_*.json       → 用語・形容語・特徴
    market_vocab_*.json           → 市場語彙（機能/用途/対象など）
    structured_semantics_*.json   → scenes/targets/features/benefits などの構造語彙
    styled_persona_*.json         → tone/style 等
    template_composer.json        → 構成ヒント
    normalized_*.json             → 禁則語（forbidden_words）
- 出力:
    ./output/semantics/knowledge_fused_structured_v2.json

設計メモ:
- 入ってくる JSON の形は辞書だったり配列だったりバラバラなので、すべて安全に吸収。
- タグ列をそのまま繋がず、短い自然文へ変換（〜に適した、〜を備えた、〜を想定した など）。
- 文は 8〜14 文を目安に生成（長文 ALT/コピーの種としてちょうど良い量）。
- 句点「。」で必ず終止。重複を除去。禁則語は最後に再検閲して除去/置換。
"""

import os
import re
import json
import glob
import random
from typing import Any, Dict, List, Iterable

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SEMANTICS_DIR = os.path.join(BASE_DIR, "output", "semantics")
OUT_PATH = os.path.join(SEMANTICS_DIR, "knowledge_fused_structured_v2.json")

# ——————————————————————————————
# 基本ユーティリティ
# ——————————————————————————————
def safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def uniq(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if not isinstance(x, str):
            continue
        x2 = x.strip()
        if not x2 or x2 in seen:
            continue
        out.append(x2)
        seen.add(x2)
    return out

def normalize_term(t: str) -> str:
    t = (t or "").strip()
    # 露骨なラベル/記号を掃除
    t = re.sub(r"^(?:[-*・●\d①-⑩]\s*[\.．、]?\s*)", "", t)
    # 連続空白
    t = re.sub(r"\s+", " ", t)
    return t

def end_with_maru(s: str) -> str:
    s = s.strip()
    if not s.endswith("。"):
        s += "。"
    return s

# ——————————————————————————————
# 既存 JSON 群を吸収
# ——————————————————————————————
def load_semantic_inputs() -> Dict[str, Any]:
    files = glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))
    payload = {
        "lexical": [],
        "market": [],
        "structured": {},
        "persona": {},
        "templates": [],
        "forbidden": [],
    }

    for p in files:
        name = os.path.basename(p).lower()
        data = safe_load_json(p)
        if data is None:
            continue

        try:
            # lexical_clusters_* 例: {"clusters":[{"terms":[...]}, ...]} / [{"terms":[...]}] / ["語1","語2"]
            if "lexical" in name and "cluster" in name:
                if isinstance(data, dict):
                    arr = data.get("clusters") or data.get("lexical") or []
                elif isinstance(data, list):
                    arr = data
                else:
                    arr = []
                for c in arr:
                    if isinstance(c, dict) and "terms" in c and isinstance(c["terms"], list):
                        payload["lexical"].extend([normalize_term(t) for t in c["terms"] if isinstance(t, str)])
                    elif isinstance(c, str):
                        payload["lexical"].append(normalize_term(c))

            # market_vocab_* 例: [{"vocabulary":"MagSafe"}, ...] / ["MagSafe", "PD"]
            elif "market" in name and "vocab" in name:
                if isinstance(data, list):
                    for v in data:
                        if isinstance(v, dict) and "vocabulary" in v:
                            payload["market"].append(normalize_term(v.get("vocabulary", "")))
                        elif isinstance(v, str):
                            payload["market"].append(normalize_term(v))
                elif isinstance(data, dict):
                    vocab = data.get("vocabulary") or data.get("vocab") or []
                    if isinstance(vocab, list):
                        payload["market"].extend([normalize_term(x) for x in vocab if isinstance(x, str)])

            # structured_semantics_* 例: {"scenes":[...], "targets":[...], "features":[...], "benefits":[...]}
            elif "structured_semantics" in name or ("structured" in name and "semantic" in name):
                if isinstance(data, dict):
                    for k in ["scenes", "targets", "features", "benefits", "concepts", "use_cases"]:
                        arr = data.get(k) or []
                        if isinstance(arr, list):
                            payload["structured"].setdefault(k, [])
                            payload["structured"][k].extend([normalize_term(x) for x in arr if isinstance(x, str)])
                elif isinstance(data, list):
                    # 想定外だが、文字列リストなら features 扱いで吸収
                    payload["structured"].setdefault("features", [])
                    payload["structured"]["features"].extend([normalize_term(x) for x in data if isinstance(x, str)])

            # styled_persona_* 例: {"tone":{"style":"〜","register":"〜"}} / {"tone":["上品","知的"]} / ["〜"]
            elif "styled_persona" in name or "persona" in name:
                tone = {}
                if isinstance(data, dict):
                    t = data.get("tone")
                    if isinstance(t, dict):
                        for k, v in t.items():
                            if isinstance(v, str):
                                tone[k] = normalize_term(v)
                    elif isinstance(t, list):
                        tone["hints"] = uniq([normalize_term(x) for x in t if isinstance(x, str)])
                elif isinstance(data, list):
                    tone["hints"] = uniq([normalize_term(x) for x in data if isinstance(x, str)])
                if tone:
                    payload["persona"] = tone

            # template_composer.json 例: {"hints":[...]} / {"templates":[...]} / ["〜"]
            elif "template_composer" in name or "template" in name:
                if isinstance(data, dict):
                    arr = data.get("hints") or data.get("templates") or []
                    if isinstance(arr, list):
                        payload["templates"].extend([normalize_term(x) for x in arr if isinstance(x, str)])
                elif isinstance(data, list):
                    payload["templates"].extend([normalize_term(x) for x in data if isinstance(x, str)])

            # normalized_* 例: {"forbidden_words":[...]} / ["画像","写真",...]
            elif "normalized" in name or "forbid" in name:
                if isinstance(data, dict):
                    fw = data.get("forbidden_words") or data.get("forbidden") or []
                    if isinstance(fw, list):
                        payload["forbidden"].extend([normalize_term(x) for x in fw if isinstance(x, str)])
                elif isinstance(data, list):
                    payload["forbidden"].extend([normalize_term(x) for x in data if isinstance(x, str)])

        except Exception:
            # 壊れた形式は黙ってスキップ（堅牢優先）
            pass

    # ユニーク化
    payload["lexical"]   = uniq(payload["lexical"])
    payload["market"]    = uniq(payload["market"])
    for k, v in list(payload["structured"].items()):
        payload["structured"][k] = uniq(v)
    payload["templates"] = uniq(payload["templates"])
    payload["forbidden"] = uniq(payload["forbidden"])
    return payload

# ——————————————————————————————
# 自然文への写像（短文ジェネレータ）
# ——————————————————————————————
CONNECTORS = [
    "、", "で", "ながら", "だから", "だからこそ", "だからと言って", "そして", "さらに"
]

PATTERNS_FEATURE = [
    "{feat}を備え、{benefit}を実現します",
    "{feat}の設計で、{scene}でも快適に使えます",
    "日常の{scene}で役立つ{feat}が魅力です",
    "{feat}により、{target}の{benefit}に貢献します",
]

PATTERNS_SCENE = [
    "{scene}に最適で、{feature}が{benefit}を後押しします",
    "{scene}を想定した設計で、{target}の使い勝手を高めます",
]

PATTERNS_TARGET = [
    "{target}に向けて作られ、{feature}で{benefit}をもたらします",
    "{target}の日常に寄り添い、{scene}でも扱いやすい配慮があります",
]

PATTERNS_GENERIC = [
    "{feature}に配慮した設計で、{benefit}を狙えます",
    "使い勝手を重視し、{scene}でも扱いやすく仕上げています",
]

def pick(xs: List[str], n: int) -> List[str]:
    xs = [x for x in xs if isinstance(x, str) and x]
    if not xs:
        return []
    if len(xs) <= n:
        return xs
    return random.sample(xs, n)

def join_terms(terms: List[str], limit: int = 3) -> str:
    terms = uniq([t for t in terms if t])
    if not terms:
        return ""
    # 3語くらいまで素直に読点接続
    return "、".join(terms[:limit])

def build_sentence(feature: str = "", scene: str = "", target: str = "", benefit: str = "") -> str:
    """
    与えられた semantic スロットから自然文を1つ生成。
    スロットが空でも破綻しないテンプレ選択。
    """
    feature = normalize_term(feature)
    scene   = normalize_term(scene)
    target  = normalize_term(target)
    benefit = normalize_term(benefit)

    # スロットに応じてパターンバリエーション
    if feature and scene and benefit:
        tpl = random.choice(PATTERNS_FEATURE + PATTERNS_SCENE)
    elif feature and target and benefit:
        tpl = random.choice(PATTERNS_FEATURE + PATTERNS_TARGET)
    elif scene and target:
        tpl = random.choice(PATTERNS_SCENE + PATTERNS_TARGET)
    else:
        tpl = random.choice(PATTERNS_GENERIC)

    # ←ここを修正： 'feat' ではなく 'feature' に統一
    s = tpl.format(
        feature=feature or "利便性",
        scene=scene or "日常利用",
        target=target or "幅広いユーザー",
        benefit=benefit or "快適性の向上",
    )

    # 接続詞で軽く豊かさを出す（必要なときだけ）
    if random.random() < 0.35 and feature and scene:
        s = f"{feature}{random.choice(CONNECTORS)}{s}"

    return end_with_maru(s)

def to_natural_sentences(payload: Dict[str, Any], aim_min=8, aim_max=14) -> List[str]:
    # 素材の取り出し
    feats   = payload.get("structured", {}).get("features", []) or payload.get("lexical", [])
    scenes  = payload.get("structured", {}).get("scenes",   [])
    targets = payload.get("structured", {}).get("targets",  [])
    bens    = payload.get("structured", {}).get("benefits", [])
    market  = payload.get("market", [])

    # “語の羅列”でなく、“文”としての素材を増やすため、少しだけ混ぜる
    # 使いすぎると不自然になるので、それぞれ上限を絞る
    feats_use   = pick(feats or market,   10)
    scenes_use  = pick(scenes or market,   8)
    targets_use = pick(targets or market,  6)
    bens_use    = pick(bens or feats,      8)

    # 文を組む
    sentences = []
    iter_max = max(aim_max * 2, 30)  # 生成余裕
    i = 0
    while len(sentences) < aim_max and i < iter_max:
        i += 1
        f = feats_use[i % len(feats_use)] if feats_use else ""
        sc = scenes_use[i % len(scenes_use)] if scenes_use else ""
        tg = targets_use[i % len(targets_use)] if targets_use else ""
        bn = bens_use[i % len(bens_use)] if bens_use else ""
        s  = build_sentence(f, sc, tg, bn)
        sentences.append(s)

    # ユニーク化
    sentences = uniq(sentences)

    # 文字数の軽い整形（60〜120字目安）
    def soft_len(s: str) -> str:
        s = s.strip()
        # 末尾調整
        s = end_with_maru(s)
        # 短すぎる場合は、featureやmarketを1語だけ足して延ばす
        if len(s) < 50 and feats:
            s = re.sub(r"。$", f"、{random.choice(feats)}を意識した設計です。", s)
        # 長過ぎる場合は、句点で自然カット
        if len(s) > 130:
            cut = s[:130]
            p = cut.rfind("。")
            s = (cut[:p+1] if p != -1 else cut)
        return s

    sentences = [soft_len(s) for s in sentences]

    # 目標数に寄せる（不足分は軽いパラフレーズ）
    def paraphrase(s: str) -> str:
        s2 = s.replace("実現します。", "叶えます。")
        s2 = s2.replace("使えます。", "しやすいです。")
        s2 = s2.replace("仕上げています。", "仕上げです。")
        if s2 == s:
            s2 = s[:-1] + "のが特長です。"
        return end_with_maru(s2)

    j = 0
    while len(sentences) < aim_min and sentences:
        sentences.append(paraphrase(sentences[j % len(sentences)]))
        j += 1

    # 仕上げ：重複除去 & 目標上限にクリップ
    return sentences[:aim_max]

# ——————————————————————————————
# 禁則適用
# ——————————————————————————————
DEFAULT_FORBIDDEN = [
    "画像", "写真", "見た目", "上の画像", "下の写真",
    "当店", "当社", "レビュー", "ランキング",
    "クリック", "こちら", "リンク", "購入はこちら",
    "競合", "優位性", "業界最高", "最安", "No.1", "ナンバーワン", "売上No1",
]

def apply_forbidden(sentences: List[str], words: List[str]) -> List[str]:
    ngs = uniq((words or []) + DEFAULT_FORBIDDEN)
    out = []
    for s in sentences:
        t = s
        for ng in ngs:
            if ng and ng in t:
                t = t.replace(ng, "")
        t = re.sub(r"\s+", " ", t).strip()
        t = end_with_maru(t)
        out.append(t)
    return out

# ——————————————————————————————
# メイン
# ——————————————————————————————
def main():
    random.seed(42)  # 再現性確保（必要に応じて外す）
    ensure_dir(SEMANTICS_DIR)

    payload = load_semantic_inputs()
    sentences = to_natural_sentences(payload, aim_min=8, aim_max=14)
    sentences = apply_forbidden(sentences, payload.get("forbidden", []))

    result = {
        "knowledge_text": sentences,
        "forbidden_words": uniq((payload.get("forbidden") or []) + DEFAULT_FORBIDDEN),
        "meta": {
            "source_files_count": len(glob.glob(os.path.join(SEMANTICS_DIR, "*.json"))),
            "lexical_count": len(payload.get("lexical", [])),
            "market_count": len(payload.get("market", [])),
            "structured_counts": {k: len(v) for k, v in payload.get("structured", {}).items()},
            "templates_count": len(payload.get("templates", [])),
            "persona_keys": list(payload.get("persona", {}).keys()),
        }
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 出力完了: {OUT_PATH}")
    print(f"📘 知見文数: {len(sentences)} / 禁則語数: {len(result['forbidden_words'])}")

if __name__ == "__main__":
    main()
import atlas_autosave_core
