# -*- coding: utf-8 -*-
"""
Atlas AutoSave Core v2.0
-------------------------------------
全スクリプト共通の自動スナップショット＋Git AutoCommit中枢。
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

CONFIG_DIR = os.path.join(os.getcwd(), "config")
SESSION_PATH = os.path.join(CONFIG_DIR, "atlas_session_cache.json")
TIMELINE_PATH = os.path.join(CONFIG_DIR, "atlas_timeline.json")

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_snapshot():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script = os.path.basename(sys.argv[0])
    summary = {
        "timestamp": now,
        "script": script,
        "status": "completed",
        "notes": f"Executed successfully on {now}",
    }
    sess = safe_load_json(SESSION_PATH)
    sess.update({
        "last_run": now,
        "last_script": script,
        "status": "OK",
    })
    safe_write_json(SESSION_PATH, sess)
    timeline = safe_load_json(TIMELINE_PATH)
    if "timeline" not in timeline:
        timeline["timeline"] = []
    timeline["timeline"].append(summary)
    safe_write_json(TIMELINE_PATH, timeline)
    print(f"🧭 Atlas Snapshot Saved: {script} ({now})")

def auto_commit():
    repo_path = os.getenv("GIT_REPO_PATH", os.getcwd()).strip()
    try:
        subprocess.run(["python", "atlas_timeline_autocommit.py"], cwd=repo_path)
        print("✅ AutoCommit executed.")
    except Exception as e:
        print(f"⚠️ AutoCommit skipped: {e}")

if __name__ != "__main__":
    save_snapshot()
    auto_commit()