# -*- coding: utf-8 -*-
"""
Atlas AutoSave Global Injector v1.0
-------------------------------------
全スクリプトに自動で Atlas AutoSave Core を注入。
"""

import os

TARGET_DIR = "/Users/tsuyoshi/Desktop/python_lesson"
INJECT_LINE = "import atlas_autosave_core"

def inject_autosave(target_dir=TARGET_DIR):
    count = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("atlas_autosave_"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r+", encoding="utf-8") as f:
                        content = f.read()
                        if INJECT_LINE not in content:
                            if not content.endswith("\n"):
                                content += "\n"
                            content += f"{INJECT_LINE}\n"
                            f.seek(0)
                            f.write(content)
                            f.truncate()
                            count += 1
                            print(f"✅ Injected into {file}")
                        else:
                            print(f"⏩ Skipped (already injected): {file}")
                except Exception as e:
                    print(f"⚠️ Error injecting into {file}: {e}")
    print(f"\n🎯 Injection completed: {count} files updated.")

if __name__ == "__main__":
    inject_autosave()