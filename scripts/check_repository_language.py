#!/usr/bin/env python3
"""Reject Chinese/Han text in tracked repository content, including notebooks."""

import json
import os
from pathlib import Path
import subprocess
import unicodedata


def contains_han(text):
    for character in text:
        name = unicodedata.name(character, "")
        if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            return True
        if name == "IDEOGRAPHIC NUMBER ZERO":
            return True
    return False


def check_repository(root):
    names = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    errors = []
    for name in filter(None, names):
        path = root / name
        if contains_han(name):
            errors.append(f"{name}: keep filenames in English")
        if not path.exists():
            continue  # A tracked file may have been deleted in a local checkout.
        try:
            text = os.readlink(path) if path.is_symlink() else path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # Binary assets are reviewed visually.
        if path.suffix in {".json", ".ipynb"}:
            # Notebook writers may encode non-ASCII characters as JSON escapes.
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False)
            except json.JSONDecodeError:
                errors.append(f"{name}: invalid JSON; cannot check decoded text")
                continue
        if contains_han(text):
            errors.append(f"{name}: Chinese/Han text found; keep content in English")
    return errors


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    violations = check_repository(root)
    if violations:
        print("\n".join(violations))
        raise SystemExit(1)
    print("English repository content check passed.")
