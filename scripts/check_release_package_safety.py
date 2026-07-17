from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "release_build" / "AndoFrontier-Lab-v0.3.1-Windows"
LOCAL_USERNAME = os.environ.get("USERNAME", "").strip()

SKIP_SUFFIXES = {
    ".exe",
    ".dll",
    ".pak",
    ".bin",
    ".dat",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".zip",
}

PATTERNS = [
    ("CRITICAL", "windows_user_path", re.compile(r"C:\\Users\\", re.IGNORECASE)),
    ("CRITICAL", "workspace_absolute_path", re.compile(r"C:\\Workspaces", re.IGNORECASE)),
    ("CRITICAL", "private_input_folder", re.compile(r"uap desclafisicados", re.IGNORECASE)),
    ("CRITICAL", "env_file_reference", re.compile(r"(^|[\\/])\.env($|[\s`'\"])", re.IGNORECASE)),
    ("CRITICAL", "openai_api_key", re.compile(r"OPENAI_API_KEY", re.IGNORECASE)),
    ("CRITICAL", "anthropic_api_key", re.compile(r"ANTHROPIC_API_KEY", re.IGNORECASE)),
    ("CRITICAL", "google_api_key", re.compile(r"GOOGLE_API_KEY", re.IGNORECASE)),
    ("CRITICAL", "private_data_dir", re.compile(r"data[\\/](cases|outputs|reports|content_packs|publisher_packs)", re.IGNORECASE)),
    ("CRITICAL", "private_content_pack", re.compile(r"content pack|content_pack", re.IGNORECASE)),
    ("CRITICAL", "private_publisher_pack", re.compile(r"publisher pack|publisher_pack", re.IGNORECASE)),
    ("CRITICAL", "original_video", re.compile(r"\.(mp4|mov|avi|mkv|webm|m4v)\b", re.IGNORECASE)),
]

if len(LOCAL_USERNAME) >= 3:
    PATTERNS.append(("CRITICAL", "personal_username", re.compile(re.escape(LOCAL_USERNAME), re.IGNORECASE)))

ALLOWLIST = {
    Path("README_RUN_WINDOWS.md"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a built AndoFrontier Lab package for private or unsafe content.")
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    if not package_dir.exists():
        print(f"Release package folder not found: {package_dir}")
        return 1
    findings = []
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(package_dir)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for severity, name, regex in PATTERNS:
                if regex.search(line):
                    if rel == Path("resources/app.asar") and name == "original_video":
                        continue
                    if rel in ALLOWLIST and name in {"private_content_pack", "private_publisher_pack"}:
                        continue
                    findings.append((severity, rel, line_no, name, line.strip()[:180]))
    for severity, rel, line_no, name, preview in findings:
        print(f"{severity}\t{rel}:{line_no}\t{name}\t{preview}")
    if any(item[0] == "CRITICAL" for item in findings):
        print(f"\nRelease package safety failed: {len(findings)} finding(s).")
        return 1
    print("Release package safety passed: no critical findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
