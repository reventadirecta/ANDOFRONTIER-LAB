from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_USERNAME = os.environ.get("USERNAME", "").strip()

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "node_modules",
    "dist",
    "release",
    "release_build",
    "runtime",
    "logs",
    "tmp",
    "cache",
    "private",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".pt",
    ".pth",
    ".onnx",
    ".zip",
    ".7z",
    ".rar",
    ".lock",
}

PATTERNS = [
    ("CRITICAL", "windows_user_path", re.compile(r"C:\\Users\\", re.IGNORECASE)),
    ("CRITICAL", "workspace_absolute_path", re.compile(r"C:\\Workspaces", re.IGNORECASE)),
    ("CRITICAL", "private_input_folder", re.compile(r"uap desclafisicados", re.IGNORECASE)),
    ("CRITICAL", "env_file_reference", re.compile(r"(^|[\\/])\.env($|[\s`'\"])", re.IGNORECASE)),
    ("CRITICAL", "openai_api_key", re.compile(r"OPENAI_API_KEY", re.IGNORECASE)),
    ("CRITICAL", "anthropic_api_key", re.compile(r"ANTHROPIC_API_KEY", re.IGNORECASE)),
    ("CRITICAL", "google_api_key", re.compile(r"GOOGLE_API_KEY", re.IGNORECASE)),
    ("HIGH", "secret_word", re.compile(r"\bSECRET\b", re.IGNORECASE)),
    ("HIGH", "token_word", re.compile(r"\bTOKEN\b", re.IGNORECASE)),
    ("CRITICAL", "absolute_video_path", re.compile(r"[A-Z]:\\[^:\n]+?\.(mp4|mov|avi|mkv|webm|m4v)", re.IGNORECASE)),
    ("CRITICAL", "private_content_pack_reference", re.compile(r"content_pack", re.IGNORECASE)),
    ("CRITICAL", "private_publisher_pack_reference", re.compile(r"publisher_pack", re.IGNORECASE)),
    ("HIGH", "tiktok_reference", re.compile(r"TikTok", re.IGNORECASE)),
    ("HIGH", "youtube_tags_reference", re.compile(r"YouTube tags", re.IGNORECASE)),
]

if len(LOCAL_USERNAME) >= 3:
    PATTERNS.append(("CRITICAL", "personal_username", re.compile(re.escape(LOCAL_USERNAME), re.IGNORECASE)))

ALLOWLIST = {
    Path("lab/scripts/check_public_release_safety.py"),
    Path("scripts/check_release_package_safety.py"),
    Path("PRIVATE_MODULES.md"),
    Path("README.md"),
    Path("PUBLIC_RELEASE_CHECKLIST.md"),
    Path("PUBLIC_RELEASE_SMOKE.md"),
    Path("RELEASE_NOTES_V0.3.md"),
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def is_allowed(rel: Path, pattern_name: str) -> bool:
    if rel in ALLOWLIST:
        return True
    if rel == Path(".gitignore"):
        return pattern_name in {
            "env_file_reference",
            "private_content_pack_reference",
            "private_publisher_pack_reference",
        }
    return False


def main() -> int:
    findings: list[tuple[str, Path, int, str, str]] = []
    for path in iter_files():
        rel = path.relative_to(REPO_ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for severity, name, regex in PATTERNS:
                if regex.search(line) and not is_allowed(rel, name):
                    preview = line.strip()[:180]
                    findings.append((severity, rel, line_no, name, preview))

    for severity, rel, line_no, name, preview in findings:
        print(f"{severity}\t{rel}:{line_no}\t{name}\t{preview}")

    critical = [item for item in findings if item[0] == "CRITICAL"]
    if critical:
        print(f"\nPublic release safety check failed: {len(critical)} critical finding(s).")
        return 1
    print("Public release safety check passed: no critical findings.")
    if findings:
        print(f"Non-critical findings: {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
