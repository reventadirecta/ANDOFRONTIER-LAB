import argparse
from pathlib import Path

from uap_forensics.source import register_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a video source and write source/config JSON.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--case-id", help="Stable case id. Defaults to video stem.")
    parser.add_argument("--origin", default="local file", help="Human-readable source origin")
    parser.add_argument("--source-url", default=None, help="Source URL if available")
    parser.add_argument(
        "--source-type",
        default="unknown",
        choices=[
            "original",
            "export directo",
            "export direct",
            "YouTube-derived",
            "Reddit-derived",
            "youtube-derived",
            "reddit-derived",
            "4chan mirror",
            "unknown",
        ],
    )
    parser.add_argument("--notes", default="", help="Chain-of-custody notes")
    args = parser.parse_args()
    video = Path(args.video)
    case_id = args.case_id or video.stem.replace(" ", "_").lower()
    record = register_source(video, case_id, args.origin, args.source_type, args.source_url, args.notes)
    print(f"Registered source for case: {record['case_id']}")
    print(f"SHA256: {record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
