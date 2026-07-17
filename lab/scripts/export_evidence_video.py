from __future__ import annotations

import argparse
import json

from uap_forensics.evidence_video_export import export_evidence_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evidence/review videos from existing case outputs.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--format", choices=["16x9", "9x16", "1x1", "all"], default="all")
    args = parser.parse_args()
    print(json.dumps(export_evidence_videos(args.case_id, args.format), indent=2))


if __name__ == "__main__":
    main()
