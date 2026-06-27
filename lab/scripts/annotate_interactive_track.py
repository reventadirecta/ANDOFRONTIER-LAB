from __future__ import annotations

import argparse
import json

from uap_forensics.tracking.annotation import annotate_interactive_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a simple manual box annotator for interactive tracking.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(annotate_interactive_track(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
