from __future__ import annotations

import argparse
import json

from uap_forensics.batch import batch_track_objects


def main() -> int:
    parser = argparse.ArgumentParser(description="Track object candidates for top batch cases before human review.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(batch_track_objects(args.batch_id, args.top), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
