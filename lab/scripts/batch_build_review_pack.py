from __future__ import annotations

import argparse
import json

from uap_forensics.batch import build_batch_review_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a visual human-review pack for top batch cases.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(build_batch_review_pack(args.batch_id, args.top), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
