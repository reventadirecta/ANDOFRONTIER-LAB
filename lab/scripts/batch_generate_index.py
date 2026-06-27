from __future__ import annotations

import argparse
import json

from uap_forensics.batch import generate_batch_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate batch triage index.")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_batch_index(args.batch_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
