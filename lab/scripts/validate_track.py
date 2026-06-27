from __future__ import annotations

import argparse
import json

from uap_forensics.tracking import validate_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update human validation state for an interactive track.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_track(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
