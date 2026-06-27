from __future__ import annotations

import argparse
import json

from uap_forensics.batch import probe_batch_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh metadata/probe information for a registered batch.")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(json.dumps(probe_batch_sources(args.batch_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

