from __future__ import annotations

import argparse
import json

from uap_forensics.batch import run_batch_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quick batch triage pipeline.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--mode", default="quick", choices=["quick"])
    args = parser.parse_args()
    print(json.dumps(run_batch_pipeline(args.batch_id, args.mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

