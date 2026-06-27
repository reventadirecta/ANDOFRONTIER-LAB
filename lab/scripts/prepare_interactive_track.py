from __future__ import annotations

import argparse
import json

from uap_forensics.tracking import prepare_interactive_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an editable manual-first-frame interactive tracking request.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_interactive_track(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
