from __future__ import annotations

import argparse
import json

from uap_forensics.unified_case_report import generate_unified_case_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a unified forensic report from existing case module outputs.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_unified_case_report(args.case_id), indent=2))


if __name__ == "__main__":
    main()
