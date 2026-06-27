import argparse

from uap_forensics.io import load_case_config
from uap_forensics.report import generate_case_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown forensic report for a case.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    path = generate_case_report(load_case_config(args.case_id))
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
