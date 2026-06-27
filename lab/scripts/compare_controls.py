import argparse

from uap_forensics.controls import compare_against_controls


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a case against already-processed control cases.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--control-case-id", action="append", default=[], help="Processed control case id")
    args = parser.parse_args()
    path = compare_against_controls(args.case_id, args.control_case_id)
    print(f"Comparison table written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
