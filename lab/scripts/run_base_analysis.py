import argparse
import json

from uap_forensics.analysis import run_base_analysis
from uap_forensics.io import load_case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run base visual analysis for an existing ROI frame set.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = run_base_analysis(load_case_config(args.case_id))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
