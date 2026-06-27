import argparse
import json

from uap_forensics.analysis_target import run_entity_analysis
from uap_forensics.io import load_case_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity-structure analysis excluding saturated light events.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    config = load_case_config(args.case_id)
    result = run_entity_analysis(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
