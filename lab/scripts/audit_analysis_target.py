import argparse
import json

from uap_forensics.analysis_target import audit_analysis_target
from uap_forensics.io import load_case_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether prior analysis targets light or entity structure.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    config = load_case_config(args.case_id)
    result = audit_analysis_target(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
