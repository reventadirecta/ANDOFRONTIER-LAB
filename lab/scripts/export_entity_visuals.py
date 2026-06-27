import argparse
import json

from uap_forensics.entity_visuals import export_entity_visuals
from uap_forensics.io import load_case_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export content-ready visuals for entity_structure_analysis.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    config = load_case_config(args.case_id)
    result = export_entity_visuals(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
