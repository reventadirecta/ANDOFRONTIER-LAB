import argparse
import json

from uap_forensics.io import load_case_config
from uap_forensics.pca_analysis import run_pca_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PCA/SVD analysis for an existing ROI frame set.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--max-components", type=int, default=20)
    args = parser.parse_args()
    result = run_pca_analysis(load_case_config(args.case_id), max_components=args.max_components)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
