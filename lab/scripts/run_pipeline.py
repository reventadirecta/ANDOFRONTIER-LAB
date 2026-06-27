import argparse
import json

from uap_forensics.io import load_case_config
from uap_forensics.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction, ROI, base analysis, PCA and autoencoder.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--skip-autoencoder", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(load_case_config(args.case_id), skip_autoencoder=args.skip_autoencoder)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
