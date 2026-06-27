import argparse
import json

from uap_forensics.autoencoder import run_autoencoder_analysis
from uap_forensics.io import load_case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train background autoencoder and produce anomaly metrics.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()
    result = run_autoencoder_analysis(load_case_config(args.case_id), epochs=args.epochs, patch_size=args.patch_size)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
