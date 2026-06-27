import argparse
import json

from uap_forensics.io import load_case_config
from uap_forensics.reproduction_audit import run_reproduction_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ROI/PCA/autoencoder reproduction sensitivity for a case.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()
    result = run_reproduction_audit(
        load_case_config(args.case_id),
        seed=args.seed,
        epochs=args.epochs,
        patch_size=args.patch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
