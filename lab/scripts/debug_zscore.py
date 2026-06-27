import argparse
import json

from uap_forensics.io import load_case_config
from uap_forensics.zscore_debug import run_zscore_debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug z-score sensitivity to object patch and background choices.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-background-patches", type=int, default=4000)
    args = parser.parse_args()
    result = run_zscore_debug(
        load_case_config(args.case_id),
        seed=args.seed,
        max_background_patches=args.max_background_patches,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
