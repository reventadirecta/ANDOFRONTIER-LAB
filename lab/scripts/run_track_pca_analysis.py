from __future__ import annotations

import argparse
import json

from uap_forensics.track_pca_analysis import run_track_pca_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA against clean controls baseline from a human-validated track.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_track_pca_analysis(args.case_id), indent=2))


if __name__ == "__main__":
    main()
