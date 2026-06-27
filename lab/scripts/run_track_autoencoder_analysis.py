from __future__ import annotations

import argparse
import json

from uap_forensics.track_autoencoder_analysis import run_track_autoencoder_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoencoder analysis against clean controls baseline from a human-validated track.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--quick", action="store_true", help="Use fewer samples/epochs for a faster CPU validation pass.")
    args = parser.parse_args()
    print(json.dumps(run_track_autoencoder_analysis(args.case_id, quick=args.quick), indent=2))


if __name__ == "__main__":
    main()
