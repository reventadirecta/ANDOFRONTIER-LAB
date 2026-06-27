from __future__ import annotations

import argparse
import json

from uap_forensics.track_spectral_analysis import run_track_spectral_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run track-based spectral analysis from a human-validated track.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_track_spectral_analysis(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
