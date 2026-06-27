from __future__ import annotations

import argparse
import json

from uap_forensics.track_controls_analysis import run_track_controls_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controls analysis from a human-validated track and dynamic ROIs.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_track_controls_analysis(args.case_id), indent=2))


if __name__ == "__main__":
    main()
