from __future__ import annotations

import argparse
import json

from uap_forensics.track_srv_analysis import run_track_srv_analysis
from uap_forensics.track_srv_core_analysis import run_track_srv_core_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run conservative track-based SRV / visual reconstruction.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--object-core", action="store_true", help="Run object-core SRV inside the validated tracking bbox.")
    args = parser.parse_args()
    if args.object_core:
        print(json.dumps(run_track_srv_core_analysis(args.case_id), indent=2))
        return 0
    print(json.dumps(run_track_srv_analysis(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
