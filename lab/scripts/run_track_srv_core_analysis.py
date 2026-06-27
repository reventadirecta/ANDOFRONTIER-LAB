from __future__ import annotations

import argparse
import json

from uap_forensics.track_srv_core_analysis import run_track_srv_core_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run object-core SRV inside human-validated tracking bboxes.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_track_srv_core_analysis(args.case_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
