import argparse
import json

from uap_forensics.io import load_case_config
from uap_forensics.roi import save_roi_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve configured ROI and save ROI crops plus roi.json.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = save_roi_frames(load_case_config(args.case_id))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
