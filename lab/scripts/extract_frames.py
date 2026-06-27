import argparse

from uap_forensics.frames import extract_frames
from uap_forensics.io import load_case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract PNG frames for a registered case.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = extract_frames(load_case_config(args.case_id))
    print(f"Saved {result['frames_saved']} frames to {result['frames_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
