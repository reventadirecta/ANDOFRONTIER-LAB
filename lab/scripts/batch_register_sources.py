from __future__ import annotations

import argparse
import json

from uap_forensics.batch import INPUT_FOLDER, register_batch_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Register all videos in a local batch folder.")
    parser.add_argument("--input-folder", default=INPUT_FOLDER)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    if args.input_folder == INPUT_FOLDER:
        raise SystemExit("Set --input-folder or edit INPUT_FOLDER before running.")
    result = register_batch_sources(args.input_folder, args.batch_id)
    print(json.dumps({"batch_id": args.batch_id, "total_videos": result["total_videos"], "manifest": f"data/batches/{args.batch_id}/batch_manifest.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

