from __future__ import annotations

import argparse
import json

from uap_forensics.reddit_post_template import generate_reddit_post_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public Reddit technical post templates from a unified case report.")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_reddit_post_template(args.case_id), indent=2))


if __name__ == "__main__":
    main()
