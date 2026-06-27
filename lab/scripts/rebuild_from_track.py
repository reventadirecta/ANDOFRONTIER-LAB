from __future__ import annotations

import argparse
import json

from uap_forensics.track_based_analysis import rebuild_from_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild analysis artifacts from a human-validated interactive track.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--with-autoencoder", action="store_true", help="Reserved optional mode; not run by default.")
    parser.add_argument("--margin", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(rebuild_from_track(args.case_id, with_autoencoder=args.with_autoencoder, margin=args.margin), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
