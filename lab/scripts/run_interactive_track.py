from __future__ import annotations

import argparse
import json
import sys

from uap_forensics.tracking import render_existing_track_outputs, run_interactive_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Run interactive tracking from a human first-frame prompt.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--backend", default="sam2", choices=["sam2", "cotracker", "cutie", "xmem", "opencv"])
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of falling back to OpenCV when requested backend is unavailable.")
    parser.add_argument("--render-only", action="store_true", help="Regenerate overlay previews from existing track.json without re-running tracking.")
    parser.add_argument("--progress-jsonl", action="store_true", help="Emit tracking progress events as one JSON object per line.")
    args = parser.parse_args()
    def emit_progress(payload: dict) -> None:
        try:
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except OSError:
            return

    if args.render_only:
        print(json.dumps(render_existing_track_outputs(args.case_id), indent=2))
        return 0
    print(
        json.dumps(
            run_interactive_track(
                args.case_id,
                args.backend,
                allow_fallback=not args.no_fallback,
                progress_callback=emit_progress if args.progress_jsonl else None,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
