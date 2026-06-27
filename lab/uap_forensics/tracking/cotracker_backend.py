from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import TrackingBackend, backend_status


class CoTrackerBackend(TrackingBackend):
    name = "cotracker"

    def track(self, case: dict[str, Any], request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        status = backend_status()["cotracker"]
        if not status.available:
            raise RuntimeError(f"CoTracker backend unavailable: {status.reason}. Install/configure CoTracker locally or use --backend opencv.")
        raise NotImplementedError("CoTracker adapter is detected but model wiring/checkpoint configuration is not implemented in this local pass.")
