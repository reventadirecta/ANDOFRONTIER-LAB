from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import TrackingBackend, backend_status


class SAM2Backend(TrackingBackend):
    name = "sam2"

    def track(self, case: dict[str, Any], request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        status = backend_status()["sam2"]
        if not status.available:
            raise RuntimeError(f"SAM2 backend unavailable: {status.reason}. Install/configure SAM2 locally or use --backend opencv.")
        raise NotImplementedError("SAM2 adapter is detected but model wiring/checkpoint configuration is not implemented in this local pass.")
