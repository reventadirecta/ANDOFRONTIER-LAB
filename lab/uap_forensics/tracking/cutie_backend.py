from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import TrackingBackend, backend_status


class CutieBackend(TrackingBackend):
    name = "cutie"

    def track(self, case: dict[str, Any], request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        status = backend_status()["cutie"]
        if not status.available:
            raise RuntimeError(f"Cutie backend unavailable: {status.reason}. Install/configure Cutie locally or use --backend opencv.")
        raise NotImplementedError("Cutie adapter is detected but model wiring/checkpoint configuration is not implemented in this local pass.")


class XMemBackend(TrackingBackend):
    name = "xmem"

    def track(self, case: dict[str, Any], request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        status = backend_status()["xmem"]
        if not status.available:
            raise RuntimeError(f"XMem backend unavailable: {status.reason}. Install/configure XMem locally or use --backend opencv.")
        raise NotImplementedError("XMem adapter is detected but model wiring/checkpoint configuration is not implemented in this local pass.")
