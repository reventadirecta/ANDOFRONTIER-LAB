from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    reason: str


BACKEND_MODULES = {
    "sam2": ["sam2"],
    "cotracker": ["cotracker"],
    "cutie": ["cutie"],
    "xmem": ["xmem"],
    "opencv": ["cv2"],
}


def backend_status() -> dict[str, BackendInfo]:
    statuses: dict[str, BackendInfo] = {}
    for name, modules in BACKEND_MODULES.items():
        missing = [module for module in modules if importlib.util.find_spec(module) is None]
        if missing:
            statuses[name] = BackendInfo(name=name, available=False, reason=f"missing module(s): {', '.join(missing)}")
        else:
            statuses[name] = BackendInfo(name=name, available=True, reason="available")
    return statuses


def select_backend(requested: str, allow_fallback: bool = True) -> str:
    statuses = backend_status()
    requested = requested.lower()
    if requested not in statuses:
        raise ValueError(f"Unknown tracking backend: {requested}")
    if statuses[requested].available:
        return requested
    if allow_fallback and statuses["opencv"].available:
        return "opencv"
    raise RuntimeError(f"Backend {requested} is not available: {statuses[requested].reason}")


class TrackingBackend:
    name = "base"

    def track(self, case: dict[str, Any], request: dict[str, Any], output_dir) -> dict[str, Any]:
        raise NotImplementedError
