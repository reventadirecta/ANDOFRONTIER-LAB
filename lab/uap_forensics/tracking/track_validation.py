from __future__ import annotations

from pathlib import Path
from typing import Any

from uap_forensics.io import read_json, write_json
from uap_forensics.paths import DATA_DIR, ensure_dir


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def default_validation(case_id: str, selected_backend: str = "sam2") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "track_validated": False,
        "selected_backend": selected_backend,
        "track_is_correct": None,
        "object_is_real_target": None,
        "needs_reprompt": None,
        "bad_segments": [],
        "notes": "",
    }


def create_or_update_validation(case_id: str, selected_backend: str = "sam2") -> dict[str, Any]:
    path = validation_path(case_id)
    if path.exists():
        data = read_json(path)
        template = default_validation(case_id, selected_backend)
        for key, value in template.items():
            data.setdefault(key, value)
        if selected_backend:
            data["selected_backend"] = data.get("selected_backend") or selected_backend
    else:
        data = default_validation(case_id, selected_backend)
    ensure_dir(path.parent)
    write_json(path, data)
    return data


def is_validated(case_id: str) -> bool:
    path = validation_path(case_id)
    if not path.exists():
        return False
    data = read_json(path)
    return bool(data.get("track_validated") and data.get("track_is_correct") and data.get("object_is_real_target"))
