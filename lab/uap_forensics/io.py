import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_case_config(case_id: str) -> dict[str, Any]:
    from .paths import case_config_path

    path = case_config_path(case_id)
    if not path.exists():
        raise FileNotFoundError(f"Case config not found: {path}")
    data = read_json(path)
    data.setdefault("case_id", case_id)
    return data
