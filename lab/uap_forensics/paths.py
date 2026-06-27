import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ANDOFRONTIER_LAB_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_config_path(case_id: str) -> Path:
    return CONFIG_DIR / f"{case_id}.json"


def case_frames_dir(case_id: str) -> Path:
    return DATA_DIR / "frames" / case_id


def case_roi_dir(case_id: str) -> Path:
    return DATA_DIR / "roi" / case_id


def case_output_dir(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id


def case_report_dir(case_id: str) -> Path:
    return DATA_DIR / "reports" / case_id
