import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from .io import write_json
from .paths import CONFIG_DIR, DATA_DIR


VALID_SOURCE_TYPES = {
    "original",
    "export directo",
    "export direct",
    "YouTube-derived",
    "Reddit-derived",
    "youtube-derived",
    "reddit-derived",
    "4chan mirror",
    "unknown",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_metadata(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {"available": False, "error": "ffprobe not found on PATH"}
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"available": True, "error": result.stderr.strip()}
    return json.loads(result.stdout or "{}")


def opencv_video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "OpenCV could not open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip()
    cap.release()
    duration = frame_count / fps if fps else None
    return {
        "duration_seconds": duration,
        "resolution": {"width": width, "height": height},
        "fps": fps,
        "frame_count": frame_count,
        "codec": codec,
    }


def register_source(
    video: Path,
    case_id: str,
    origin: str,
    source_type: str,
    source_url: str | None,
    notes: str,
) -> dict[str, Any]:
    video = video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type: {source_type}")

    ffprobe = ffprobe_metadata(video)
    cv_meta = opencv_video_metadata(video)
    bitrate = None
    if isinstance(ffprobe.get("format"), dict):
        bitrate = ffprobe["format"].get("bit_rate")

    record = {
        "case_id": case_id,
        "video_path": str(video),
        "origin": origin,
        "source_url": source_url,
        "source_type": source_type,
        "chain_of_custody_notes": notes,
        "sha256": sha256_file(video),
        "duration_seconds": cv_meta.get("duration_seconds"),
        "resolution": cv_meta.get("resolution"),
        "fps": cv_meta.get("fps"),
        "codec": cv_meta.get("codec"),
        "bitrate": bitrate,
        "metadata_ffprobe": ffprobe,
        "metadata_opencv": cv_meta,
    }
    write_json(DATA_DIR / "sources" / f"{case_id}.source.json", record)

    case_config = {
        "case_id": case_id,
        "video_path": str(video),
        "source_type": source_type,
        "source_url": source_url,
        "notes": notes,
        "roi": {"mode": "manual", "x": 0, "y": 0, "width": 0, "height": 0},
        "frame_window": {"start": 0, "end": None, "step": 1},
        "output_dir": f"data/outputs/{case_id}",
        "control_mode": False,
    }
    write_json(CONFIG_DIR / f"{case_id}.json", case_config)
    return record
