from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uap_forensics.io import read_json, write_json
from uap_forensics.paths import CONFIG_DIR, DATA_DIR, ensure_dir

from .backends import backend_status, select_backend
from .cotracker_backend import CoTrackerBackend
from .cutie_backend import CutieBackend, XMemBackend
from .opencv_backend import OpenCVBackend, write_centroids_csv
from .sam2_backend import SAM2Backend
from .track_validation import create_or_update_validation, validation_path
from .visualization import contact_sheet, write_overlay_video, write_track_contact_sheet


def interactive_request_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "interactive_track_request.json"


def interactive_output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "interactive_tracking")


def _load_case(case_id: str) -> dict[str, Any]:
    config = CONFIG_DIR / f"{case_id}.json"
    if config.exists():
        case = read_json(config)
        case.setdefault("case_id", case_id)
        return case
    for manifest in (DATA_DIR / "batches").glob("*/batch_manifest.json"):
        data = read_json(manifest)
        for case in data.get("cases", []):
            if case.get("case_id") == case_id:
                return case
    raise FileNotFoundError(f"Case not found in config or batch manifests: {case_id}")


def _video_meta(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return meta


def _sample_temporal_frames(video_path: str, output: Path, max_samples: int = 24) -> list[int]:
    meta = _video_meta(video_path)
    total = meta["frame_count"]
    indices = sorted(set(np.linspace(0, max(0, total - 1), min(max_samples, max(1, total)), dtype=int).tolist()))
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    labels = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        labels.append(f"f{frame_idx}")
    cap.release()
    contact_sheet(frames, labels, output, cols=4)
    return indices


def _possible_appearance_frames(video_path: str, max_items: int = 12) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 1:
        cap.release()
        return []
    prev = None
    scores = []
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            scores.append({"frame": frame_idx, "motion_score": float(diff.mean()), "note": "candidate only; human must choose first true object frame"})
        prev = gray
    cap.release()
    scores.sort(key=lambda item: item["motion_score"], reverse=True)
    return sorted(scores[:max_items], key=lambda item: item["frame"])


def default_request(case_id: str, backend_preference: list[str] | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "tracking_mode": "manual_first_appearance",
        "backend_preference": backend_preference or ["sam2", "cotracker", "cutie", "xmem", "opencv"],
        "first_object_frame": None,
        "object_prompt": {
            "type": "box_or_points",
            "box": None,
            "positive_points": [],
            "negative_points": [],
        },
        "do_not_track_before_first_frame": True,
        "notes": "Set first_object_frame when the object enters the frame. Mark the real object only, not background texture.",
    }


def prepare_interactive_track(case_id: str) -> dict[str, Any]:
    case = _load_case(case_id)
    out_dir = interactive_output_dir(case_id)
    request_path = interactive_request_path(case_id)
    contact_path = out_dir / "temporal_contact_sheet.png"
    sampled = _sample_temporal_frames(case["video_path"], contact_path)
    candidates = _possible_appearance_frames(case["video_path"])
    if request_path.exists():
        request = read_json(request_path)
        template = default_request(case_id)
        for key, value in template.items():
            request.setdefault(key, value)
    else:
        request = default_request(case_id)
    request["video_path"] = case["video_path"]
    request["temporal_contact_sheet"] = str(contact_path)
    request["sampled_frames"] = sampled
    request["possible_appearance_frames"] = candidates
    request["backend_status"] = {name: {"available": info.available, "reason": info.reason} for name, info in backend_status().items()}
    ensure_dir(request_path.parent)
    write_json(request_path, request)
    return {"case_id": case_id, "request": str(request_path), "contact_sheet": str(contact_path), "possible_appearance_frames": candidates}


def _backend_instance(name: str):
    if name == "sam2":
        return SAM2Backend()
    if name == "cotracker":
        return CoTrackerBackend()
    if name == "cutie":
        return CutieBackend()
    if name == "xmem":
        return XMemBackend()
    if name == "opencv":
        return OpenCVBackend()
    raise ValueError(f"Unknown backend: {name}")


def _write_report(case_id: str, track: dict[str, Any], output: Path, request: dict[str, Any]) -> None:
    statuses = backend_status()
    lines = [
        f"# Interactive Tracking Report: {case_id}",
        "",
        f"Backend used: `{track.get('backend_used')}`",
        f"Tracker backend: `{track.get('tracker_backend')}`",
        f"First object frame: `{track.get('first_object_frame')}`",
        f"Frames tracked: `{track.get('frames_tracked')}`",
        f"Track status: `{track.get('track_status')}`",
        "",
        "## Rule",
        "",
        "- No ROI is drawn before `first_object_frame`.",
        "- This track is not valid for deep analysis until `track_validation.json` is explicitly validated.",
        "- Maximum brightness is not used as the primary target criterion.",
        "- HUD, reticle, border and fixed overlay candidates are rejected before they can become tracked targets.",
        "",
        "## Tracking Quality",
        "",
        "```json",
        json.dumps(track.get("tracking_quality", track.get("summary", {})), indent=2),
        "```",
        "",
        "## Backend Availability",
        "",
    ]
    for name, status in statuses.items():
        lines.append(f"- `{name}`: {'available' if status.available else 'unavailable'} ({status.reason})")
    lines.extend(
        [
            "",
            "## Prompt",
            "",
            "```json",
            json.dumps(request.get("object_prompt", {}), indent=2),
            "```",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in track.get("warnings", []) or ["none"]:
        lines.append(f"- {warning}")
    if track.get("summary"):
        lines.extend(["", "## Summary", "", "```json", json.dumps(track["summary"], indent=2), "```"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_web_mp4(input_path: Path, output_path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"ok": False, "error": "ffmpeg not found in PATH", "output": str(output_path)}
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(command, cwd=str(input_path.parent), capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0,
        "returncode": proc.returncode,
        "command": " ".join(command),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "output": str(output_path),
    }


def run_interactive_track(case_id: str, backend: str = "sam2", allow_fallback: bool = True, progress_callback=None) -> dict[str, Any]:
    case = _load_case(case_id)
    request_path = interactive_request_path(case_id)
    if not request_path.exists():
        raise FileNotFoundError(f"Interactive request not found. Run prepare_interactive_track first: {request_path}")
    request = read_json(request_path)
    if request.get("first_object_frame") is None:
        raise ValueError(f"Set first_object_frame in {request_path} before running interactive tracking.")
    if not request.get("do_not_track_before_first_frame", True):
        raise ValueError("do_not_track_before_first_frame must remain true for this workflow.")
    selected = select_backend(backend, allow_fallback=allow_fallback)
    request["backend_requested"] = backend
    out_dir = interactive_output_dir(case_id)
    if progress_callback:
        progress_callback(
            {
                "stage": "started",
                "message": f"Tracking started at frame {request.get('first_object_frame')}",
                "frame": int(request.get("first_object_frame")),
            }
        )
    instance = _backend_instance(selected)
    if selected == "opencv":
        track = instance.track(case, request, out_dir, progress=progress_callback)
    else:
        track = instance.track(case, request, out_dir)
    track_path = out_dir / "track.json"
    if progress_callback:
        progress_callback({"stage": "writing_track", "message": "Writing track.json"})
    write_json(track_path, track)
    write_json(out_dir / "tracking_quality.json", track.get("tracking_quality", track.get("summary", {})))
    write_centroids_csv(track, out_dir / "track_centroids.csv")
    overlay_path = out_dir / "track_overlay_preview.mp4"
    web_overlay_path = out_dir / "track_overlay_preview_web.mp4"
    if progress_callback:
        progress_callback({"stage": "rendering_overlay", "message": "Rendering overlay preview"})
    write_overlay_video(case, track, overlay_path)
    web_result = _write_web_mp4(overlay_path, web_overlay_path)
    track["web_overlay"] = web_result
    write_json(track_path, track)
    write_json(out_dir / "tracking_quality.json", track.get("tracking_quality", track.get("summary", {})))
    write_track_contact_sheet(case, track, out_dir / "track_contact_sheet.png")
    _write_report(case_id, track, out_dir / "tracking_report.md", request)
    create_or_update_validation(case_id, selected_backend=track.get("backend_used", selected))
    if progress_callback:
        progress_callback({"stage": "complete", "message": "Tracking complete"})
    return {
        "case_id": case_id,
        "backend_requested": backend,
        "backend_used": track.get("backend_used"),
        "track": str(track_path),
        "overlay": str(overlay_path),
        "web_overlay": str(web_overlay_path),
        "web_overlay_ok": web_result.get("ok", False),
        "contact_sheet": str(out_dir / "track_contact_sheet.png"),
        "validation": str(validation_path(case_id)),
        "frames_tracked": track.get("frames_tracked"),
    }


def render_existing_track_outputs(case_id: str) -> dict[str, Any]:
    """Regenerate visual previews from an existing track.json without modifying it."""
    case = _load_case(case_id)
    out_dir = interactive_output_dir(case_id)
    track_path = out_dir / "track.json"
    if not track_path.exists():
        raise FileNotFoundError(f"Existing track.json not found: {track_path}")
    track = read_json(track_path)
    overlay_path = out_dir / "track_overlay_preview.mp4"
    web_overlay_path = out_dir / "track_overlay_preview_web.mp4"
    contact_sheet_path = out_dir / "track_contact_sheet.png"
    write_overlay_video(case, track, overlay_path)
    web_result = _write_web_mp4(overlay_path, web_overlay_path)
    write_track_contact_sheet(case, track, contact_sheet_path)
    return {
        "case_id": case_id,
        "mode": "render_only",
        "track": str(track_path),
        "overlay": str(overlay_path),
        "web_overlay": str(web_overlay_path),
        "web_overlay_ok": web_result.get("ok", False),
        "contact_sheet": str(contact_sheet_path),
    }


def validate_track(case_id: str) -> dict[str, Any]:
    out_dir = interactive_output_dir(case_id)
    selected = "sam2"
    track_path = out_dir / "track.json"
    if track_path.exists():
        selected = read_json(track_path).get("backend_used", selected)
    validation = create_or_update_validation(case_id, selected_backend=selected)
    return {"case_id": case_id, "validation": str(validation_path(case_id)), "current": validation}
