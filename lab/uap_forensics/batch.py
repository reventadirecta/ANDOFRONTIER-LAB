from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import read_json, write_json
from .paths import CONFIG_DIR, DATA_DIR, ensure_dir
from .source import ffprobe_metadata, opencv_video_metadata, sha256_file


INPUT_FOLDER = "PEGAR_AQUI_RUTA_DE_LA_CARPETA_CON_VIDEOS"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def batch_dir(batch_id: str) -> Path:
    return ensure_dir(DATA_DIR / "batches" / batch_id)


def batch_manifest_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "batch_manifest.json"


def batch_report_dir(batch_id: str) -> Path:
    return ensure_dir(DATA_DIR / "reports" / "batches" / batch_id)


def case_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "cases" / case_id)


def batch_case_output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "batch_quick")


def batch_case_report_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "reports" / case_id)


def _slug(text: str) -> str:
    stem = Path(text).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "case"


def _unique_case_id(video: Path, used: set[str]) -> str:
    base = _slug(video.stem)
    case_id = base
    suffix = 2
    while case_id in used or (CONFIG_DIR / f"{case_id}.json").exists():
        if (CONFIG_DIR / f"{case_id}.json").exists() and case_id not in used:
            existing = read_json(CONFIG_DIR / f"{case_id}.json")
            if Path(existing.get("video_path", "")).resolve() == video.resolve():
                break
        case_id = f"{base}_{suffix:02d}"
        suffix += 1
    used.add(case_id)
    return case_id


def _backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    if path.is_file():
        shutil.copy2(path, backup)


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    _backup(path)
    write_json(path, payload)


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    path.write_text(text, encoding="utf-8")


def find_videos(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    return sorted(
        p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _stream_info(ffprobe: dict[str, Any]) -> dict[str, Any]:
    streams = ffprobe.get("streams") if isinstance(ffprobe, dict) else None
    if not isinstance(streams, list):
        return {"audio": None, "video_codec": None, "bitrate": None}
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = any(s.get("codec_type") == "audio" for s in streams)
    return {
        "audio": audio,
        "video_codec": video.get("codec_name"),
        "bitrate": (ffprobe.get("format") or {}).get("bit_rate") if isinstance(ffprobe.get("format"), dict) else None,
    }


def _source_quality(video: Path) -> tuple[str, str]:
    name = video.name.lower()
    if "youtube" in name or "yt-" in name:
        return "secondary_copy", "Filename suggests platform-derived copy."
    if "reddit" in name:
        return "secondary_copy", "Filename suggests Reddit-derived copy."
    return "unknown", "Local file has no verifiable release URL or custody evidence."


def _register_case_config(case: dict[str, Any]) -> None:
    case_id = case["case_id"]
    source = {
        "case_id": case_id,
        "video_path": case["video_path"],
        "origin": "local batch folder",
        "source_url": None,
        "source_type": case["source_quality"],
        "source_quality_note": case["source_quality_note"],
        "chain_of_custody_notes": "Batch import from local folder; original release provenance not verified.",
        "sha256": case["sha256"],
        "duration_seconds": case.get("duration_seconds"),
        "resolution": case.get("resolution"),
        "fps": case.get("fps"),
        "codec": case.get("codec"),
        "bitrate": case.get("bitrate"),
        "audio": case.get("audio"),
        "size_bytes": case.get("size_bytes"),
        "original_filename": case.get("original_filename"),
        "metadata_ffprobe": case.get("metadata_ffprobe", {}),
        "metadata_opencv": case.get("metadata_opencv", {}),
    }
    _safe_write_json(DATA_DIR / "sources" / f"{case_id}.source.json", source)
    cfg = {
        "case_id": case_id,
        "video_path": case["video_path"],
        "source_type": case["source_quality"],
        "source_url": None,
        "notes": "Batch quick triage case. ROI is proposed, not definitive.",
        "roi": {"mode": "manual_review_required", "x": 0, "y": 0, "width": 0, "height": 0},
        "frame_window": {"start": 0, "end": None, "step": 1},
        "output_dir": f"data/outputs/{case_id}",
        "control_mode": False,
    }
    _safe_write_json(CONFIG_DIR / f"{case_id}.json", cfg)
    _safe_write_json(case_dir(case_id) / "batch_source.json", source)


def register_batch_sources(input_folder: str, batch_id: str) -> dict[str, Any]:
    root = Path(input_folder).expanduser().resolve()
    videos = find_videos(root)
    used: set[str] = set()
    cases = []
    for video in videos:
        case_id = _unique_case_id(video, used)
        quality, quality_note = _source_quality(video)
        cv_meta = opencv_video_metadata(video)
        ffprobe = ffprobe_metadata(video)
        stream = _stream_info(ffprobe)
        resolution = cv_meta.get("resolution") or {"width": 0, "height": 0}
        case = {
            "batch_id": batch_id,
            "case_id": case_id,
            "original_filename": video.name,
            "video_path": str(video),
            "relative_path": str(video.relative_to(root)),
            "extension": video.suffix.lower(),
            "size_bytes": video.stat().st_size,
            "sha256": sha256_file(video),
            "duration_seconds": cv_meta.get("duration_seconds"),
            "resolution": resolution,
            "fps": cv_meta.get("fps"),
            "codec": stream.get("video_codec") or cv_meta.get("codec"),
            "bitrate": stream.get("bitrate"),
            "audio": stream.get("audio"),
            "source_quality": quality,
            "source_quality_note": quality_note,
            "metadata_ffprobe": ffprobe,
            "metadata_opencv": cv_meta,
            "status": "registered",
        }
        _register_case_config(case)
        cases.append(case)
    manifest = {
        "batch_id": batch_id,
        "input_folder": str(root),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_extensions": sorted(VIDEO_EXTENSIONS),
        "total_videos": len(cases),
        "cases": cases,
    }
    _safe_write_json(batch_manifest_path(batch_id), manifest)
    return manifest


def load_batch(batch_id: str) -> dict[str, Any]:
    path = batch_manifest_path(batch_id)
    if not path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {path}")
    return read_json(path)


def save_batch(manifest: dict[str, Any]) -> None:
    _safe_write_json(batch_manifest_path(manifest["batch_id"]), manifest)


def probe_batch_sources(batch_id: str) -> dict[str, Any]:
    manifest = load_batch(batch_id)
    errors = []
    for case in manifest["cases"]:
        path = Path(case["video_path"])
        try:
            cv_meta = opencv_video_metadata(path)
            ffprobe = ffprobe_metadata(path)
            stream = _stream_info(ffprobe)
            case.update(
                {
                    "duration_seconds": cv_meta.get("duration_seconds"),
                    "resolution": cv_meta.get("resolution"),
                    "fps": cv_meta.get("fps"),
                    "frame_count": cv_meta.get("frame_count"),
                    "codec": stream.get("video_codec") or cv_meta.get("codec"),
                    "bitrate": stream.get("bitrate"),
                    "audio": stream.get("audio"),
                    "metadata_ffprobe": ffprobe,
                    "metadata_opencv": cv_meta,
                    "probe_status": "ok" if "error" not in cv_meta else "error",
                }
            )
        except Exception as exc:  # noqa: BLE001
            case["probe_status"] = "error"
            case["error"] = str(exc)
            errors.append({"case_id": case["case_id"], "error": str(exc)})
    save_batch(manifest)
    return {"batch_id": batch_id, "total": len(manifest["cases"]), "errors": errors}


def _sample_frames(video_path: Path, max_samples: int = 24) -> tuple[list[int], list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        indices = list(range(max_samples))
    else:
        n = min(max_samples, frame_count)
        indices = sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, n)))
    frames = []
    actual = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            actual.append(idx)
    cap.release()
    return actual, frames, {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def _resize_for_grid(frame: np.ndarray, width: int = 240) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0:
        return frame
    return cv2.resize(frame, (width, max(1, int(width * h / w))), interpolation=cv2.INTER_AREA)


def _contact_sheet(frames: list[np.ndarray], labels: list[str], out: Path, cols: int = 6) -> None:
    if not frames:
        return
    thumbs = [_resize_for_grid(f) for f in frames]
    th = max(t.shape[0] for t in thumbs) + 28
    tw = max(t.shape[1] for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * th + 24, c * tw
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        cv2.putText(sheet, labels[i], (x + 6, r * th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def _roi_from_motion(frames: list[np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(frames) < 3:
        return {"status": "manual_review_required", "x": 0, "y": 0, "w": 0, "h": 0}, {"reason": "not enough frames"}
    small = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(gray, (320, max(1, int(320 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA))
    acc = np.zeros_like(small[0], dtype=np.float32)
    prev = small[0]
    for cur in small[1:]:
        acc += cv2.absdiff(prev, cur).astype(np.float32)
        prev = cur
    acc = cv2.GaussianBlur(acc, (9, 9), 0)
    norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_val = max(18, int(np.percentile(norm, 96)))
    _, mask = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "manual_review_required", "x": 0, "y": 0, "w": 0, "h": 0}, {"reason": "no persistent motion/contrast candidate"}
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h0, w0 = frames[0].shape[:2]
    sh, sw = small[0].shape[:2]
    best = contours[0]
    x, y, w, h = cv2.boundingRect(best)
    area_frac = (w * h) / float(sw * sh)
    scale_x, scale_y = w0 / sw, h0 / sh
    pad = 0.35
    rx = max(0, int((x - w * pad) * scale_x))
    ry = max(0, int((y - h * pad) * scale_y))
    rw = min(w0 - rx, int(w * (1 + 2 * pad) * scale_x))
    rh = min(h0 - ry, int(h * (1 + 2 * pad) * scale_y))
    confidence = "medium" if 0.0003 <= area_frac <= 0.35 else "low"
    status = "proposed" if confidence == "medium" else "manual_review_required"
    return (
        {"status": status, "x": rx, "y": ry, "w": rw, "h": rh, "confidence": confidence},
        {
            "motion_area_fraction": area_frac,
            "motion_threshold": thresh_val,
            "motion_score_mean": float(acc.mean()),
            "motion_score_max": float(acc.max()),
        },
    )


def _metrics(frames: list[np.ndarray], roi: dict[str, Any]) -> dict[str, Any]:
    if not frames:
        return {}
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [float(cv2.absdiff(grays[i - 1], grays[i]).mean()) for i in range(1, len(grays))]
    sat_fracs = []
    contrast = []
    roi_contrast = []
    for frame, gray in zip(frames, grays):
        sat_fracs.append(float((gray >= 245).mean()))
        contrast.append(float(gray.std()))
        if roi.get("w", 0) and roi.get("h", 0):
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            crop = gray[y:y + h, x:x + w]
            if crop.size:
                roi_contrast.append(float(crop.std()))
    return {
        "mean_frame_difference": float(np.mean(diffs)) if diffs else 0.0,
        "max_frame_difference": float(np.max(diffs)) if diffs else 0.0,
        "mean_saturated_fraction": float(np.mean(sat_fracs)),
        "max_saturated_fraction": float(np.max(sat_fracs)),
        "mean_contrast": float(np.mean(contrast)),
        "mean_roi_contrast": float(np.mean(roi_contrast)) if roi_contrast else None,
    }


def _priority(case: dict[str, Any], roi: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, str]:
    if case.get("probe_status") == "error" or case.get("quick_status") == "error":
        return "discard_candidate", "video could not be processed"
    width = (case.get("resolution") or {}).get("width") or 0
    height = (case.get("resolution") or {}).get("height") or 0
    duration = case.get("duration_seconds") or 0
    if not roi or roi.get("status") == "manual_review_required":
        return "manual_review_required", "ROI candidate not reliable enough"
    if width < 320 or height < 180 or duration < 1:
        return "low_priority", "low resolution or very short clip"
    motion = metrics.get("mean_frame_difference", 0) or 0
    sat = metrics.get("max_saturated_fraction", 0) or 0
    roi_contrast = metrics.get("mean_roi_contrast") or 0
    if motion > 6 and roi_contrast > 20 and sat < 0.12:
        return "high_priority", "clear motion/contrast candidate with acceptable saturation"
    if motion > 2 and roi_contrast > 10:
        return "medium_priority", "usable candidate with limitations"
    return "low_priority", "weak motion/contrast signal"


def _draw_panel(case: dict[str, Any], frames: list[np.ndarray], labels: list[str], roi: dict[str, Any], out: Path) -> None:
    if not frames:
        return
    selected = frames[: min(6, len(frames))]
    panels = []
    for i, frame in enumerate(selected):
        img = frame.copy()
        if roi.get("w", 0) and roi.get("h", 0):
            cv2.rectangle(img, (int(roi["x"]), int(roi["y"])), (int(roi["x"] + roi["w"]), int(roi["y"] + roi["h"])), (0, 255, 0), 2)
        panels.append(img)
    _contact_sheet(panels, labels[: len(panels)], out, cols=3)


def _write_case_reports(case: dict[str, Any], roi: dict[str, Any], metrics: dict[str, Any], priority: str, reason: str) -> None:
    report_dir = batch_case_report_dir(case["case_id"])
    summary = f"""# Case Summary

Case: `{case['case_id']}`

Original file: `{case['original_filename']}`

Source quality: `{case.get('source_quality', 'unknown')}`

Priority: `{priority}`

Reason: {reason}

ROI status: `{roi.get('status', 'unknown')}`

This is batch quick triage only. It is not `entity_structure_analysis`.
"""
    _safe_write_text(report_dir / "case_summary.md", summary)
    source_quality = f"""# Source Quality

Case: `{case['case_id']}`

Classification: `{case.get('source_quality', 'unknown')}`

Note: {case.get('source_quality_note', 'No custody note.')}

SHA256: `{case.get('sha256')}`

Original filename: `{case.get('original_filename')}`

Video path: `{case.get('video_path')}`
"""
    _safe_write_text(report_dir / "source_quality.md", source_quality)
    quick = f"""# Quick Analysis Report

Case: `{case['case_id']}`

Mode: `quick`

Priority: `{priority}`

ROI proposal:

```json
{json.dumps(roi, indent=2)}
```

Metrics:

```json
{json.dumps(metrics, indent=2)}
```

Limitations:

- ROI is proposed, not definitive.
- No heavy autoencoder is run by default.
- No `entity_structure_analysis` is run by default.
- If ROI is uncertain, case is marked for manual review.
"""
    _safe_write_text(report_dir / "quick_analysis_report.md", quick)


def _write_metrics_csv(path: Path, case: dict[str, Any], metrics: dict[str, Any], roi: dict[str, Any], priority: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "priority", "roi_status", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"case_id": case["case_id"], "priority": priority, "roi_status": roi.get("status"), **metrics})


def run_batch_pipeline(batch_id: str, mode: str = "quick") -> dict[str, Any]:
    if mode != "quick":
        raise ValueError("Only quick mode is implemented for batch triage.")
    manifest = load_batch(batch_id)
    errors = []
    for case in manifest["cases"]:
        try:
            video = Path(case["video_path"])
            out_dir = batch_case_output_dir(case["case_id"])
            indices, frames, sample_meta = _sample_frames(video)
            labels = [f"f{idx}" for idx in indices]
            _contact_sheet(frames, labels, out_dir / "contact_sheet.png")
            roi, roi_metrics = _roi_from_motion(frames)
            metrics = {**_metrics(frames, roi), **roi_metrics}
            priority, reason = _priority(case, roi, metrics)
            _draw_panel(case, frames, labels, roi, out_dir / "quick_panel.png")
            _safe_write_json(case_dir(case["case_id"]) / "roi_proposal.json", {"case_id": case["case_id"], "roi": roi, "metrics": roi_metrics})
            _safe_write_json(out_dir / "quick_metrics.json", {"case_id": case["case_id"], "metrics": metrics, "roi": roi, "priority": priority})
            _write_metrics_csv(out_dir / "quick_metrics.csv", case, metrics, roi, priority)
            _write_case_reports(case, roi, metrics, priority, reason)
            case.update(
                {
                    "quick_status": "ok",
                    "sampled_frames": len(frames),
                    "sampled_frame_indices": indices,
                    "sample_meta": sample_meta,
                    "roi_proposal": roi,
                    "quick_metrics": metrics,
                    "priority": priority,
                    "priority_reason": reason,
                    "quick_outputs": {
                        "contact_sheet": str(out_dir / "contact_sheet.png"),
                        "quick_panel": str(out_dir / "quick_panel.png"),
                        "metrics_csv": str(out_dir / "quick_metrics.csv"),
                        "case_summary": str(batch_case_report_dir(case["case_id"]) / "case_summary.md"),
                        "source_quality": str(batch_case_report_dir(case["case_id"]) / "source_quality.md"),
                        "quick_analysis_report": str(batch_case_report_dir(case["case_id"]) / "quick_analysis_report.md"),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            case["quick_status"] = "error"
            case["error"] = str(exc)
            case["priority"] = "discard_candidate"
            case["priority_reason"] = "processing error"
            errors.append({"case_id": case.get("case_id"), "error": str(exc)})
    save_batch(manifest)
    return {"batch_id": batch_id, "mode": mode, "processed": len(manifest["cases"]) - len(errors), "errors": errors}


PRIORITY_ORDER = {
    "high_priority": 0,
    "medium_priority": 1,
    "low_priority": 2,
    "manual_review_required": 3,
    "discard_candidate": 4,
}


def generate_batch_index(batch_id: str) -> dict[str, Any]:
    manifest = load_batch(batch_id)
    cases = sorted(manifest["cases"], key=lambda c: (PRIORITY_ORDER.get(c.get("priority", "manual_review_required"), 9), c.get("case_id", "")))
    report_dir = batch_report_dir(batch_id)
    triage = {
        "batch_id": batch_id,
        "total_cases": len(cases),
        "counts": {key: 0 for key in PRIORITY_ORDER},
        "cases": [],
    }
    for case in cases:
        priority = case.get("priority", "manual_review_required")
        triage["counts"][priority] = triage["counts"].get(priority, 0) + 1
        triage["cases"].append(
            {
                "case_id": case["case_id"],
                "priority": priority,
                "reason": case.get("priority_reason"),
                "original_filename": case.get("original_filename"),
                "duration_seconds": case.get("duration_seconds"),
                "resolution": case.get("resolution"),
                "source_quality": case.get("source_quality"),
                "quick_status": case.get("quick_status"),
                "report": f"data/reports/{case['case_id']}/quick_analysis_report.md",
            }
        )
    _safe_write_json(report_dir / "batch_triage.json", triage)
    csv_path = report_dir / "batch_index.csv"
    _backup(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["case_id", "priority", "reason", "original_filename", "duration_seconds", "width", "height", "source_quality", "quick_status", "report"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in triage["cases"]:
            res = item.get("resolution") or {}
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "priority": item["priority"],
                    "reason": item.get("reason"),
                    "original_filename": item.get("original_filename"),
                    "duration_seconds": item.get("duration_seconds"),
                    "width": res.get("width"),
                    "height": res.get("height"),
                    "source_quality": item.get("source_quality"),
                    "quick_status": item.get("quick_status"),
                    "report": item.get("report"),
                }
            )
    lines = [
        f"# Batch Index: {batch_id}",
        "",
        f"Input folder: `{manifest.get('input_folder')}`",
        "",
        "## Counts",
        "",
    ]
    for key in PRIORITY_ORDER:
        lines.append(f"- `{key}`: {triage['counts'].get(key, 0)}")
    lines.extend(["", "## Cases", "", "| Priority | Case | File | Reason | Report |", "| --- | --- | --- | --- | --- |"])
    for item in triage["cases"]:
        lines.append(
            f"| `{item['priority']}` | `{item['case_id']}` | `{item.get('original_filename')}` | {item.get('reason') or ''} | `{item.get('report')}` |"
        )
    _safe_write_text(report_dir / "batch_index.md", "\n".join(lines) + "\n")
    return triage


REVIEW_RECOMMENDATIONS = {
    "tracking_required",
    "tracking_unvalidated",
    "promote_to_deep_analysis",
    "needs_manual_roi",
    "content_candidate",
    "likely_artifact",
    "low_value_after_review",
}


def review_pack_dir(batch_id: str) -> Path:
    return ensure_dir(batch_report_dir(batch_id) / "review_pack")


def _review_case_dir(batch_id: str, rank: int, case_id: str) -> Path:
    return ensure_dir(review_pack_dir(batch_id) / f"{rank:02d}_{case_id}")


def _load_triage_cases(batch_id: str) -> list[dict[str, Any]]:
    triage_path = batch_report_dir(batch_id) / "batch_triage.json"
    if not triage_path.exists():
        return generate_batch_index(batch_id)["cases"]
    return read_json(triage_path)["cases"]


def _case_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in manifest["cases"]}


def _score(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high <= low:
        return 0.0
    return float(max(0.0, min(100.0, 100.0 * (value - low) / (high - low))))


def _artifact_risk(metrics: dict[str, Any], case: dict[str, Any], roi: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    sat = metrics.get("max_saturated_fraction") or 0.0
    motion = metrics.get("mean_frame_difference") or 0.0
    duration = case.get("duration_seconds") or 0.0
    width = (case.get("resolution") or {}).get("width") or 0
    height = (case.get("resolution") or {}).get("height") or 0
    if sat > 0.18:
        warnings.append("high saturation risk")
    elif sat > 0.06:
        warnings.append("some saturated highlights")
    if motion > 25:
        warnings.append("large global motion / camera movement possible")
    if duration < 6:
        warnings.append("very short clip")
    if width < 640 or height < 360:
        warnings.append("low resolution")
    if roi.get("status") == "manual_review_required" or roi.get("confidence") == "low":
        warnings.append("ROI uncertain")
    if not warnings:
        return "low", []
    if len(warnings) >= 3 or sat > 0.18:
        return "high", warnings
    return "medium", warnings


def _interactive_track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def _interactive_validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def _interactive_track_gate(case: dict[str, Any]) -> tuple[str, str]:
    case_id = case["case_id"]
    track_path = _interactive_track_path(case_id)
    if not track_path.exists():
        return "tracking_required", "Interactive tracking has not been run; automatic ROI recommendations are blocked."
    validation = _interactive_validation_path(case_id)
    if not validation.exists():
        return "tracking_unvalidated", "Interactive track exists but human validation file is missing."
    data = read_json(validation)
    if not (data.get("track_validated") and data.get("track_is_correct") and data.get("object_is_real_target")):
        return "tracking_unvalidated", "Interactive track exists but is not human validated."
    return "validated", "Interactive track is human validated."


def _roi_from_interactive_track(case_id: str) -> dict[str, Any] | None:
    path = _interactive_track_path(case_id)
    if not path.exists():
        return None
    track = read_json(path)
    boxes = [item.get("bbox") for item in track.get("track", []) if item.get("bbox")]
    if not boxes:
        return None
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[0]) + int(b[2]) for b in boxes)
    y1 = max(int(b[1]) + int(b[3]) for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "status": "interactive_track_validated", "source": str(path)}


def _roi_from_track_based_analysis(case_id: str) -> dict[str, Any] | None:
    path = DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.json"
    if not path.exists():
        return None
    data = read_json(path)
    boxes = [row.get("expanded_bbox") or row.get("bbox") for row in data.get("dynamic_rois", []) if row.get("status") in {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"} and (row.get("expanded_bbox") or row.get("bbox"))]
    if not boxes:
        return None
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[0]) + int(b[2]) for b in boxes)
    y1 = max(int(b[1]) + int(b[3]) for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "status": "track_based_analysis", "source": str(path)}


def _recommend(case: dict[str, Any], metrics: dict[str, Any], roi: dict[str, Any], artifact_risk: str) -> tuple[str, str]:
    interactive_status, interactive_reason = _interactive_track_gate(case)
    if interactive_status != "validated":
        return interactive_status, interactive_reason
    visibility = _score(metrics.get("mean_roi_contrast"), 8, 45)
    motion = _score(metrics.get("mean_frame_difference"), 1, 18)
    contrast = _score(metrics.get("mean_contrast"), 10, 55)
    duration = case.get("duration_seconds") or 0
    if roi.get("status") == "manual_review_required":
        return "needs_manual_roi", "ROI candidate is not reliable enough for automated deep analysis."
    if artifact_risk == "high":
        return "likely_artifact", "Artifact risk is high; inspect manually before spending compute."
    if visibility >= 55 and motion >= 45 and artifact_risk == "low" and duration >= 8:
        return "promote_to_deep_analysis", "Good visibility/motion balance and low artifact risk."
    if visibility >= 45 and contrast >= 45 and duration < 30:
        return "content_candidate", "Short, visually readable case suitable for quick content review."
    if visibility < 25 and motion < 20:
        return "low_value_after_review", "Weak visibility and motion in quick review pack."
    return "needs_manual_roi", "Usable, but manual ROI/target confirmation should come first."


def _select_review_frames(indices: list[int], frames: list[np.ndarray], metrics: dict[str, Any]) -> tuple[list[int], list[np.ndarray], int]:
    if not frames:
        return [], [], 0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [0.0]
    for i in range(1, len(grays)):
        diffs.append(float(cv2.absdiff(grays[i - 1], grays[i]).mean()))
    max_i = int(np.argmax(diffs))
    pick_positions = sorted(set([0, len(frames) // 2, len(frames) - 1, max_i]))
    return [indices[i] for i in pick_positions], [frames[i] for i in pick_positions], indices[max_i]


def _draw_roi(frame: np.ndarray, roi: dict[str, Any], label: str = "ROI proposal") -> np.ndarray:
    out = frame.copy()
    if roi.get("w", 0) and roi.get("h", 0):
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), max(2, out.shape[1] // 600))
        cv2.putText(out, label, (max(8, x), max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _motion_preview(frames: list[np.ndarray], out: Path) -> None:
    if len(frames) < 2:
        return
    previews = []
    labels = []
    max_pairs = min(8, len(frames) - 1)
    positions = np.linspace(1, len(frames) - 1, max_pairs, dtype=int)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for pos in positions:
        gray = cv2.cvtColor(frames[pos], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_MAGMA)
        previews.append(heat)
        labels.append(f"diff {pos}")
        prev_gray = gray
    _contact_sheet(previews, labels, out, cols=4)


def _review_panel(case: dict[str, Any], frames: list[np.ndarray], labels: list[str], roi: dict[str, Any], metrics: dict[str, Any], review: dict[str, Any], out: Path) -> None:
    width, height = 1800, 1400
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 46
    lines = [
        f"{review['rank']:02d}  {case['case_id']}",
        f"file: {case.get('original_filename')}",
        f"duration: {case.get('duration_seconds'):.2f}s   resolution: {(case.get('resolution') or {}).get('width')}x{(case.get('resolution') or {}).get('height')}   fps: {case.get('fps'):.2f}",
        f"codec: {case.get('codec')}   bitrate: {case.get('bitrate') or 'unknown'}",
        f"quick priority: {case.get('priority')}   recommendation: {review['recommended_next_step']}",
        f"visibility: {review['object_visibility_score']:.1f}   motion: {review['motion_score']:.1f}   contrast: {review['contrast_score']:.1f}   artifact risk: {review['artifact_risk']}",
    ]
    for i, line in enumerate(lines):
        color = (10, 10, 10) if i != 4 else (0, 120, 0)
        cv2.putText(panel, line[:135], (34, y), font, 0.78 if i == 0 else 0.58, color, 2 if i == 0 else 1, cv2.LINE_AA)
        y += 38
    notes = review.get("warnings") or ["no obvious quick-review warning"]
    cv2.putText(panel, ("warnings: " + "; ".join(notes))[:135], (34, y + 8), font, 0.55, (0, 80, 180), 1, cv2.LINE_AA)
    thumbs = [_resize_for_grid(_draw_roi(f, roi), 520) for f in frames[:4]]
    x0, y0 = 34, 330
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, 2)
        x = x0 + c * 600
        yy = y0 + r * 390
        panel[yy:yy + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        cv2.putText(panel, labels[i], (x + 8, yy - 10), font, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, "Human review pack: not a publication panel; high_priority is only a triage signal.", (34, height - 34), font, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)


def _write_review_notes(case: dict[str, Any], review: dict[str, Any], out: Path) -> None:
    text = f"""# Human Review Notes

Case: `{case['case_id']}`

Original file: `{case.get('original_filename')}`

Quick priority: `{case.get('priority')}`

Recommended next step: `{review['recommended_next_step']}`

Rationale: {review['recommendation_reason']}

Scores:

- object_visibility_score: `{review['object_visibility_score']:.2f}`
- motion_score: `{review['motion_score']:.2f}`
- contrast_score: `{review['contrast_score']:.2f}`
- artifact_risk: `{review['artifact_risk']}`

Warnings:

{chr(10).join('- ' + warning for warning in (review.get('warnings') or ['none']))}

Human decision:

- decision:
- reviewer:
- notes:

Important: this review pack is for quick human triage only. It does not assert origin and does not replace deep analysis.
"""
    _safe_write_text(out, text)


def _review_contact_sheet(review_items: list[dict[str, Any]], out: Path) -> None:
    thumbs = []
    labels = []
    for item in review_items:
        panel_path = Path(item["outputs"]["case_review_panel"])
        img = cv2.imread(str(panel_path))
        if img is not None:
            thumbs.append(img)
            labels.append(f"{item['rank']:02d} {item['case_id']} {item['recommended_next_step']}")
    _contact_sheet(thumbs, labels, out, cols=2)


def build_batch_review_pack(batch_id: str, top: int = 10) -> dict[str, Any]:
    manifest = load_batch(batch_id)
    cases_by_id = _case_by_id(manifest)
    triage_cases = _load_triage_cases(batch_id)[:top]
    root = review_pack_dir(batch_id)
    review_items = []
    for rank, triage_case in enumerate(triage_cases, start=1):
        case_id = triage_case["case_id"]
        case = cases_by_id[case_id]
        case_review_dir = _review_case_dir(batch_id, rank, case_id)
        quick_metrics_path = batch_case_output_dir(case_id) / "quick_metrics.json"
        quick_metrics = read_json(quick_metrics_path) if quick_metrics_path.exists() else {}
        metrics = quick_metrics.get("metrics", case.get("quick_metrics", {}))
        roi = quick_metrics.get("roi", case.get("roi_proposal", {}))
        interactive_status, _interactive_reason = _interactive_track_gate(case)
        track_based_roi = _roi_from_track_based_analysis(case_id)
        if track_based_roi:
            roi = track_based_roi
        elif interactive_status == "validated":
            roi = _roi_from_interactive_track(case_id) or roi
        indices, frames, _sample_meta = _sample_frames(Path(case["video_path"]), max_samples=24)
        selected_indices, selected_frames, max_motion_frame = _select_review_frames(indices, frames, metrics)
        selected_labels = [f"frame {idx}" for idx in selected_indices]
        object_visibility = _score(metrics.get("mean_roi_contrast"), 8, 45)
        motion_score = _score(metrics.get("mean_frame_difference"), 1, 18)
        contrast_score = _score(metrics.get("mean_contrast"), 10, 55)
        artifact_risk, warnings = _artifact_risk(metrics, case, roi)
        recommended, reason = _recommend(case, metrics, roi, artifact_risk)
        review = {
            "rank": rank,
            "case_id": case_id,
            "quick_priority": case.get("priority"),
            "duration": case.get("duration_seconds"),
            "resolution": case.get("resolution"),
            "fps": case.get("fps"),
            "codec": case.get("codec"),
            "bitrate": case.get("bitrate"),
            "object_visibility_score": object_visibility,
            "motion_score": motion_score,
            "contrast_score": contrast_score,
            "artifact_risk": artifact_risk,
            "interactive_tracking_status": interactive_status,
            "warnings": warnings,
            "recommended_next_step": recommended,
            "recommendation_reason": reason,
            "notes": "Quick human review only; high_priority is not an anomaly claim.",
            "roi_proposal": roi,
            "max_motion_frame": max_motion_frame,
        }
        case_contact = case_review_dir / "case_contact_sheet.png"
        case_motion = case_review_dir / "case_motion_preview.png"
        case_roi = case_review_dir / "case_roi_preview.png"
        case_panel = case_review_dir / "case_review_panel.png"
        case_notes = case_review_dir / "case_review_notes.md"
        case_metrics = case_review_dir / "case_review_metrics.json"
        _contact_sheet(frames, [f"f{idx}" for idx in indices], case_contact)
        _motion_preview(frames, case_motion)
        if frames:
            max_idx_pos = indices.index(max_motion_frame) if max_motion_frame in indices else min(len(frames) - 1, len(frames) // 2)
            cv2.imwrite(str(case_roi), _draw_roi(frames[max_idx_pos], roi))
        _review_panel(case, selected_frames, selected_labels, roi, metrics, review, case_panel)
        _write_review_notes(case, review, case_notes)
        review["outputs"] = {
            "case_review_panel": str(case_panel),
            "case_contact_sheet": str(case_contact),
            "case_motion_preview": str(case_motion),
            "case_roi_preview": str(case_roi),
            "case_review_notes": str(case_notes),
            "case_review_metrics": str(case_metrics),
        }
        _safe_write_json(case_metrics, review)
        review_items.append(review)
    _write_review_index(batch_id, review_items, root)
    _review_contact_sheet(review_items, root / f"top{top}_review_contact_sheet.png")
    return {
        "batch_id": batch_id,
        "top": top,
        "review_pack_dir": str(root),
        "cases_reviewed": len(review_items),
        "recommendation_counts": _recommendation_counts(review_items),
        "cases": review_items,
    }


def _recommendation_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in sorted(REVIEW_RECOMMENDATIONS)}
    for item in items:
        step = item["recommended_next_step"]
        counts[step] = counts.get(step, 0) + 1
    return counts


def _write_review_index(batch_id: str, items: list[dict[str, Any]], root: Path) -> None:
    top = len(items)
    md = root / f"top{top}_review_index.md"
    csv_path = root / f"top{top}_review_index.csv"
    decisions = root / "human_review_decisions_template.csv"
    lines = [
        f"# Top {top} Human Review Pack",
        "",
        f"Batch: `{batch_id}`",
        "",
        "Important: `high_priority` is a quick triage signal, not an anomaly/origin claim.",
        "",
        "| rank | case_id | quick_priority | duration | resolution | fps | object_visibility_score | motion_score | contrast_score | artifact_risk | recommended_next_step | notes |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in items:
        res = item.get("resolution") or {}
        resolution = f"{res.get('width')}x{res.get('height')}"
        lines.append(
            f"| {item['rank']} | `{item['case_id']}` | `{item['quick_priority']}` | {item.get('duration'):.2f} | {resolution} | {item.get('fps'):.2f} | {item['object_visibility_score']:.1f} | {item['motion_score']:.1f} | {item['contrast_score']:.1f} | `{item['artifact_risk']}` | `{item['recommended_next_step']}` | {item['notes']} |"
        )
    _safe_write_text(md, "\n".join(lines) + "\n")
    _backup(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "case_id",
            "quick_priority",
            "duration",
            "resolution",
            "fps",
            "object_visibility_score",
            "motion_score",
            "contrast_score",
            "artifact_risk",
            "recommended_next_step",
            "notes",
            "case_review_panel",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            res = item.get("resolution") or {}
            row = {key: item.get(key) for key in fieldnames}
            row["resolution"] = f"{res.get('width')}x{res.get('height')}"
            row["case_review_panel"] = item["outputs"]["case_review_panel"]
            writer.writerow(row)
    _backup(decisions)
    with decisions.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rank", "case_id", "recommended_next_step", "human_decision", "manual_roi_required", "promote", "reviewer", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "rank": item["rank"],
                    "case_id": item["case_id"],
                    "recommended_next_step": item["recommended_next_step"],
                    "human_decision": "",
                    "manual_roi_required": "",
                    "promote": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def _top_batch_cases(batch_id: str, top: int) -> list[str]:
    triage_cases = _load_triage_cases(batch_id)
    return [case["case_id"] for case in triage_cases[:top]]


def _open_video(case: dict[str, Any]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(Path(case["video_path"])))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {case['video_path']}")
    return cap


def _edge_artifact(candidate: dict[str, Any], width: int, height: int) -> bool:
    x, y, w, h = candidate["bbox"]
    margin_x = width * 0.025
    margin_y = height * 0.025
    area_frac = (w * h) / float(width * height)
    near_edge = x <= margin_x or y <= margin_y or x + w >= width - margin_x or y + h >= height - margin_y
    too_wide = w > width * 0.65 or h > height * 0.65
    too_small = w * h < max(12, width * height * 0.000015)
    too_large = area_frac > 0.18
    return bool(too_small or too_large or too_wide or (near_edge and area_frac < 0.0003))


def _detect_candidates(frame: np.ndarray, prev_gray: np.ndarray | None, subtractor: Any) -> tuple[list[dict[str, Any]], np.ndarray]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fg = subtractor.apply(blur)
    _, fg = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
    diff = np.zeros_like(fg)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, blur)
        _, diff = cv2.threshold(diff, max(8, int(np.percentile(diff, 96))), 255, cv2.THRESH_BINARY)
    bright = cv2.inRange(blur, int(np.percentile(blur, 99.4)), 255)
    # Brightness is deliberately secondary: it can help candidate edges but cannot dominate alone.
    mask = cv2.bitwise_or(fg, diff)
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(bright, diff))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    dark_threshold = int(np.percentile(blur, 7))
    dark = cv2.inRange(blur, 0, min(115, dark_threshold + 8))
    local_bg = cv2.GaussianBlur(blur, (31, 31), 0)
    dark_contrast = cv2.subtract(local_bg, blur)
    _, dark_local = cv2.threshold(dark_contrast, max(8, int(np.percentile(dark_contrast, 96))), 255, cv2.THRESH_BINARY)
    dark = cv2.bitwise_and(dark, dark_local)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour, source in [(c, "motion") for c in contours] + [(c, "dark_elongated") for c in dark_contours]:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        aspect_ratio = float(max(w, h) / max(1, min(w, h)))
        local_pad = max(8, int(max(w, h) * 0.7))
        x0, y0 = max(0, x - local_pad), max(0, y - local_pad)
        x1, y1 = min(width, x + w + local_pad), min(height, y + h + local_pad)
        local = gray[y0:y1, x0:x1]
        local_mean = float(local.mean()) if local.size else float(crop.mean())
        dark_score = max(0.0, local_mean - float(crop.mean()))
        if source == "dark_elongated":
            area_frac = (w * h) / float(width * height)
            if aspect_ratio < 2.0 or area_frac > 0.025 or area_frac < 0.00001:
                continue
        cand = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [float(x + w / 2), float(y + h / 2)],
            "area": float(w * h),
            "contour_area": area,
            "contrast": float(crop.std()),
            "brightness": float(crop.mean()),
            "source": source,
            "aspect_ratio": aspect_ratio,
            "dark_score": dark_score,
        }
        if not _edge_artifact(cand, width, height):
            candidates.append(cand)
    candidates.sort(
        key=lambda c: (
            (2.8 if c.get("source") == "dark_elongated" else 1.0)
            * (1.0 + min(2.5, c.get("dark_score", 0.0) / 18.0))
            * (1.0 + min(2.0, c.get("aspect_ratio", 1.0) / 4.0))
            * (c["contrast"] + 3)
            * math.sqrt(max(c["area"], 1))
        ),
        reverse=True,
    )
    return candidates[:24], blur


def _link_tracks(detections: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    next_id = 1
    max_gap = 8
    max_dist = max(30.0, min(width, height) * 0.10)
    for frame_item in detections:
        frame_idx = frame_item["frame"]
        assigned: set[int] = set()
        for cand in frame_item["candidates"]:
            cx, cy = cand["centroid"]
            best_track = None
            best_dist = max_dist
            for track in tracks:
                if frame_idx - track["last_frame"] > max_gap:
                    continue
                lx, ly = track["centroids"][-1]
                dist = math.hypot(cx - lx, cy - ly)
                if dist < best_dist:
                    best_dist = dist
                    best_track = track
            if best_track is None or best_track["track_id"] in assigned:
                tracks.append(
                    {
                        "track_id": f"T{next_id:03d}",
                        "frames": [frame_idx],
                        "bboxes": [cand["bbox"]],
                        "centroids": [cand["centroid"]],
                        "areas": [cand["area"]],
                        "contrasts": [cand["contrast"]],
                        "brightness": [cand["brightness"]],
                        "sources": [cand.get("source", "unknown")],
                        "aspect_ratios": [cand.get("aspect_ratio", 1.0)],
                        "dark_scores": [cand.get("dark_score", 0.0)],
                        "last_frame": frame_idx,
                    }
                )
                assigned.add(next_id)
                next_id += 1
            else:
                best_track["frames"].append(frame_idx)
                best_track["bboxes"].append(cand["bbox"])
                best_track["centroids"].append(cand["centroid"])
                best_track["areas"].append(cand["area"])
                best_track["contrasts"].append(cand["contrast"])
                best_track["brightness"].append(cand["brightness"])
                best_track["sources"].append(cand.get("source", "unknown"))
                best_track["aspect_ratios"].append(cand.get("aspect_ratio", 1.0))
                best_track["dark_scores"].append(cand.get("dark_score", 0.0))
                best_track["last_frame"] = frame_idx
                assigned.add(int(best_track["track_id"][1:]))
    return tracks


def _score_track(track: dict[str, Any], total_frames: int, fps: float, width: int, height: int) -> dict[str, Any]:
    frames = track["frames"]
    centroids = track["centroids"]
    areas = np.array(track["areas"], dtype=np.float32)
    contrasts = np.array(track["contrasts"], dtype=np.float32)
    brightness = np.array(track.get("brightness", []), dtype=np.float32)
    aspect_ratios = np.array(track.get("aspect_ratios", []), dtype=np.float32)
    dark_scores = np.array(track.get("dark_scores", []), dtype=np.float32)
    dark_source_fraction = sum(1 for s in track.get("sources", []) if s == "dark_elongated") / max(1, len(track.get("sources", [])))
    velocities = []
    for i in range(1, len(centroids)):
        dt = max(1, frames[i] - frames[i - 1])
        velocities.append(math.hypot(centroids[i][0] - centroids[i - 1][0], centroids[i][1] - centroids[i - 1][1]) / dt)
    duration_frames = frames[-1] - frames[0] + 1
    detected = len(frames)
    continuity = detected / max(1, duration_frames)
    persistence = detected / max(1, total_frames)
    motion = float(np.mean(velocities)) if velocities else 0.0
    contrast = float(np.mean(contrasts)) if contrasts.size else 0.0
    mean_brightness = float(np.mean(brightness)) if brightness.size else 0.0
    mean_aspect_ratio = float(np.mean(aspect_ratios)) if aspect_ratios.size else 1.0
    mean_dark_score = float(np.mean(dark_scores)) if dark_scores.size else 0.0
    size_stability = 1.0 / (1.0 + float(np.std(areas) / max(1.0, np.mean(areas)))) if areas.size else 0.0
    edge_hits = 0
    for bbox in track["bboxes"]:
        x, y, w, h = bbox
        if x <= width * 0.03 or y <= height * 0.03 or x + w >= width * 0.97 or y + h >= height * 0.97:
            edge_hits += 1
    artifact = min(1.0, edge_hits / max(1, detected) + (0.25 if np.mean(areas) > width * height * 0.07 else 0.0))
    dark_elongated_score = min(1.0, (mean_dark_score / 22.0) * 0.55 + (mean_aspect_ratio / 5.0) * 0.25 + dark_source_fraction * 0.35)
    brightness_penalty = 0.12 if mean_brightness > 185 and dark_source_fraction < 0.3 else 0.0
    confidence = (
        0.30 * min(1.0, continuity)
        + 0.24 * min(1.0, persistence * 8)
        + 0.13 * min(1.0, contrast / 40)
        + 0.12 * min(1.0, motion / 12)
        + 0.11 * min(1.0, size_stability)
        + 0.18 * dark_elongated_score
        - 0.28 * artifact
        - brightness_penalty
    )
    confidence = float(max(0.0, min(1.0, confidence)))
    if detected < 5:
        reason = "discarded: too few detected frames"
    elif artifact > 0.55:
        reason = "discarded: artifact/edge risk is high"
    elif confidence >= 0.55:
        reason = "selected candidate: persistent, continuous moving target"
    elif confidence >= 0.35:
        reason = "low confidence: candidate exists but needs manual track review"
    else:
        reason = "discarded: weak persistence/continuity"
    return {
        "track_id": track["track_id"],
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "duration_frames": int(duration_frames),
        "duration_seconds": float(duration_frames / fps) if fps else None,
        "number_of_detected_frames": int(detected),
        "continuity_score": float(continuity),
        "persistence_score": float(persistence),
        "motion_score": motion,
        "contrast_score": contrast,
        "size_stability_score": float(size_stability),
        "dark_elongated_score": float(dark_elongated_score),
        "mean_dark_score": mean_dark_score,
        "mean_aspect_ratio": mean_aspect_ratio,
        "dark_source_fraction": float(dark_source_fraction),
        "mean_brightness": mean_brightness,
        "artifact_risk_score": float(artifact),
        "confidence": confidence,
        "bbox_by_frame": [{"frame": int(f), "bbox": b} for f, b in zip(frames, track["bboxes"])],
        "centroid_by_frame": [{"frame": int(f), "centroid": c} for f, c in zip(frames, centroids)],
        "velocity_px_per_frame": velocities,
        "area_by_frame": [{"frame": int(f), "area": float(a)} for f, a in zip(frames, areas)],
        "reason": reason,
    }


def _track_lookup(track: dict[str, Any]) -> dict[int, list[int]]:
    return {int(item["frame"]): item["bbox"] for item in track.get("bbox_by_frame", [])}


def _track_item_lookup(track: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["frame"]): item for item in track.get("bbox_by_frame", [])}


def _nearest_bbox(track: dict[str, Any], frame_idx: int, max_gap: int = 10) -> tuple[list[int] | None, bool, dict[str, Any] | None]:
    items = _track_item_lookup(track)
    if frame_idx in items:
        item = items[frame_idx]
        low = item.get("tracking_state") == "LOW CONFIDENCE" or float(item.get("match_score", 1.0)) < 0.28
        return item["bbox"], low, item
    if track.get("manual_dark_bar_tracking"):
        return None, True, None
    boxes = {frame: item["bbox"] for frame, item in items.items()}
    frames = sorted(boxes)
    if not frames:
        return None, True, None
    nearest = min(frames, key=lambda f: abs(f - frame_idx))
    if abs(nearest - frame_idx) <= max_gap:
        return boxes[nearest], True, items.get(nearest)
    return None, True, None


def _draw_track_overlay(frame: np.ndarray, frame_idx: int, track: dict[str, Any], status: str) -> np.ndarray:
    out = frame.copy()
    bbox, lost, item = _nearest_bbox(track, frame_idx)
    manual_hint = bool(track.get("manual_target_hint_applied"))
    if manual_hint:
        color = (255, 220, 0) if not lost else (170, 170, 170)
    else:
        color = (0, 255, 0) if not lost and status == "TRACKING ACTIVE" else ((0, 220, 255) if lost else (0, 120, 255))
    trail = []
    for prev in track.get("centroid_by_frame", []):
        prev_frame = int(prev.get("frame", -1))
        if frame_idx - 45 <= prev_frame <= frame_idx:
            trail.append(tuple(int(v) for v in prev["centroid"]))
    if len(trail) > 1:
        for p0, p1 in zip(trail[:-1], trail[1:]):
            cv2.line(out, p0, p1, color, max(1, out.shape[1] // 900), cv2.LINE_AA)
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, out.shape[1] // 500))
        cv2.putText(out, f"{track.get('track_id', 'T?')}", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if manual_hint:
        score = "" if not item else f" score {float(item.get('match_score', 0.0)):.2f}"
        label = ("LOW CONFIDENCE TEMPLATE TRACK" if lost else "BLUE TARGET TEMPLATE TRACK") + score
    else:
        label = "TRACK LOST" if lost else status
    cv2.putText(out, f"frame {frame_idx}  {label}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def _write_track_overlay_video(case: dict[str, Any], track: dict[str, Any], output: Path, max_frames: int = 360) -> None:
    cap = _open_video(case)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), min(30, fps or 30), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not write tracking preview: {output}")
    manual_hint = bool(track.get("manual_target_hint_applied"))
    if total <= 0:
        indices = list(range(max_frames))
    else:
        if manual_hint:
            start = 0
            end = total - 1
        else:
            start = max(0, int(track.get("start_frame", 0)) - int(fps))
            end = min(total - 1, int(track.get("end_frame", total - 1)) + int(fps))
        count = min(max_frames, max(1, end - start + 1))
        indices = sorted(set(int(round(x)) for x in np.linspace(start, end, count)))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox, lost, _item = _nearest_bbox(track, idx)
        status = "LOW CONFIDENCE" if track.get("confidence", 0) < 0.55 else "TRACKING ACTIVE"
        if lost:
            status = "TRACK LOST"
        rendered = _draw_track_overlay(frame, idx, track, status)
        hold = 4 if manual_hint else 1
        for _ in range(hold):
            writer.write(rendered)
    writer.release()
    cap.release()


def _track_panel(case: dict[str, Any], track: dict[str, Any] | None, discarded: list[dict[str, Any]], output: Path) -> None:
    cap = _open_video(case)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if track:
        frames_to_show = sorted(set([track["start_frame"], (track["start_frame"] + track["end_frame"]) // 2, track["end_frame"]]))
        track_frames = [item["frame"] for item in track.get("bbox_by_frame", [])]
        if track_frames:
            frames_to_show.extend(np.linspace(min(track_frames), max(track_frames), min(8, len(track_frames)), dtype=int).tolist())
    else:
        frames_to_show = np.linspace(0, max(0, total - 1), 8, dtype=int).tolist() if total else []
    frames_to_show = sorted(set(int(f) for f in frames_to_show if f >= 0 and (total <= 0 or f < total)))[:12]
    thumbs = []
    labels = []
    for idx in frames_to_show:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if track:
            frame = _draw_track_overlay(frame, idx, track, "TRACKING ACTIVE")
        thumbs.append(frame)
        labels.append(f"f{idx}")
    cap.release()
    temp = output.with_name(output.stem + "_sheet_tmp.png")
    _contact_sheet(thumbs, labels, temp, cols=4)
    sheet = cv2.imread(str(temp))
    if sheet is None:
        return
    info_h = 250
    panel = np.full((sheet.shape[0] + info_h, max(sheet.shape[1], 1400), 3), 245, dtype=np.uint8)
    panel[:sheet.shape[0], :sheet.shape[1]] = sheet
    y = sheet.shape[0] + 34
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, f"{case['case_id']} tracking preview", (24, y), font, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
    y += 38
    if track:
        cv2.putText(panel, f"best track: {track['track_id']} confidence={track['confidence']:.3f} frames={track['number_of_detected_frames']} reason={track['reason'][:80]}", (24, y), font, 0.55, (0, 90, 0), 1, cv2.LINE_AA)
        y += 32
        lost = max(0, int(track["duration_frames"]) - int(track["number_of_detected_frames"]))
        cv2.putText(panel, f"lost/undetected frames inside span: {lost}; discarded candidates shown in summary CSV/JSON", (24, y), font, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "No valid track found. Manual tracking required.", (24, y), font, 0.6, (0, 0, 180), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), panel)
    try:
        temp.unlink()
    except OSError:
        pass


def _tracking_status(best: dict[str, Any] | None) -> str:
    if not best:
        return "tracking_failed"
    if best.get("confidence", 0) >= 0.55 and best.get("number_of_detected_frames", 0) >= 8 and best.get("artifact_risk_score", 1) < 0.55:
        return "tracking_candidate_ready"
    return "tracking_low_confidence"


def _tracking_hint_path(case_id: str) -> Path:
    return case_dir(case_id) / "tracking_target_hint.json"


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    w = max(8, min(w, width))
    h = max(8, min(h, height))
    x = max(0, min(width - w, x))
    y = max(0, min(height - h, y))
    return [x, y, w, h]


def _prep_template(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.equalizeHist(gray)


def _template_match_bbox(frame_gray: np.ndarray, template: np.ndarray, prev_bbox: list[int], velocity: tuple[float, float], confidence: float) -> tuple[list[int], float, str]:
    height, width = frame_gray.shape[:2]
    px, py, pw, ph = _clip_bbox(prev_bbox, width, height)
    pred_x = int(round(px + velocity[0]))
    pred_y = int(round(py + velocity[1]))
    margin_x = int(max(pw * 1.8, 90 if confidence >= 0.28 else 170))
    margin_y = int(max(ph * 4.0, 80 if confidence >= 0.28 else 150))
    sx0 = max(0, pred_x - margin_x)
    sy0 = max(0, pred_y - margin_y)
    sx1 = min(width, pred_x + pw + margin_x)
    sy1 = min(height, pred_y + ph + margin_y)
    if sx1 - sx0 < pw or sy1 - sy0 < ph:
        return _clip_bbox([pred_x, pred_y, pw, ph], width, height), 0.0, "LOW CONFIDENCE"
    search = _prep_template(frame_gray[sy0:sy1, sx0:sx1])
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    x = sx0 + int(max_loc[0])
    y = sy0 + int(max_loc[1])
    bbox = _clip_bbox([x, y, pw, ph], width, height)
    bx, by, bw, bh = bbox
    crop = frame_gray[by : by + bh, bx : bx + bw]
    band = frame_gray[max(0, by - bh) : min(height, by + 2 * bh), max(0, bx - bw) : min(width, bx + 2 * bw)]
    dark_delta = float(np.mean(band) - np.mean(crop)) if band.size and crop.size else 0.0
    score = float(max_val) * 0.75 + min(0.25, max(0.0, dark_delta) / 80.0)
    state = "TRACKING ACTIVE" if score >= 0.28 else "LOW CONFIDENCE"
    return bbox, score, state


def _dark_bar_candidates(gray: np.ndarray, hint_bbox: list[int]) -> list[dict[str, Any]]:
    height, width = gray.shape[:2]
    hx, hy, hw, hh = hint_bbox
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=13, sigmaY=13)
    diff = cv2.subtract(blur, gray)
    mask = ((diff > 18) & (gray < 95)).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < 230 or y > height - 90:
            continue
        if abs((y + h / 2) - (hy + hh / 2)) > 120:
            continue
        if w < 28 or w > 260 or h < 3 or h > 36:
            continue
        aspect = w / max(1, h)
        if aspect < 4.0:
            continue
        crop = gray[y : y + h, x : x + w]
        band = gray[max(0, y - 4 * h) : min(height, y + 5 * h), max(0, x - w) : min(width, x + 2 * w)]
        dark_delta = float(np.mean(band) - np.mean(crop)) if band.size and crop.size else 0.0
        mean_brightness = float(np.mean(crop)) if crop.size else 255.0
        if dark_delta < 5:
            continue
        if x > width * 0.86 and w > width * 0.09 and mean_brightness < 18:
            continue
        score = dark_delta * aspect * math.sqrt(max(1, w * h))
        candidates.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": [float(x + w / 2), float(y + h / 2)],
                "score": float(score),
                "dark_delta": dark_delta,
                "mean_brightness": mean_brightness,
                "aspect_ratio": float(aspect),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _sequence_dark_bar_track(case: dict[str, Any], hint: dict[str, Any], fps: float, total: int, width: int, height: int) -> dict[str, Any] | None:
    hint_frame = int(hint["frame"])
    hint_bbox = _clip_bbox([int(v) for v in hint["bbox"]], width, height)
    cap = _open_video(case)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in range(total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        by_frame[frame_idx] = _dark_bar_candidates(gray, hint_bbox)
    cap.release()
    seed_candidates = by_frame.get(hint_frame, [])
    if not seed_candidates:
        return None
    seed = max(
        seed_candidates,
        key=lambda cand: _bbox_iou(cand["bbox"], hint_bbox)
        + 0.2 / (1.0 + abs((cand["bbox"][1] + cand["bbox"][3] / 2) - (hint_bbox[1] + hint_bbox[3] / 2)) / 30.0)
        + 0.000001 * cand["score"],
    )

    def choose_neighbor(candidates: list[dict[str, Any]], current: dict[str, Any], direction: int) -> dict[str, Any] | None:
        cx, cy, cw, ch = current["bbox"]
        ccx = cx + cw / 2
        ccy = cy + ch / 2
        scored = []
        for cand in candidates:
            x, y, w, h = cand["bbox"]
            center_x = x + w / 2
            center_y = y + h / 2
            dx = (center_x - ccx) * direction
            dy = abs(center_y - ccy)
            if dx < 35 or dx > 190:
                continue
            if dy > 75:
                continue
            size_ratio = min(w / max(1, cw), cw / max(1, w))
            if size_ratio < 0.35:
                continue
            score = cand["score"] + 80.0 * size_ratio - 2.0 * abs(dx - 122.0) - 3.0 * dy
            scored.append((score, cand))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    track: dict[int, dict[str, Any]] = {hint_frame: seed}
    current = seed
    for frame_idx in range(hint_frame - 1, -1, -1):
        chosen = choose_neighbor(by_frame.get(frame_idx, []), current, direction=-1)
        if chosen is None:
            break
        track[frame_idx] = chosen
        current = chosen
    current = seed
    for frame_idx in range(hint_frame + 1, total):
        chosen = choose_neighbor(by_frame.get(frame_idx, []), current, direction=1)
        if chosen is None:
            break
        track[frame_idx] = chosen
        current = chosen

    ordered = []
    centroids = []
    areas = []
    velocities = []
    previous = None
    for frame_idx in sorted(track):
        cand = track[frame_idx]
        bbox = cand["bbox"]
        bx, by, bw, bh = bbox
        centroid = [float(bx + bw / 2), float(by + bh / 2)]
        ordered.append(
            {
                "frame": int(frame_idx),
                "bbox": bbox,
                "match_score": float(min(1.0, cand["dark_delta"] / 45.0)),
                "tracking_state": "TRACKING ACTIVE",
                "dark_delta": float(cand["dark_delta"]),
                "source": "dark_bar_sequence",
            }
        )
        centroids.append({"frame": int(frame_idx), "centroid": centroid})
        areas.append({"frame": int(frame_idx), "area": float(bw * bh)})
        if previous is not None:
            dt = max(1, frame_idx - previous["frame"])
            pc = previous["centroid"]
            velocities.append(math.hypot(centroid[0] - pc[0], centroid[1] - pc[1]) / dt)
        previous = {"frame": frame_idx, "centroid": centroid}
    if len(ordered) < 3:
        return None
    span = ordered[-1]["frame"] - ordered[0]["frame"] + 1
    continuity = len(ordered) / max(1, span)
    confidence = min(0.95, 0.35 + 0.35 * continuity + 0.20 * min(1.0, len(ordered) / 12.0))
    return {
        "track_id": hint.get("track_id", "T084"),
        "start_frame": int(ordered[0]["frame"]),
        "end_frame": int(ordered[-1]["frame"]),
        "duration_frames": int(span),
        "duration_seconds": float(span / fps) if fps else None,
        "number_of_detected_frames": int(len(ordered)),
        "continuity_score": float(continuity),
        "persistence_score": float(len(ordered) / max(1, total)),
        "motion_score": float(np.mean(velocities)) if velocities else 0.0,
        "contrast_score": float(np.mean([item.get("dark_delta", 0.0) for item in ordered])),
        "size_stability_score": 1.0,
        "dark_elongated_score": 1.0,
        "mean_dark_score": float(np.mean([item.get("dark_delta", 0.0) for item in ordered])),
        "mean_aspect_ratio": float(np.mean([track[item["frame"]]["aspect_ratio"] for item in ordered])),
        "dark_source_fraction": 1.0,
        "mean_brightness": float(np.mean([track[item["frame"]]["mean_brightness"] for item in ordered])),
        "artifact_risk_score": 0.0,
        "confidence": float(confidence),
        "bbox_by_frame": ordered,
        "centroid_by_frame": centroids,
        "velocity_px_per_frame": velocities,
        "area_by_frame": areas,
        "manual_target_hint_applied": True,
        "manual_dark_bar_tracking": True,
        "hint_frame": hint_frame,
        "hint_bbox": hint_bbox,
        "reason": "manual user target tracked as dark elongated FLIR bar from first left-margin appearance to exit",
    }


def _manual_template_track(case: dict[str, Any], hint: dict[str, Any], fps: float, total: int, width: int, height: int) -> dict[str, Any] | None:
    cap = _open_video(case)
    hint_frame = int(hint["frame"])
    hint_bbox = _clip_bbox([int(v) for v in hint["bbox"]], width, height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, hint_frame)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = hint_bbox
    template = _prep_template(gray[y : y + h, x : x + w])
    if template.size == 0:
        cap.release()
        return None

    def walk(indices: list[int], start_bbox: list[int]) -> dict[int, dict[str, Any]]:
        items: dict[int, dict[str, Any]] = {}
        prev_bbox = start_bbox
        prev_center = (start_bbox[0] + start_bbox[2] / 2.0, start_bbox[1] + start_bbox[3] / 2.0)
        velocity = (0.0, 0.0)
        confidence = 1.0
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok_frame, current = cap.read()
            if not ok_frame:
                continue
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            bbox, score, state = _template_match_bbox(current_gray, template, prev_bbox, velocity, confidence)
            center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
            if score >= 0.24:
                velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
                prev_bbox = bbox
                prev_center = center
                confidence = score
            else:
                predicted = _clip_bbox([int(round(prev_bbox[0] + velocity[0])), int(round(prev_bbox[1] + velocity[1])), prev_bbox[2], prev_bbox[3]], width, height)
                bbox = predicted
                center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
                prev_bbox = bbox
                prev_center = center
                confidence = score
            items[frame_idx] = {"frame": int(frame_idx), "bbox": bbox, "match_score": float(score), "tracking_state": state}
        return items

    forward = walk(list(range(hint_frame + 1, total)), hint_bbox)
    backward = walk(list(range(hint_frame - 1, -1, -1)), hint_bbox)
    cap.release()
    hint_item = {"frame": hint_frame, "bbox": hint_bbox, "match_score": 1.0, "tracking_state": "TRACKING ACTIVE"}
    items = {hint_frame: hint_item, **forward, **backward}
    ordered_items = [items[idx] for idx in sorted(items)]
    centroids = []
    areas = []
    scores = []
    velocities = []
    previous = None
    for item in ordered_items:
        bx, by, bw, bh = item["bbox"]
        centroid = [float(bx + bw / 2.0), float(by + bh / 2.0)]
        centroids.append({"frame": int(item["frame"]), "centroid": centroid})
        areas.append({"frame": int(item["frame"]), "area": float(bw * bh)})
        scores.append(float(item.get("match_score", 0.0)))
        if previous is not None:
            dt = max(1, int(item["frame"]) - int(previous["frame"]))
            pc = previous["centroid"]
            velocities.append(math.hypot(centroid[0] - pc[0], centroid[1] - pc[1]) / dt)
        previous = {"frame": item["frame"], "centroid": centroid}
    active = sum(1 for item in ordered_items if item.get("tracking_state") == "TRACKING ACTIVE")
    continuity = active / max(1, len(ordered_items))
    confidence = float(np.mean(scores)) if scores else 0.0
    return {
        "track_id": hint.get("track_id", "T084"),
        "start_frame": 0 if total else hint_frame,
        "end_frame": max(0, total - 1),
        "duration_frames": int(total),
        "duration_seconds": float(total / fps) if fps else None,
        "number_of_detected_frames": int(active),
        "continuity_score": float(continuity),
        "persistence_score": float(continuity),
        "motion_score": float(np.mean(velocities)) if velocities else 0.0,
        "contrast_score": 0.0,
        "size_stability_score": 1.0,
        "dark_elongated_score": 1.0,
        "mean_dark_score": 0.0,
        "mean_aspect_ratio": hint_bbox[2] / max(1, hint_bbox[3]),
        "dark_source_fraction": 1.0,
        "mean_brightness": 0.0,
        "artifact_risk_score": 0.0,
        "confidence": confidence,
        "bbox_by_frame": ordered_items,
        "centroid_by_frame": centroids,
        "velocity_px_per_frame": velocities,
        "area_by_frame": areas,
        "manual_target_hint_applied": True,
        "manual_template_tracking": True,
        "hint_frame": hint_frame,
        "hint_bbox": hint_bbox,
        "reason": "manual blue/cyan target hint expanded into full-frame template track; requires visual validation",
    }


def _apply_tracking_hint(case_id: str, summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = _tracking_hint_path(case_id)
    if not path.exists():
        return summaries, None
    hint = read_json(path)
    hint_frame = int(hint["frame"])
    hint_bbox = [int(v) for v in hint["bbox"]]
    scored = []
    for summary in summaries:
        best = 0.0
        best_gap = 999999
        for item in summary.get("bbox_by_frame", []):
            frame = int(item["frame"])
            gap = abs(frame - hint_frame)
            if gap > 12:
                continue
            iou = _bbox_iou(item["bbox"], hint_bbox)
            cx = item["bbox"][0] + item["bbox"][2] / 2
            cy = item["bbox"][1] + item["bbox"][3] / 2
            hx = hint_bbox[0] + hint_bbox[2] / 2
            hy = hint_bbox[1] + hint_bbox[3] / 2
            center_score = 1.0 / (1.0 + math.hypot(cx - hx, cy - hy) / 80.0)
            score = max(iou, center_score * 0.65) - gap * 0.01
            if score > best:
                best = score
                best_gap = gap
        if best > 0:
            scored.append((best, best_gap, summary))
    if not scored:
        manual_summary = {
            "track_id": hint.get("track_id", "MANUAL_HINT"),
            "start_frame": hint_frame,
            "end_frame": hint_frame,
            "duration_frames": 1,
            "duration_seconds": None,
            "number_of_detected_frames": 1,
            "continuity_score": 1.0,
            "persistence_score": 0.0,
            "motion_score": 0.0,
            "contrast_score": 0.0,
            "size_stability_score": 0.0,
            "dark_elongated_score": 1.0,
            "mean_dark_score": 0.0,
            "mean_aspect_ratio": hint_bbox[2] / max(1, hint_bbox[3]),
            "dark_source_fraction": 1.0,
            "mean_brightness": 0.0,
            "artifact_risk_score": 0.0,
            "confidence": 0.25,
            "bbox_by_frame": [{"frame": hint_frame, "bbox": hint_bbox}],
            "centroid_by_frame": [{"frame": hint_frame, "centroid": [hint_bbox[0] + hint_bbox[2] / 2, hint_bbox[1] + hint_bbox[3] / 2]}],
            "velocity_px_per_frame": [],
            "area_by_frame": [{"frame": hint_frame, "area": float(hint_bbox[2] * hint_bbox[3])}],
            "reason": "manual visual target hint; requires manual temporal tracking",
        }
        return [manual_summary, *summaries], hint
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen = scored[0][2]
    chosen["manual_target_hint_applied"] = True
    chosen["hint_frame"] = hint_frame
    chosen["hint_bbox"] = hint_bbox
    chosen["reason"] = "manual visual target hint matched this track; requires manual validation"
    reordered = [chosen] + [summary for summary in summaries if summary["track_id"] != chosen["track_id"]]
    return reordered, hint


def track_case_objects(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cap = _open_video(case)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or case.get("fps") or 30)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (case.get("resolution") or {}).get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (case.get("resolution") or {}).get("height") or 0)
    scale = 480 / width if width > 480 else 1.0
    proc_w = int(width * scale)
    proc_h = int(height * scale)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=22, detectShadows=False)
    detections: list[dict[str, Any]] = []
    prev_gray = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
        candidates, prev_gray = _detect_candidates(small, prev_gray, subtractor)
        if scale != 1.0:
            inv = 1.0 / scale
            for cand in candidates:
                x, y, w, h = cand["bbox"]
                cand["bbox"] = [int(x * inv), int(y * inv), int(w * inv), int(h * inv)]
                cx, cy = cand["centroid"]
                cand["centroid"] = [float(cx * inv), float(cy * inv)]
                cand["area"] = float(cand["bbox"][2] * cand["bbox"][3])
        detections.append({"frame": frame_idx, "candidates": candidates})
        frame_idx += 1
    cap.release()
    tracks_raw = _link_tracks(detections, width, height)
    summaries = [_score_track(track, max(1, frame_idx), fps, width, height) for track in tracks_raw]
    summaries.sort(key=lambda s: s["confidence"], reverse=True)
    summaries, hint = _apply_tracking_hint(case["case_id"], summaries)
    if hint is not None:
        manual_track = _sequence_dark_bar_track(case, hint, fps, frame_idx, width, height)
        if manual_track is None:
            manual_track = _manual_template_track(case, hint, fps, frame_idx, width, height)
        if manual_track is not None:
            summaries = [manual_track] + [summary for summary in summaries if summary.get("track_id") != manual_track["track_id"]]
    best = summaries[0] if summaries else None
    status = _tracking_status(best)
    if hint is not None:
        status = "tracking_low_confidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case["case_id"],
        "tracking_status": status,
        "video_path": case["video_path"],
        "frame_count_processed": frame_idx,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "strategies": [
            "frame differencing",
            "background subtraction",
            "contour detection",
            "small moving object detection",
            "brightness as secondary signal",
            "temporal linking",
            "edge/overlay artifact filtering",
            "manual blue target template tracking when a human target hint exists",
        ],
        "primary_track_id": best["track_id"] if best else None,
        "manual_target_hint": hint,
        "tracks": summaries,
    }
    _safe_write_json(output_dir / "object_tracks.json", payload)
    _backup(output_dir / "track_summary.csv")
    with (output_dir / "track_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "track_id",
            "start_frame",
            "end_frame",
            "duration_frames",
            "number_of_detected_frames",
            "continuity_score",
            "persistence_score",
            "motion_score",
            "contrast_score",
            "size_stability_score",
            "dark_elongated_score",
            "mean_dark_score",
            "mean_aspect_ratio",
            "dark_source_fraction",
            "mean_brightness",
            "artifact_risk_score",
            "confidence",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})
    _track_panel(case, best, summaries[1:8], output_dir / "track_preview_panel.png")
    if best:
        _write_track_overlay_video(case, best, output_dir / "track_overlay_preview.mp4")
    diag = [
        f"# Tracking Diagnostics: {case['case_id']}",
        "",
        f"Status: `{status}`",
        f"Frames processed: `{frame_idx}`",
        f"Best track: `{best['track_id'] if best else 'none'}`",
        f"Confidence: `{best['confidence']:.3f}`" if best else "Confidence: `0.000`",
        "",
        "Warnings:",
        "",
    ]
    if hint is not None:
        diag.append("- Manual/user visual target hint applied; automated tracking is not considered validated.")
    if best and best.get("artifact_risk_score", 0) > 0.35:
        diag.append("- Artifact/edge risk is non-trivial.")
    if best and best.get("continuity_score", 0) < 0.45:
        diag.append("- Track continuity is low; manual review required.")
    if not best:
        diag.append("- No track passed minimum detection criteria.")
    if len(diag) <= 9:
        diag.append("- No major automated tracking warning.")
    diag.extend(["", "This is tracking validation only. It does not assert origin."])
    _safe_write_text(output_dir / "tracking_diagnostics.md", "\n".join(diag) + "\n")
    return payload


def batch_track_objects(batch_id: str, top: int = 10) -> dict[str, Any]:
    manifest = load_batch(batch_id)
    case_ids = _top_batch_cases(batch_id, top)
    cases = _case_by_id(manifest)
    review_root = ensure_dir(batch_report_dir(batch_id) / "tracking_review")
    results = []
    for case_id in case_ids:
        case = cases[case_id]
        out_dir = ensure_dir(DATA_DIR / "outputs" / case_id / "tracking")
        try:
            result = track_case_objects(case, out_dir)
            case["tracking_status"] = result["tracking_status"]
            case["tracking_outputs"] = {
                "object_tracks": str(out_dir / "object_tracks.json"),
                "track_summary": str(out_dir / "track_summary.csv"),
                "track_preview_panel": str(out_dir / "track_preview_panel.png"),
                "track_overlay_preview": str(out_dir / "track_overlay_preview.mp4"),
                "tracking_diagnostics": str(out_dir / "tracking_diagnostics.md"),
            }
            case["primary_track_id"] = result.get("primary_track_id")
            case["tracking_confidence"] = (result["tracks"][0]["confidence"] if result.get("tracks") else 0.0)
            results.append({"case_id": case_id, "status": case["tracking_status"], "primary_track_id": case.get("primary_track_id"), "confidence": case.get("tracking_confidence"), "outputs": case["tracking_outputs"]})
        except Exception as exc:  # noqa: BLE001
            case["tracking_status"] = "tracking_failed"
            case["tracking_error"] = str(exc)
            results.append({"case_id": case_id, "status": "tracking_failed", "error": str(exc)})
    save_batch(manifest)
    _write_tracking_review_index(batch_id, results, review_root)
    _mark_review_pack_superseded(batch_id)
    return {"batch_id": batch_id, "top": top, "tracking_review_dir": str(review_root), "results": results, "counts": _tracking_counts(results)}


def _tracking_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def _write_tracking_review_index(batch_id: str, results: list[dict[str, Any]], root: Path) -> None:
    decisions = root / "human_tracking_decisions.csv"
    _backup(decisions)
    with decisions.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["case_id", "selected_track_id", "tracking_correct", "object_is_real_target", "needs_manual_track", "reject_tracking", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({"case_id": item["case_id"], "selected_track_id": item.get("primary_track_id") or "", "tracking_correct": "", "object_is_real_target": "", "needs_manual_track": "", "reject_tracking": "", "notes": ""})
    lines = [
        "# Top 10 Tracking Review Index",
        "",
        f"Batch: `{batch_id}`",
        "",
        "Review pack recommendations are blocked until tracking is candidate-ready or human validated.",
        "",
        "| case | best track | confidence | tracking reliable | pass to review pack | requires manual track | blocked | preview |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        status = item["status"]
        conf = float(item.get("confidence") or 0)
        reliable = "yes" if status == "tracking_candidate_ready" else "no"
        pass_review = "yes" if status == "tracking_candidate_ready" else "no"
        manual = "yes" if status in {"tracking_low_confidence", "tracking_failed"} else "no"
        blocked = "no" if status == "tracking_candidate_ready" else "yes"
        preview = (item.get("outputs") or {}).get("track_overlay_preview", "")
        lines.append(f"| `{item['case_id']}` | `{item.get('primary_track_id') or ''}` | {conf:.3f} | {reliable} | {pass_review} | {manual} | {blocked} | `{preview}` |")
    _safe_write_text(root / "top10_tracking_index.md", "\n".join(lines) + "\n")


def _mark_review_pack_superseded(batch_id: str) -> None:
    root = review_pack_dir(batch_id)
    if not root.exists():
        return
    _safe_write_text(
        root / "SUPERSEDED_BY_TRACKING_REQUIRED.md",
        "# Superseded By Tracking Required\n\nThe previous review pack was generated before temporal object tracking was available. Its recommendations are limited and must not be used for deep-analysis promotion until a valid track or human tracking decision exists.\n",
    )
