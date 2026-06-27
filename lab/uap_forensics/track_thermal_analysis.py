from __future__ import annotations

import csv
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


VALID_STATES = {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"}
REQUIRED_MESSAGE = "Thermal / IR analysis requires a human-validated track and dynamic ROIs."


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def dynamic_rois_csv(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "thermal_analysis")


def case_status_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "case_status.json"


def _require_valid_inputs(case_id: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not track_path(case_id).exists() or not validation_path(case_id).exists() or not dynamic_rois_csv(case_id).exists():
        raise RuntimeError(REQUIRED_MESSAGE)
    track = read_json(track_path(case_id))
    validation = read_json(validation_path(case_id))
    if not (validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target")):
        raise RuntimeError(REQUIRED_MESSAGE)
    rows = _read_dynamic_rois(dynamic_rois_csv(case_id))
    if not rows:
        raise RuntimeError(REQUIRED_MESSAGE)
    return track, validation, rows


def _read_dynamic_rois(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            def i(name: str) -> int | None:
                value = raw.get(name, "")
                return int(float(value)) if value not in {"", None} else None

            bbox = [i("expanded_x"), i("expanded_y"), i("expanded_w"), i("expanded_h")]
            rows.append(
                {
                    "frame": int(raw["frame"]),
                    "status": raw.get("status", ""),
                    "bbox": bbox if all(v is not None for v in bbox) else None,
                }
            )
    return sorted(rows, key=lambda row: row["frame"])


def _open_video(track: dict[str, Any]) -> tuple[cv2.VideoCapture, float, int, int, int]:
    video_path = track.get("video", {}).get("path")
    if not video_path:
        raise RuntimeError("track.json does not include video.path")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or track.get("video", {}).get("fps") or 30)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or track.get("video", {}).get("frame_count") or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or track.get("video", {}).get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or track.get("video", {}).get("height") or 0)
    return cap, fps, total, width, height


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _source_ir_mode(track: dict[str, Any], first_frame: np.ndarray | None) -> str:
    text = " ".join(str(value).lower() for value in [track.get("case_id"), track.get("video", {}).get("path"), track.get("source_type"), track.get("original_filename")])
    hints = ["flir", "ir", "thermal", "infrared", "dod_", "aerial", "uap", "navy"]
    if any(hint in text for hint in hints):
        return "true_ir_unknown_calibration"
    if first_frame is not None:
        b, g, r = cv2.split(first_frame)
        color_delta = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) + np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))
        if color_delta < 4.0:
            return "true_ir_unknown_calibration"
    return "pseudo_intensity_only"


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _background_patch(gray: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = max(12, int(w * 0.75))
    pad_y = max(12, int(h * 0.75))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(gray.shape[1], x + w + pad_x)
    y1 = min(gray.shape[0], y + h + pad_y)
    patch = gray[y0:y1, x0:x1].copy()
    ix0, iy0 = x - x0, y - y0
    patch[iy0 : iy0 + h, ix0 : ix0 + w] = 0
    mask = np.ones(patch.shape, dtype=np.uint8)
    mask[iy0 : iy0 + h, ix0 : ix0 + w] = 0
    values = patch[mask > 0]
    return values if values.size else gray.reshape(-1)


def _pseudo_color(gray: np.ndarray, size: int = 160) -> np.ndarray:
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.resize(color, (size, size), interpolation=cv2.INTER_NEAREST)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_sheet(images: list[np.ndarray], path: Path, cols: int = 4) -> None:
    if not images:
        return
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    cv2.imwrite(str(path), sheet)


def _collect_series(track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    cap, fps, total, width, height = _open_video(track)
    active = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox")]
    first_frame = _read_frame(cap, active[0]["frame"]) if active else None
    mode = _source_ir_mode(track, first_frame)
    series: list[dict[str, Any]] = []
    contact_tiles: list[np.ndarray] = []
    example_tiles: list[np.ndarray] = []
    delta_tiles: list[np.ndarray] = []
    pick_every = max(1, len(active) // 12) if active else 1
    for pos, row in enumerate(active):
        frame_idx = int(row["frame"])
        frame = _read_frame(cap, frame_idx)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = _clamp_bbox(row["bbox"], width, height)
        roi = gray[y : y + h, x : x + w]
        bg = _background_patch(gray, [x, y, w, h])
        bg_mean = float(bg.mean())
        bg_std = float(bg.std())
        roi_mean = float(roi.mean())
        roi_std = float(roi.std())
        hot_thr = bg_mean + max(8.0, bg_std)
        cold_thr = bg_mean - max(8.0, bg_std)
        hot_ratio = float(np.mean(roi > hot_thr))
        cold_ratio = float(np.mean(roi < cold_thr))
        delta = roi_mean - bg_mean
        series.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps if fps else 0.0,
                "roi_intensity_mean": roi_mean,
                "roi_intensity_std": roi_std,
                "roi_intensity_min": float(roi.min()),
                "roi_intensity_max": float(roi.max()),
                "background_intensity_mean": bg_mean,
                "roi_background_delta": delta,
                "hot_pixel_ratio": hot_ratio,
                "cold_pixel_ratio": cold_ratio,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "track_status": row["status"],
            }
        )
        if pos % pick_every == 0 and len(contact_tiles) < 12:
            color = _pseudo_color(roi)
            contact_tiles.append(_label(color, f"f{frame_idx} I={roi_mean:.1f} d={delta:.1f}"))
        if pos % pick_every == 0 and len(example_tiles) < 4:
            raw = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), (180, 180), interpolation=cv2.INTER_NEAREST)
            norm = cv2.resize(cv2.cvtColor(cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR), (180, 180), interpolation=cv2.INTER_NEAREST)
            pseudo = _pseudo_color(roi, 180)
            bg_tile = np.full((180, 180, 3), int(max(0, min(255, bg_mean))), dtype=np.uint8)
            example_tiles.extend([_label(raw, "IR raw"), _label(norm, "normalized"), _label(pseudo, "pseudo-color"), _label(bg_tile, "local bg")])
        if pos % pick_every == 0 and len(delta_tiles) < 8:
            delta_img = np.clip(roi.astype(np.float32) - bg_mean + 128, 0, 255).astype(np.uint8)
            delta_tiles.append(_label(_pseudo_color(delta_img), f"f{frame_idx} delta"))
    cap.release()
    metrics = _metrics(track, mode, fps, total, series)
    return series, metrics, contact_tiles, example_tiles, delta_tiles


def _metrics(track: dict[str, Any], mode: str, fps: float, total: int, series: list[dict[str, Any]]) -> dict[str, Any]:
    roi = np.array([row["roi_intensity_mean"] for row in series], dtype=np.float32)
    bg = np.array([row["background_intensity_mean"] for row in series], dtype=np.float32)
    delta = np.array([row["roi_background_delta"] for row in series], dtype=np.float32)
    hot = np.array([row["hot_pixel_ratio"] for row in series], dtype=np.float32)
    cold = np.array([row["cold_pixel_ratio"] for row in series], dtype=np.float32)
    duration = (series[-1]["timestamp"] - series[0]["timestamp"]) if len(series) > 1 else 0.0
    stability = float(1.0 / (1.0 + (float(np.std(delta)) / max(1e-6, abs(float(np.mean(delta))))))) if len(delta) else 0.0
    contrast_index = float(np.mean(np.abs(delta)) / max(1.0, float(np.mean(bg)))) if len(bg) else 0.0
    peak_frames = []
    if len(delta):
        threshold = float(np.percentile(np.abs(delta), 95))
        peak_frames = [int(row["frame"]) for row in series if abs(row["roi_background_delta"]) >= threshold][:20]
    return {
        "case_id": track.get("case_id"),
        "source_ir_mode": mode,
        "calibration_available": False,
        "temperature_units_available": False,
        "total_frames": int(total or track.get("summary", {}).get("total_frames") or len(series)),
        "valid_tracked_frames": len(series),
        "fps": fps,
        "duration_analyzed": duration,
        "mean_roi_intensity": float(np.mean(roi)) if len(roi) else 0.0,
        "std_roi_intensity": float(np.std(roi)) if len(roi) else 0.0,
        "min_roi_intensity": float(np.min(roi)) if len(roi) else 0.0,
        "max_roi_intensity": float(np.max(roi)) if len(roi) else 0.0,
        "mean_background_intensity": float(np.mean(bg)) if len(bg) else 0.0,
        "roi_background_delta_mean": float(np.mean(delta)) if len(delta) else 0.0,
        "roi_background_delta_std": float(np.std(delta)) if len(delta) else 0.0,
        "hot_pixel_ratio": float(np.mean(hot)) if len(hot) else 0.0,
        "cold_pixel_ratio": float(np.mean(cold)) if len(cold) else 0.0,
        "thermal_contrast_index": contrast_index,
        "intensity_stability_score": stability,
        "intensity_peak_frames": peak_frames,
        "notes": [
            "FLIR/IR relative intensity analysis - no radiometric temperature units available.",
            "No Celsius/Fahrenheit temperatures are computed.",
            "Results depend on auto-gain, compression, sensor, zoom, tracking and local background.",
        ],
    }


def _write_timeseries(path: Path, series: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "roi_intensity_mean",
        "roi_intensity_std",
        "roi_intensity_min",
        "roi_intensity_max",
        "background_intensity_mean",
        "roi_background_delta",
        "hot_pixel_ratio",
        "cold_pixel_ratio",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "track_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(series)


def _intensity_panel(path: Path, series: list[dict[str, Any]]) -> None:
    frames = np.array([row["frame"] for row in series])
    roi = np.array([row["roi_intensity_mean"] for row in series])
    bg = np.array([row["background_intensity_mean"] for row in series])
    delta = np.array([row["roi_background_delta"] for row in series])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, roi, label="ROI intensity", color="#e15759")
    axes[0].plot(frames, bg, label="local background", color="#4e79a7")
    axes[0].legend()
    axes[0].set_title("Relative IR intensity: ROI vs local background")
    axes[1].plot(frames, delta, color="#59a14f")
    if len(delta):
        peaks = np.abs(delta) >= np.percentile(np.abs(delta), 95)
        axes[1].scatter(frames[peaks], delta[peaks], color="red", s=15, label="top 5% delta")
        axes[1].legend()
    axes[1].set_title("ROI-background delta")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _contrast_panel(path: Path, series: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    frames = np.array([row["frame"] for row in series])
    hot = np.array([row["hot_pixel_ratio"] for row in series])
    cold = np.array([row["cold_pixel_ratio"] for row in series])
    delta = np.array([row["roi_background_delta"] for row in series])
    rel = np.abs(delta) / max(1.0, abs(float(metrics.get("mean_background_intensity") or 1.0)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(frames, rel, color="#edc948")
    axes[0].set_title(f"Thermal contrast index mean: {metrics['thermal_contrast_index']:.4f}")
    axes[1].plot(frames, hot, color="#e15759", label="hot")
    axes[1].plot(frames, cold, color="#4e79a7", label="cold")
    axes[1].legend()
    axes[1].set_title("Hot/cold pixel ratio")
    axes[2].plot(frames, np.abs(delta), color="#76b7b2")
    axes[2].set_title(f"Relative intensity stability score: {metrics['intensity_stability_score']:.4f}")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _delta_panel(path: Path, tiles: list[np.ndarray]) -> None:
    _make_sheet(tiles, path, cols=4)


def _write_video(case_id: str, series: list[dict[str, Any]], track: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    out = output_dir(case_id)
    raw = out / "thermal_roi_sequence_raw.mp4"
    final = out / "thermal_roi_sequence.mp4"
    cap, _fps, total, width, height = _open_video(track)
    by_frame = {
        int(item["frame"]): {
            "bbox": [int(item["bbox_x"]), int(item["bbox_y"]), int(item["bbox_w"]), int(item["bbox_h"])]
        }
        for item in series
    }
    wanted = set(by_frame)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 160))
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in wanted:
            continue
        x, y, w, h = _clamp_bbox(by_frame[frame_idx]["bbox"], width, height)
        gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        writer.write(_pseudo_color(gray))
    cap.release()
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(final)]
        subprocess.run(command, cwd=str(out), capture_output=True, text=True)
        if final.exists() and final.stat().st_size > 0:
            raw.unlink(missing_ok=True)
            return final
    return raw


def _write_report(case_id: str, track: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report = case_report_dir(case_id) / "thermal_analysis_report.md"
    lines = [
        f"# Thermal / IR Track-Based Report: {case_id}",
        "",
        "## Scope",
        "",
        "This is relative FLIR/IR intensity analysis inside the human-validated dynamic tracking ROI.",
        "",
        "## Calibration",
        "",
        f"- source_ir_mode: `{metrics['source_ir_mode']}`",
        f"- calibration_available: `{metrics['calibration_available']}`",
        f"- temperature_units_available: `{metrics['temperature_units_available']}`",
        "- No Celsius/Fahrenheit temperature is computed.",
        "",
        "## Source Track",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Metrics",
        "",
        f"- valid tracked frames: `{metrics['valid_tracked_frames']}`",
        f"- fps: `{metrics['fps']:.4f}`",
        f"- duration analyzed: `{metrics['duration_analyzed']:.4f}`",
        f"- mean ROI intensity: `{metrics['mean_roi_intensity']:.4f}`",
        f"- mean background intensity: `{metrics['mean_background_intensity']:.4f}`",
        f"- ROI-background delta mean: `{metrics['roi_background_delta_mean']:.4f}`",
        f"- thermal contrast index: `{metrics['thermal_contrast_index']:.6f}`",
        f"- intensity stability score: `{metrics['intensity_stability_score']:.6f}`",
        "",
        "## Limitations",
        "",
        "- This is not radiometric temperature.",
        "- Results depend on auto-gain, compression, sensor response, zoom, tracking and local background.",
        "- No origin/nature claim is made.",
        "- No ROI automation, redetection, autoencoder or generative model was used.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "thermal_manifest.md"
    lines = ["# Thermal / IR Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in outputs.items():
        cls = "public_safe" if name.endswith(".png") or name.endswith(".md") else "technical"
        lines.append(f"| `{name}` | `{path}` | `{cls}` | relative IR intensity only; no radiometric temperature |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_thermal_analysis(case_id: str) -> dict[str, Any]:
    track, validation, rows = _require_valid_inputs(case_id)
    out = output_dir(case_id)
    series, metrics, contact, examples, delta_tiles = _collect_series(track, rows)
    outputs: dict[str, Path] = {
        "thermal_metrics.json": out / "thermal_metrics.json",
        "thermal_timeseries.csv": out / "thermal_timeseries.csv",
        "thermal_intensity_panel.png": out / "thermal_intensity_panel.png",
        "thermal_contrast_panel.png": out / "thermal_contrast_panel.png",
        "thermal_roi_examples_panel.png": out / "thermal_roi_examples_panel.png",
        "thermal_delta_panel.png": out / "thermal_delta_panel.png",
        "thermal_contact_sheet.png": out / "thermal_contact_sheet.png",
    }
    write_json(outputs["thermal_metrics.json"], metrics)
    _write_timeseries(outputs["thermal_timeseries.csv"], series)
    _intensity_panel(outputs["thermal_intensity_panel.png"], series)
    _contrast_panel(outputs["thermal_contrast_panel.png"], series, metrics)
    _make_sheet(examples, outputs["thermal_roi_examples_panel.png"], cols=4)
    _delta_panel(outputs["thermal_delta_panel.png"], delta_tiles)
    _make_sheet(contact, outputs["thermal_contact_sheet.png"], cols=4)
    outputs["thermal_roi_sequence.mp4"] = _write_video(case_id, series, track, rows)
    report = _write_report(case_id, track, validation, metrics, outputs)
    outputs["thermal_analysis_report.md"] = report
    manifest = _write_manifest(case_id, outputs)
    outputs["thermal_manifest.md"] = manifest
    status_path = _update_case_status(
        case_id,
        {
            "thermal_analysis_status": "complete",
            "thermal_analysis_ready": True,
            "last_thermal_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_ir_mode": metrics["source_ir_mode"],
            "thermal_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(outputs["thermal_metrics.json"]),
                "csv": str(outputs["thermal_timeseries.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "video": str(outputs["thermal_roi_sequence.mp4"]),
                "panels": {name: str(path) for name, path in outputs.items() if name.endswith(".png")},
            },
        },
    )
    return {
        "case_id": case_id,
        "thermal_analysis_ready": True,
        "source_ir_mode": metrics["source_ir_mode"],
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "case_status": str(status_path),
    }
