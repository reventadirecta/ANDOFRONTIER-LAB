from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


VALID_STATES = {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"}
REQUIRED_MESSAGE = "Motion analysis requires a human-validated track and dynamic ROIs."


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def track_based_dir(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis"


def dynamic_rois_csv(case_id: str) -> Path:
    return track_based_dir(case_id) / "dynamic_rois.csv"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "motion_analysis")


def case_status_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "case_status.json"


def _require_valid_inputs(case_id: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    tpath = track_path(case_id)
    vpath = validation_path(case_id)
    rpath = dynamic_rois_csv(case_id)
    if not tpath.exists() or not vpath.exists() or not rpath.exists():
        raise RuntimeError(REQUIRED_MESSAGE)
    track = read_json(tpath)
    validation = read_json(vpath)
    if not (validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target")):
        raise RuntimeError(REQUIRED_MESSAGE)
    rows = _read_dynamic_rois(rpath)
    if not rows:
        raise RuntimeError(REQUIRED_MESSAGE)
    return track, validation, rows


def _read_dynamic_rois(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            def to_int(name: str) -> int | None:
                value = raw.get(name, "")
                return int(float(value)) if value not in {"", None} else None

            def to_float(name: str) -> float | None:
                value = raw.get(name, "")
                return float(value) if value not in {"", None} else None

            bbox = [to_int("bbox_x"), to_int("bbox_y"), to_int("bbox_w"), to_int("bbox_h")]
            expanded = [to_int("expanded_x"), to_int("expanded_y"), to_int("expanded_w"), to_int("expanded_h")]
            rows.append(
                {
                    "frame": int(raw["frame"]),
                    "status": raw.get("status", ""),
                    "bbox": bbox if all(v is not None for v in bbox) else None,
                    "expanded_bbox": expanded if all(v is not None for v in expanded) else None,
                    "centroid": [to_float("centroid_x"), to_float("centroid_y")],
                    "area": float(raw.get("area") or 0),
                    "confidence": float(raw.get("confidence") or 0),
                }
            )
    return sorted(rows, key=lambda row: row["frame"])


def _open_video(track: dict[str, Any]) -> tuple[cv2.VideoCapture, float, int]:
    video_path = track.get("video", {}).get("path")
    if not video_path:
        raise RuntimeError("track.json does not include video.path")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or track.get("video", {}).get("fps") or 30)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or track.get("video", {}).get("frame_count") or 0)
    return cap, fps, total


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _crop_gray(frame: np.ndarray, bbox: list[int], size: int = 128) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _flow_tile(prev_gray: np.ndarray, gray: np.ndarray, label: str) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    heat = cv2.applyColorMap(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tile = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    step = 16
    for y in range(step // 2, tile.shape[0], step):
        for x in range(step // 2, tile.shape[1], step):
            dx, dy = flow[y, x]
            end = (int(x + dx * 2.5), int(y + dy * 2.5))
            cv2.arrowedLine(tile, (x, y), end, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(tile, (220, 220), interpolation=cv2.INTER_AREA)


def _make_sheet(tiles: list[np.ndarray], path: Path, cols: int = 4) -> None:
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + tile.shape[0], c * w : c * w + tile.shape[1]] = tile
    cv2.imwrite(str(path), sheet)


def _compute_timeseries(track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[np.ndarray]]:
    cap, fps, total = _open_video(track)
    valid_rows = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox") and row.get("expanded_bbox")]
    prev_centroid = None
    prev_velocity = None
    prev_frame = None
    prev_gray = None
    series: list[dict[str, Any]] = []
    flow_tiles: list[np.ndarray] = []
    for row in valid_rows:
        frame_idx = int(row["frame"])
        centroid = row.get("centroid") or [None, None]
        bbox = row["bbox"]
        frame_gap = max(1, frame_idx - prev_frame) if prev_frame is not None else 1
        velocity = 0.0
        acceleration = 0.0
        if prev_centroid is not None and centroid[0] is not None and centroid[1] is not None:
            dist = float(np.linalg.norm(np.array(centroid, dtype=np.float32) - np.array(prev_centroid, dtype=np.float32)))
            velocity = dist / frame_gap
            if prev_velocity is not None:
                acceleration = (velocity - prev_velocity) / frame_gap
        frame_diff_mean = 0.0
        flow_mean = 0.0
        flow_max = 0.0
        frame = _read_frame(cap, frame_idx)
        if frame is not None:
            gray = _crop_gray(frame, row["expanded_bbox"])
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                frame_diff_mean = float(diff.mean())
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mean = float(mag.mean())
                flow_max = float(mag.max())
                if len(flow_tiles) < 8:
                    flow_tiles.append(_flow_tile(prev_gray, gray, f"f{frame_idx}"))
            prev_gray = gray
        series.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps if fps else 0.0,
                "centroid_x": float(centroid[0] or 0),
                "centroid_y": float(centroid[1] or 0),
                "bbox_x": int(bbox[0]),
                "bbox_y": int(bbox[1]),
                "bbox_w": int(bbox[2]),
                "bbox_h": int(bbox[3]),
                "bbox_area": float(row.get("area") or bbox[2] * bbox[3]),
                "velocity_px_per_frame": velocity,
                "acceleration_px_per_frame2": acceleration,
                "frame_diff_mean": frame_diff_mean,
                "optical_flow_mean": flow_mean,
                "optical_flow_max": flow_max,
                "track_status": row["status"],
            }
        )
        prev_centroid = centroid
        prev_velocity = velocity
        prev_frame = frame_idx
    cap.release()
    lost_frames = [row for row in rows if row["status"] in {"TRACK_LOST", "TRACK LOST", "LOW_CONFIDENCE"}]
    metrics = _summarize_metrics(track, rows, series, lost_frames, total)
    return series, metrics, flow_tiles


def _summarize_metrics(track: dict[str, Any], rows: list[dict[str, Any]], series: list[dict[str, Any]], lost_frames: list[dict[str, Any]], total_video_frames: int) -> dict[str, Any]:
    velocities = np.array([row["velocity_px_per_frame"] for row in series[1:]], dtype=np.float32)
    accelerations = np.array([abs(row["acceleration_px_per_frame2"]) for row in series[2:]], dtype=np.float32)
    areas = np.array([row["bbox_area"] for row in series], dtype=np.float32)
    diffs = np.array([row["frame_diff_mean"] for row in series[1:]], dtype=np.float32)
    flows = np.array([row["optical_flow_mean"] for row in series[1:]], dtype=np.float32)
    flow_max = np.array([row["optical_flow_max"] for row in series[1:]], dtype=np.float32)
    valid_frames = len(series)
    span = (series[-1]["frame"] - series[0]["frame"] + 1) if series else 0
    continuity = float(valid_frames / span) if span else 0.0
    area_variation = float(np.std(areas) / max(1.0, float(np.mean(areas)))) if len(areas) else 0.0
    jitter = float(np.std(velocities) / max(1e-6, float(np.mean(velocities)))) if len(velocities) and float(np.mean(velocities)) > 0 else 0.0
    stability = float(max(0.0, min(1.0, continuity * (1.0 / (1.0 + area_variation + jitter)))))
    return {
        "case_id": track.get("case_id"),
        "total_frames": int(total_video_frames or track.get("summary", {}).get("total_frames") or len(rows)),
        "valid_tracked_frames": valid_frames,
        "lost_frames": len(lost_frames),
        "mean_velocity_px_frame": float(np.mean(velocities)) if len(velocities) else 0.0,
        "median_velocity_px_frame": float(np.median(velocities)) if len(velocities) else 0.0,
        "max_velocity_px_frame": float(np.max(velocities)) if len(velocities) else 0.0,
        "mean_acceleration_px_frame2": float(np.mean(accelerations)) if len(accelerations) else 0.0,
        "max_acceleration_px_frame2": float(np.max(accelerations)) if len(accelerations) else 0.0,
        "mean_bbox_area": float(np.mean(areas)) if len(areas) else 0.0,
        "bbox_area_variation": area_variation,
        "mean_frame_difference_inside_roi": float(np.mean(diffs)) if len(diffs) else 0.0,
        "mean_optical_flow_magnitude_inside_roi": float(np.mean(flows)) if len(flows) else 0.0,
        "max_optical_flow_magnitude_inside_roi": float(np.max(flow_max)) if len(flow_max) else 0.0,
        "motion_continuity_score": continuity,
        "jitter_score": jitter,
        "track_stability_score": stability,
        "tracker_used": track.get("tracker_backend") or track.get("backend_used"),
        "notes": [
            "Motion is measured in pixels/frame inside human-validated dynamic track ROIs.",
            "This module does not estimate real-world speed in m/s.",
            "No automatic ROI, brightness redetection, object switching, autoencoder, or origin claim is used.",
        ],
    }


def _write_timeseries(path: Path, series: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "centroid_x",
        "centroid_y",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "velocity_px_per_frame",
        "acceleration_px_per_frame2",
        "frame_diff_mean",
        "optical_flow_mean",
        "optical_flow_max",
        "track_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(series)


def _trajectory_panel(path: Path, series: list[dict[str, Any]]) -> None:
    xs = np.array([row["centroid_x"] for row in series])
    ys = np.array([row["centroid_y"] for row in series])
    frames = np.array([row["frame"] for row in series])
    velocities = np.array([row["velocity_px_per_frame"] for row in series])
    fig, ax = plt.subplots(figsize=(9, 7))
    if len(series):
        sc = ax.scatter(xs, ys, c=velocities, cmap="plasma", s=14)
        ax.plot(xs, ys, color="#555555", linewidth=0.8, alpha=0.65)
        ax.scatter(xs[0], ys[0], color="lime", s=70, label=f"start f{frames[0]}")
        ax.scatter(xs[-1], ys[-1], color="red", s=70, label=f"end f{frames[-1]}")
        if len(velocities) > 3:
            threshold = np.percentile(velocities, 90)
            high = velocities >= threshold
            ax.scatter(xs[high], ys[high], facecolors="none", edgecolors="orange", s=70, label="high velocity")
        fig.colorbar(sc, ax=ax, label="px/frame")
    ax.invert_yaxis()
    ax.set_title("Centroid trajectory from validated track")
    ax.set_xlabel("x px")
    ax.set_ylabel("y px")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _velocity_panel(path: Path, series: list[dict[str, Any]]) -> None:
    frames = np.array([row["frame"] for row in series])
    velocities = np.array([row["velocity_px_per_frame"] for row in series])
    accelerations = np.array([row["acceleration_px_per_frame2"] for row in series])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, velocities, color="#0b84ff", linewidth=1)
    if len(velocities) > 3:
        peaks = velocities >= np.percentile(velocities, 95)
        axes[0].scatter(frames[peaks], velocities[peaks], color="red", s=20, label="top 5%")
        axes[0].legend()
    axes[0].set_title("Velocity px/frame")
    axes[0].set_ylabel("px/frame")
    axes[1].plot(frames, accelerations, color="#f28e2b", linewidth=1)
    axes[1].set_title("Acceleration px/frame^2")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("px/frame^2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _stability_panel(path: Path, series: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    frames = np.array([row["frame"] for row in series])
    areas = np.array([row["bbox_area"] for row in series])
    velocities = np.array([row["velocity_px_per_frame"] for row in series])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frames, areas, color="#4e79a7")
    axes[0].set_title("bbox area over time")
    axes[0].set_xlabel("frame")
    axes[1].plot(frames, velocities, color="#59a14f")
    axes[1].set_title(f"jitter score: {metrics['jitter_score']:.3f}")
    axes[1].set_xlabel("frame")
    bars = ["continuity", "stability", "area variation"]
    values = [metrics["motion_continuity_score"], metrics["track_stability_score"], metrics["bbox_area_variation"]]
    axes[2].bar(bars, values, color=["#76b7b2", "#edc948", "#e15759"])
    axes[2].set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    axes[2].set_title("track stability summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(case_id: str, track: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], paths: dict[str, Path]) -> Path:
    report = output_dir(case_id) / "motion_analysis_report.md"
    lines = [
        f"# Motion / Optical Flow Track-Based Report: {case_id}",
        "",
        "## Scope",
        "",
        "This module analyzes apparent motion only from the human-validated track and dynamic ROIs reconstructed from `track.json`.",
        "",
        "## Source Track",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- tracker used: `{metrics.get('tracker_used')}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        f"- track is correct: `{bool(validation.get('track_is_correct'))}`",
        f"- object is real target: `{bool(validation.get('object_is_real_target'))}`",
        "",
        "## Metrics",
        "",
        f"- total frames: `{metrics['total_frames']}`",
        f"- valid tracked frames: `{metrics['valid_tracked_frames']}`",
        f"- lost frames: `{metrics['lost_frames']}`",
        f"- mean velocity px/frame: `{metrics['mean_velocity_px_frame']:.4f}`",
        f"- median velocity px/frame: `{metrics['median_velocity_px_frame']:.4f}`",
        f"- max velocity px/frame: `{metrics['max_velocity_px_frame']:.4f}`",
        f"- mean acceleration px/frame^2: `{metrics['mean_acceleration_px_frame2']:.4f}`",
        f"- max acceleration px/frame^2: `{metrics['max_acceleration_px_frame2']:.4f}`",
        f"- mean frame difference inside ROI: `{metrics['mean_frame_difference_inside_roi']:.4f}`",
        f"- mean optical flow magnitude inside ROI: `{metrics['mean_optical_flow_magnitude_inside_roi']:.4f}`",
        f"- max optical flow magnitude inside ROI: `{metrics['max_optical_flow_magnitude_inside_roi']:.4f}`",
        f"- motion continuity score: `{metrics['motion_continuity_score']:.4f}`",
        f"- jitter score: `{metrics['jitter_score']:.4f}`",
        f"- track stability score: `{metrics['track_stability_score']:.4f}`",
        "",
        "## Interpretation",
        "",
        "The measurements describe apparent image-plane motion in pixels/frame. They are suitable for comparing track continuity, local motion intensity, and ROI stability inside this file.",
        "",
        "## Limitations",
        "",
        "- Pixels/frame is not real-world velocity in m/s.",
        "- Results depend on camera motion, zoom, stabilization, compression, frame rate, and manual track precision.",
        "- Optical flow is computed inside dynamic ROI crops and may include target blur or local background within the crop margin.",
        "- This is not an origin or nature claim.",
        "- No automatic ROI, brightness redetection, object switching, autoencoder, SRV, or spectral module was used.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "motion_manifest.md"
    classes = {
        "motion_metrics.json": "technical",
        "motion_timeseries.csv": "technical",
        "motion_trajectory_panel.png": "public_safe",
        "motion_velocity_panel.png": "technical",
        "motion_optical_flow_panel.png": "technical",
        "motion_stability_panel.png": "technical",
        "motion_analysis_report.md": "public_safe",
    }
    lines = ["# Motion Analysis Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in outputs.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | apparent px/frame motion only; depends on validated track |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_motion_analysis(case_id: str) -> dict[str, Any]:
    track, validation, rows = _require_valid_inputs(case_id)
    out = output_dir(case_id)
    series, metrics, flow_tiles = _compute_timeseries(track, rows)
    paths = {
        "motion_metrics.json": out / "motion_metrics.json",
        "motion_timeseries.csv": out / "motion_timeseries.csv",
        "motion_trajectory_panel.png": out / "motion_trajectory_panel.png",
        "motion_velocity_panel.png": out / "motion_velocity_panel.png",
        "motion_optical_flow_panel.png": out / "motion_optical_flow_panel.png",
        "motion_stability_panel.png": out / "motion_stability_panel.png",
    }
    write_json(paths["motion_metrics.json"], metrics)
    _write_timeseries(paths["motion_timeseries.csv"], series)
    _trajectory_panel(paths["motion_trajectory_panel.png"], series)
    _velocity_panel(paths["motion_velocity_panel.png"], series)
    _make_sheet(flow_tiles, paths["motion_optical_flow_panel.png"], cols=4)
    _stability_panel(paths["motion_stability_panel.png"], series, metrics)
    report = _write_report(case_id, track, validation, metrics, paths)
    paths["motion_analysis_report.md"] = report
    manifest = _write_manifest(case_id, paths)
    paths["motion_manifest.md"] = manifest
    status_path = _update_case_status(
        case_id,
        {
            "motion_analysis_status": "complete",
            "motion_analysis_ready": True,
            "last_motion_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "motion_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(paths["motion_metrics.json"]),
                "csv": str(paths["motion_timeseries.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "panels": {
                    "trajectory": str(paths["motion_trajectory_panel.png"]),
                    "velocity": str(paths["motion_velocity_panel.png"]),
                    "optical_flow": str(paths["motion_optical_flow_panel.png"]),
                    "stability": str(paths["motion_stability_panel.png"]),
                },
            },
        },
    )
    return {
        "case_id": case_id,
        "motion_analysis_ready": True,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in paths.items()},
        "case_status": str(status_path),
    }
