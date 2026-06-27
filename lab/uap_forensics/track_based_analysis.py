from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


VALID_STATES = {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"}


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "track_based_analysis")


def crops_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "crops")


def normalized_crops_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "crops_normalized_64")


def _load_validated_track(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tpath = track_path(case_id)
    vpath = validation_path(case_id)
    if not tpath.exists():
        raise RuntimeError(f"Track not found: {tpath}")
    if not vpath.exists():
        raise RuntimeError("Track must be human validated before rebuilding analysis.")
    validation = read_json(vpath)
    if not (validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target")):
        raise RuntimeError("Track must be human validated before rebuilding analysis.")
    return read_json(tpath), validation


def _open_video(track: dict[str, Any]) -> cv2.VideoCapture:
    video_path = track.get("video", {}).get("path")
    if not video_path:
        raise RuntimeError("track.json does not include video.path")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _clamp_bbox(x: int, y: int, w: int, h: int, width: int, height: int) -> list[int]:
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    w = max(1, min(width - x, int(w)))
    h = max(1, min(height - y, int(h)))
    return [x, y, w, h]


def _expanded_bbox(bbox: list[int], margin: float, width: int, height: int) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    mx = int(round(w * margin))
    my = int(round(h * margin))
    return _clamp_bbox(x - mx, y - my, w + 2 * mx, h + 2 * my, width, height)


def _bbox_xywh(bbox: Any) -> list[int] | None:
    if not bbox:
        return None
    if isinstance(bbox, dict):
        return [int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])]
    return [int(v) for v in bbox]


def _dynamic_rois(track: dict[str, Any], margin: float = 0.25) -> list[dict[str, Any]]:
    width = int(track.get("video", {}).get("width") or 0)
    height = int(track.get("video", {}).get("height") or 0)
    rows = []
    for item in track.get("track", []):
        status = item.get("status") or item.get("state")
        bbox = _bbox_xywh(item.get("bbox") or item.get("bbox_xywh"))
        if not bbox:
            rows.append(
                {
                    "frame": int(item["frame"]),
                    "status": status,
                    "bbox": None,
                    "expanded_bbox": None,
                    "centroid": None,
                    "area": 0,
                    "confidence": float(item.get("confidence", 0.0)),
                }
            )
            continue
        bbox = _clamp_bbox(*bbox, width, height)
        expanded = _expanded_bbox(bbox, margin, width, height)
        x, y, w, h = bbox
        rows.append(
            {
                "frame": int(item["frame"]),
                "status": status,
                "bbox": bbox,
                "expanded_bbox": expanded,
                "centroid": [float(x + w / 2), float(y + h / 2)],
                "area": int(w * h),
                "confidence": float(item.get("confidence", 0.0)),
            }
        )
    return rows


def _write_dynamic_rois(case_id: str, rows: list[dict[str, Any]]) -> None:
    out = output_dir(case_id)
    write_json(out / "dynamic_rois.json", {"case_id": case_id, "dynamic_rois": rows})
    with (out / "dynamic_rois.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "status", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "expanded_x", "expanded_y", "expanded_w", "expanded_h", "centroid_x", "centroid_y", "area", "confidence"])
        writer.writeheader()
        for row in rows:
            bbox = row.get("bbox") or ["", "", "", ""]
            exp = row.get("expanded_bbox") or ["", "", "", ""]
            cen = row.get("centroid") or ["", ""]
            writer.writerow(
                {
                    "frame": row["frame"],
                    "status": row["status"],
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "expanded_x": exp[0],
                    "expanded_y": exp[1],
                    "expanded_w": exp[2],
                    "expanded_h": exp[3],
                    "centroid_x": cen[0],
                    "centroid_y": cen[1],
                    "area": row["area"],
                    "confidence": row["confidence"],
                }
            )


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _save_crops(case_id: str, track: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cap = _open_video(track)
    crop_root = crops_dir(case_id)
    norm_root = normalized_crops_dir(case_id)
    saved = []
    for row in rows:
        if row["status"] not in VALID_STATES or not row.get("expanded_bbox"):
            continue
        frame = _read_frame(cap, row["frame"])
        if frame is None:
            continue
        x, y, w, h = row["expanded_bbox"]
        crop = frame[y : y + h, x : x + w]
        crop_path = crop_root / f"crop_{row['frame']:06d}.png"
        norm_path = norm_root / f"crop_{row['frame']:06d}_64.png"
        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(norm_path), cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA))
        saved.append({"frame": row["frame"], "crop": str(crop_path), "normalized": str(norm_path)})
    cap.release()
    return saved


def _draw_bbox(frame: np.ndarray, bbox: list[int], label: str) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), max(2, out.shape[1] // 700))
    cv2.putText(out, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def _panel_contact_sheet(case_id: str, track: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    valid = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox")]
    picks = np.linspace(0, len(valid) - 1, min(12, len(valid)), dtype=int).tolist() if valid else []
    cap = _open_video(track)
    frames = []
    labels = []
    for pos in picks:
        row = valid[pos]
        frame = _read_frame(cap, row["frame"])
        if frame is None:
            continue
        x, y, w, h = row["bbox"]
        crop = frame[y : y + h, x : x + w]
        crop = cv2.resize(crop, (260, 180), interpolation=cv2.INTER_AREA)
        overview = cv2.resize(_draw_bbox(frame, row["bbox"], f"f{row['frame']}"), (360, 203), interpolation=cv2.INTER_AREA)
        tile = np.full((420, 380, 3), 245, dtype=np.uint8)
        tile[20 : 223, 10 : 370] = overview
        tile[235 : 415, 60 : 320] = crop
        cv2.putText(tile, f"frame {row['frame']}", (14, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        frames.append(tile)
        labels.append("")
    cap.release()
    path = output_dir(case_id) / "track_based_contact_sheet.png"
    _make_sheet(frames, labels, path, cols=3)
    return path


def _make_sheet(frames: list[np.ndarray], labels: list[str], path: Path, cols: int = 3) -> None:
    if not frames:
        return
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    rows = math.ceil(len(frames) / cols)
    sheet = np.full((rows * h, cols * w, 3), 250, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + frame.shape[0], c * w : c * w + frame.shape[1]] = frame
    cv2.imwrite(str(path), sheet)


def _motion_panel(case_id: str, track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, dict[str, float]]:
    valid = [row for row in rows if row["status"] in VALID_STATES and row.get("expanded_bbox")]
    cap = _open_video(track)
    diffs = []
    flows = []
    prev_gray = None
    sample_tiles = []
    for row in valid:
        frame = _read_frame(cap, row["frame"])
        if frame is None:
            continue
        x, y, w, h = row["expanded_bbox"]
        crop = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            diffs.append(float(diff.mean()))
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flows.append(float(mag.mean()))
            if len(sample_tiles) < 8:
                heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_MAGMA)
                sample_tiles.append(cv2.resize(heat, (220, 220), interpolation=cv2.INTER_NEAREST))
        prev_gray = gray
    cap.release()
    path = output_dir(case_id) / "track_based_motion_panel.png"
    _make_sheet(sample_tiles, ["" for _ in sample_tiles], path, cols=4)
    return path, {"mean_frame_diff": float(np.mean(diffs)) if diffs else 0.0, "max_frame_diff": float(np.max(diffs)) if diffs else 0.0, "mean_optical_flow": float(np.mean(flows)) if flows else 0.0, "max_optical_flow": float(np.max(flows)) if flows else 0.0}


def _trajectory_panel(case_id: str, rows: list[dict[str, Any]]) -> tuple[Path, dict[str, float]]:
    valid = [row for row in rows if row.get("centroid")]
    frames = np.array([row["frame"] for row in valid])
    centroids = np.array([row["centroid"] for row in valid], dtype=np.float32) if valid else np.zeros((0, 2), dtype=np.float32)
    areas = np.array([row["area"] for row in valid], dtype=np.float32) if valid else np.zeros(0)
    velocities = np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1)) if len(centroids) > 1 else np.zeros(0)
    path = output_dir(case_id) / "track_based_trajectory_panel.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if len(centroids):
        axes[0].plot(centroids[:, 0], centroids[:, 1], marker=".")
        axes[0].invert_yaxis()
    axes[0].set_title("centroid trajectory")
    axes[1].plot(frames[1:], velocities) if len(velocities) else axes[1].plot([])
    axes[1].set_title("velocity px/frame")
    axes[2].plot(frames, areas) if len(frames) else axes[2].plot([])
    axes[2].set_title("bbox area")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, {"mean_velocity_px_frame": float(np.mean(velocities)) if len(velocities) else 0.0, "max_velocity_px_frame": float(np.max(velocities)) if len(velocities) else 0.0, "mean_area": float(np.mean(areas)) if len(areas) else 0.0}


def _pca_panel(case_id: str, crop_records: list[dict[str, Any]]) -> tuple[Path, dict[str, float]]:
    matrices = []
    for rec in crop_records:
        img = cv2.imread(rec["normalized"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            matrices.append(img.astype(np.float32).reshape(-1) / 255.0)
    path = output_dir(case_id) / "track_based_pca_panel.png"
    if len(matrices) < 5:
        return path, {"pc1": 0.0, "k5": 0.0, "k10": 0.0, "frames": len(matrices)}
    x = np.vstack(matrices)
    x = StandardScaler(with_std=False).fit_transform(x)
    n_comp = min(10, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42).fit(x)
    mean_img = np.mean(np.vstack(matrices), axis=0).reshape(64, 64)
    pc1 = pca.components_[0].reshape(64, 64)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(mean_img, cmap="gray")
    axes[0].set_title("mean tracked crop")
    axes[1].imshow(pc1, cmap="gray")
    axes[1].set_title("PC1")
    axes[2].plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    axes[2].set_title("PCA cumulative variance")
    for ax in axes[:2]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    ratios = pca.explained_variance_ratio_
    return path, {"pc1": float(ratios[0]), "k5": float(np.sum(ratios[: min(5, len(ratios))])), "k10": float(np.sum(ratios[: min(10, len(ratios))])), "frames": len(matrices)}


def _write_report(case_id: str, track: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], panels: dict[str, Path], crop_count: int) -> Path:
    report = case_report_dir(case_id) / "track_based_analysis_report.md"
    lines = [
        f"# Track-Based Analysis Report: {case_id}",
        "",
        "## Source Track",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- tracker: `{track.get('tracker_backend') or track.get('backend_used')}`",
        f"- first_object_frame: `{track.get('first_object_frame')}`",
        f"- initial_box: `{track.get('initial_box')}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Summary",
        "",
        f"- total track frames: `{metrics['total_track_frames']}`",
        f"- valid analysis frames: `{metrics['valid_analysis_frames']}`",
        f"- lost frames: `{metrics['lost_frames']}`",
        f"- crops generated: `{crop_count}`",
        "",
        "## Motion Metrics",
        "",
        f"- mean frame diff: `{metrics['motion']['mean_frame_diff']:.4f}`",
        f"- max frame diff: `{metrics['motion']['max_frame_diff']:.4f}`",
        f"- mean optical flow: `{metrics['motion']['mean_optical_flow']:.4f}`",
        f"- max optical flow: `{metrics['motion']['max_optical_flow']:.4f}`",
        "",
        "## PCA Metrics",
        "",
        f"- PC1: `{metrics['pca']['pc1']:.4f}`",
        f"- cumulative k=5: `{metrics['pca']['k5']:.4f}`",
        f"- cumulative k=10: `{metrics['pca']['k10']:.4f}`",
        "",
        "## Panels",
        "",
    ]
    for name, path in panels.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This analysis depends on the human-validated interactive track.",
            "- It does not prove origin or nature of the object.",
            "- Source quality and compression remain limiting factors.",
            "- No frames before `first_object_frame` are treated as object frames.",
            "- No old automatic ROI is used when the validated track is available.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, panels: dict[str, Path], report: Path) -> Path:
    manifest = output_dir(case_id) / "track_based_manifest.md"
    items = [
        ("dynamic_rois.csv", output_dir(case_id) / "dynamic_rois.csv", "debug", "per-frame original/expanded track ROI table"),
        ("dynamic_rois.json", output_dir(case_id) / "dynamic_rois.json", "debug", "structured dynamic ROI data"),
        ("crops/", crops_dir(case_id), "debug", "tracked-object crops from active frames"),
        ("track_based_contact_sheet.png", panels["contact_sheet"], "public_review", "key tracked frames with bbox/crops"),
        ("track_based_motion_panel.png", panels["motion"], "public_review", "frame differencing inside dynamic track ROI"),
        ("track_based_trajectory_panel.png", panels["trajectory"], "public_review", "centroid, velocity and apparent area"),
        ("track_based_pca_panel.png", panels["pca"], "public_review", "PCA from tracked crops"),
        ("track_based_analysis_report.md", report, "public_review", "method and metrics report"),
    ]
    lines = ["# Track-Based Analysis Manifest", "", f"Case: `{case_id}`", "", "| asset | path | use | represents | warning |", "| --- | --- | --- | --- | --- |"]
    for name, path, use, desc in items:
        lines.append(f"| `{name}` | `{path}` | `{use}` | {desc} | depends on validated track |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = DATA_DIR / "cases" / case_id / "case_status.json"
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def _mark_review_superseded(case_id: str) -> None:
    reports = DATA_DIR / "reports" / "batches"
    if not reports.exists():
        return
    for review_pack in reports.glob("*/review_pack"):
        marker = review_pack / "SUPERSEDED_BY_TRACK_BASED_ANALYSIS.md"
        text = f"# Superseded by Track-Based Analysis\n\nCase `{case_id}` has track-based analysis outputs. Future review packs should use track-based outputs instead of old automatic ROIs.\n"
        marker.write_text(text, encoding="utf-8")


def rebuild_from_track(case_id: str, with_autoencoder: bool = False, margin: float = 0.25) -> dict[str, Any]:
    track, validation = _load_validated_track(case_id)
    out = output_dir(case_id)
    rows = _dynamic_rois(track, margin=margin)
    _write_dynamic_rois(case_id, rows)
    crop_records = _save_crops(case_id, track, rows)
    contact_path = _panel_contact_sheet(case_id, track, rows)
    motion_path, motion_metrics = _motion_panel(case_id, track, rows)
    trajectory_path, trajectory_metrics = _trajectory_panel(case_id, rows)
    pca_path, pca_metrics = _pca_panel(case_id, crop_records)
    autoencoder_path = out / "track_based_autoencoder_panel.png"
    panels = {"contact_sheet": contact_path, "motion": motion_path, "trajectory": trajectory_path, "pca": pca_path}
    if with_autoencoder:
        autoencoder_path.write_text("Autoencoder mode is reserved for explicit heavy-analysis runs.\n", encoding="utf-8")
        panels["autoencoder"] = autoencoder_path
    valid_frames = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox")]
    lost_frames = [row for row in rows if row["status"] in {"TRACK_LOST", "TRACK LOST"}]
    metrics = {
        "total_track_frames": len(rows),
        "valid_analysis_frames": len(valid_frames),
        "lost_frames": len(lost_frames),
        "motion": motion_metrics,
        "trajectory": trajectory_metrics,
        "pca": pca_metrics,
        "autoencoder_executed": bool(with_autoencoder),
    }
    write_json(out / "track_based_metrics.json", metrics)
    report = _write_report(case_id, track, validation, metrics, panels, len(crop_records))
    manifest = _write_manifest(case_id, panels, report)
    status_path = _update_case_status(
        case_id,
        {
            "tracking_status": "tracking_human_validated",
            "track_based_analysis_status": "complete",
            "track_based_analysis_ready": True,
            "last_track_based_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "track_based_paths": {
                "output_dir": str(out),
                "dynamic_rois_csv": str(out / "dynamic_rois.csv"),
                "dynamic_rois_json": str(out / "dynamic_rois.json"),
                "report": str(report),
                "manifest": str(manifest),
                **{key: str(value) for key, value in panels.items()},
            },
        },
    )
    _mark_review_superseded(case_id)
    return {
        "case_id": case_id,
        "track_validated": True,
        "output_dir": str(out),
        "frames_tracked": len(rows),
        "valid_analysis_frames": len(valid_frames),
        "lost_frames": len(lost_frames),
        "metrics": metrics,
        "panels": {key: str(value) for key, value in panels.items()},
        "report": str(report),
        "manifest": str(manifest),
        "case_status": str(status_path),
        "ready_for_track_based_deep_analysis": True,
    }
