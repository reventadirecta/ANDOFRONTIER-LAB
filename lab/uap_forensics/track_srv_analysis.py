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
REQUIRED_MESSAGE = "SRV analysis requires a human-validated track and dynamic ROIs."
NORMALIZED_SIZE = 128


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def dynamic_rois_csv(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "srv_analysis")


def stabilized_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "crops_stabilized")


def enhanced_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "crops_enhanced")


def stack_inputs_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "stack_inputs")


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
            tight = [i("bbox_x"), i("bbox_y"), i("bbox_w"), i("bbox_h")]
            rows.append(
                {
                    "frame": int(raw["frame"]),
                    "status": raw.get("status", ""),
                    "bbox": bbox if all(v is not None for v in bbox) else None,
                    "tight_bbox": tight if all(v is not None for v in tight) else None,
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


def _enhance(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(clahe_bgr, None, 3, 3, 7, 15)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)
    return clahe_bgr, denoised, sharpened


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def _write_timeseries(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "crop_path",
        "normalized_crop_path",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "sharpness_laplacian",
        "contrast_std",
        "luminance_mean",
        "selected_for_stack",
        "track_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fields})


def _make_sheet(images: list[np.ndarray], path: Path, cols: int = 4, bg: int = 245) -> None:
    if not images:
        return
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h, cols * w, 3), bg, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    cv2.imwrite(str(path), sheet)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _collect_crops(case_id: str, track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    cap, fps, _total = _open_video(track)
    raw_tiles: list[np.ndarray] = []
    enhanced_tiles: list[np.ndarray] = []
    source_sizes: list[tuple[int, int]] = []
    records: list[dict[str, Any]] = []
    active = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox")]
    pick_every = max(1, len(active) // 12) if active else 1
    for pos, row in enumerate(active):
        frame_idx = int(row["frame"])
        frame = _read_frame(cap, frame_idx)
        if frame is None:
            continue
        x, y, w, h = [int(v) for v in row["bbox"]]
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        source_sizes.append((crop.shape[1], crop.shape[0]))
        normalized = cv2.resize(crop, (NORMALIZED_SIZE, NORMALIZED_SIZE), interpolation=cv2.INTER_CUBIC)
        clahe, denoised, sharpened = _enhance(normalized)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        crop_path = stabilized_dir(case_id) / f"crop_{frame_idx:06d}.png"
        enhanced_path = enhanced_dir(case_id) / f"crop_{frame_idx:06d}_enhanced.png"
        cv2.imwrite(str(crop_path), normalized)
        cv2.imwrite(str(enhanced_path), sharpened)
        records.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps if fps else 0.0,
                "crop_path": str(crop_path),
                "normalized_crop_path": str(crop_path),
                "enhanced_crop_path": str(enhanced_path),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "sharpness_laplacian": _sharpness(gray),
                "contrast_std": _contrast(gray),
                "luminance_mean": float(gray.mean()),
                "selected_for_stack": False,
                "track_status": row["status"],
                "clahe_path": "",
                "denoised_path": "",
            }
        )
        if pos % pick_every == 0 and len(raw_tiles) < 12:
            raw_tiles.append(_label(cv2.resize(normalized, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{frame_idx} raw"))
            enhanced_tiles.append(_label(cv2.resize(sharpened, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{frame_idx} enh"))
    cap.release()
    return records, raw_tiles, enhanced_tiles, source_sizes


def _stack_images(records: list[dict[str, Any]], case_id: str) -> dict[str, Path]:
    out = output_dir(case_id)
    if not records:
        return {}
    ranked = sorted(records, key=lambda row: row["sharpness_laplacian"], reverse=True)
    selected = ranked[: min(64, len(ranked))]
    selected_frames = {row["frame"] for row in selected}
    arrays = []
    for row in selected:
        row["selected_for_stack"] = True
        img = cv2.imread(row["crop_path"])
        if img is None:
            continue
        arrays.append(img.astype(np.float32))
        cv2.imwrite(str(stack_inputs_dir(case_id) / Path(row["crop_path"]).name), img)
    if not arrays:
        return {}
    stack = np.stack(arrays, axis=0)
    avg = np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)
    med = np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)
    best = cv2.imread(selected[0]["crop_path"])
    best_panel = _make_best_panel(selected[: min(8, len(selected))])
    paths = {
        "srv_stack_average.png": out / "srv_stack_average.png",
        "srv_stack_median.png": out / "srv_stack_median.png",
        "srv_stack_best_sharpness.png": out / "srv_stack_best_sharpness.png",
    }
    cv2.imwrite(str(paths["srv_stack_average.png"]), avg)
    cv2.imwrite(str(paths["srv_stack_median.png"]), med)
    if best_panel is not None:
        cv2.imwrite(str(paths["srv_stack_best_sharpness.png"]), best_panel)
    elif best is not None:
        cv2.imwrite(str(paths["srv_stack_best_sharpness.png"]), best)
    for row in records:
        row["selected_for_stack"] = row["frame"] in selected_frames
    return paths


def _make_best_panel(records: list[dict[str, Any]]) -> np.ndarray | None:
    tiles = []
    for row in records:
        img = cv2.imread(row["crop_path"])
        if img is None:
            continue
        tile = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
        tiles.append(_label(tile, f"f{row['frame']} S={row['sharpness_laplacian']:.1f}"))
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    panel = np.full((h, w * len(tiles), 3), 245, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        panel[:, i * w : (i + 1) * w] = tile
    return panel


def _comparison_panel(case_id: str, records: list[dict[str, Any]], stacks: dict[str, Path]) -> Path:
    path = output_dir(case_id) / "srv_comparison_panel.png"
    if not records:
        return path
    mid = records[len(records) // 2]
    raw = cv2.imread(mid["crop_path"])
    if raw is None:
        return path
    clahe, denoised, sharpened = _enhance(raw)
    avg = cv2.imread(str(stacks.get("srv_stack_average.png", ""))) if stacks else None
    med = cv2.imread(str(stacks.get("srv_stack_median.png", ""))) if stacks else None
    parts = [
        ("raw crop", raw),
        ("normalized", raw),
        ("CLAHE", clahe),
        ("denoise", denoised),
        ("sharpen", sharpened),
        ("average stack", avg if avg is not None else raw),
        ("median stack", med if med is not None else raw),
    ]
    tiles = [_label(cv2.resize(img, (180, 180), interpolation=cv2.INTER_NEAREST), label) for label, img in parts]
    _make_sheet(tiles, path, cols=len(tiles))
    return path


def _quality_panel(case_id: str, records: list[dict[str, Any]]) -> Path:
    path = output_dir(case_id) / "srv_quality_panel.png"
    frames = np.array([row["frame"] for row in records])
    sharp = np.array([row["sharpness_laplacian"] for row in records])
    contrast = np.array([row["contrast_std"] for row in records])
    selected = np.array([bool(row["selected_for_stack"]) for row in records])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, sharp, color="#2f80ed", linewidth=1)
    if len(frames):
        axes[0].scatter(frames[selected], sharp[selected], color="red", s=14, label="selected for stack")
        axes[0].legend()
    axes[0].set_title("Sharpness by frame (Laplacian variance)")
    axes[0].set_ylabel("sharpness")
    axes[1].plot(frames, contrast, color="#f28e2b", linewidth=1)
    axes[1].set_title("Contrast by frame; compression/noise can affect both curves")
    axes[1].set_ylabel("std")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sequence_panel(case_id: str, records: list[dict[str, Any]]) -> Path:
    path = output_dir(case_id) / "srv_stabilized_sequence_panel.png"
    if not records:
        return path
    picks = np.linspace(0, len(records) - 1, min(12, len(records)), dtype=int).tolist()
    tiles = []
    for pos in picks:
        row = records[pos]
        img = cv2.imread(row["crop_path"])
        if img is not None:
            tiles.append(_label(cv2.resize(img, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{row['frame']}"))
    _make_sheet(tiles, path, cols=4)
    return path


def _write_video(case_id: str, records: list[dict[str, Any]]) -> Path:
    raw = output_dir(case_id) / "srv_stabilized_crop_sequence_raw.mp4"
    final = output_dir(case_id) / "srv_stabilized_crop_sequence.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24, (NORMALIZED_SIZE, NORMALIZED_SIZE))
    for row in records:
        img = cv2.imread(row["crop_path"])
        if img is not None:
            writer.write(img)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(final)]
        subprocess.run(command, cwd=str(output_dir(case_id)), capture_output=True, text=True)
        if final.exists() and final.stat().st_size > 0:
            raw.unlink(missing_ok=True)
            return final
    return raw


def _metrics(track: dict[str, Any], records: list[dict[str, Any]], sizes: list[tuple[int, int]], outputs: dict[str, Path]) -> dict[str, Any]:
    sharp = np.array([row["sharpness_laplacian"] for row in records], dtype=np.float32)
    contrast = np.array([row["contrast_std"] for row in records], dtype=np.float32)
    lum = np.array([row["luminance_mean"] for row in records], dtype=np.float32)
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    size_summary = {
        "count": len(sizes),
        "min_w": int(min(widths)) if widths else 0,
        "max_w": int(max(widths)) if widths else 0,
        "mean_w": float(np.mean(widths)) if widths else 0.0,
        "min_h": int(min(heights)) if heights else 0,
        "max_h": int(max(heights)) if heights else 0,
        "mean_h": float(np.mean(heights)) if heights else 0.0,
    }
    return {
        "case_id": track.get("case_id"),
        "total_frames": int(track.get("summary", {}).get("total_frames") or len(records)),
        "valid_tracked_frames": len(records),
        "crop_count": len(records),
        "normalized_crop_size": [NORMALIZED_SIZE, NORMALIZED_SIZE],
        "source_crop_size_summary": size_summary,
        "stabilization_method": "dynamic track ROI crop resized to common square canvas; no object redetection",
        "mean_crop_sharpness": float(np.mean(sharp)) if len(sharp) else 0.0,
        "median_crop_sharpness": float(np.median(sharp)) if len(sharp) else 0.0,
        "max_crop_sharpness": float(np.max(sharp)) if len(sharp) else 0.0,
        "mean_contrast": float(np.mean(contrast)) if len(contrast) else 0.0,
        "mean_luminance": float(np.mean(lum)) if len(lum) else 0.0,
        "stack_method": "top-sharpness temporal average and median over resized tracked crops",
        "denoise_method": "OpenCV fastNlMeansDenoisingColored with mild parameters",
        "clahe_enabled": True,
        "super_resolution_used": False,
        "generated_outputs": {name: str(path) for name, path in outputs.items()},
        "notes": [
            "Visual reconstruction is interpretive and does not prove physical structure or origin.",
            "No generative model was used.",
            "Resize/interpolation is standard OpenCV interpolation; no pixels are hallucinated beyond conservative enhancement.",
        ],
    }


def _write_report(case_id: str, track: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report = case_report_dir(case_id) / "srv_analysis_report.md"
    lines = [
        f"# SRV / Visual Reconstruction Report: {case_id}",
        "",
        "## Scope",
        "",
        "Visual reconstruction from human-validated track crops only. This is conservative enhancement and stacking, not generative reconstruction.",
        "",
        "## Source Track",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- tracker used: `{track.get('tracker_backend') or track.get('backend_used')}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Methods",
        "",
        f"- crops: `{metrics['crop_count']}`",
        f"- normalized crop size: `{metrics['normalized_crop_size']}`",
        f"- stabilization: `{metrics['stabilization_method']}`",
        f"- enhancement: CLAHE, mild denoise, mild sharpen",
        f"- stack method: `{metrics['stack_method']}`",
        "- resize/interpolation: OpenCV resize to common crop size; no generative model.",
        "",
        "## Quality Metrics",
        "",
        f"- mean crop sharpness: `{metrics['mean_crop_sharpness']:.4f}`",
        f"- median crop sharpness: `{metrics['median_crop_sharpness']:.4f}`",
        f"- max crop sharpness: `{metrics['max_crop_sharpness']:.4f}`",
        f"- mean contrast: `{metrics['mean_contrast']:.4f}`",
        f"- mean luminance: `{metrics['mean_luminance']:.4f}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Visual reconstruction is interpretive and does not prove physical structure or origin.",
            "- No generative model was used.",
            "- No automatic ROI, object redetection, target switching, autoencoder, Thermal/IR or origin claim was used.",
            "- Resize, denoise, CLAHE and sharpen can make existing features more visible but can also emphasize compression/noise.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path], report: Path) -> Path:
    manifest = output_dir(case_id) / "srv_manifest.md"
    classes = {
        "srv_metrics.json": "technical",
        "srv_timeseries.csv": "technical",
        "srv_contact_sheet_raw.png": "public_safe",
        "srv_contact_sheet_enhanced.png": "interpretive",
        "srv_stabilized_sequence_panel.png": "public_safe",
        "srv_stack_average.png": "interpretive",
        "srv_stack_median.png": "interpretive",
        "srv_stack_best_sharpness.png": "technical",
        "srv_comparison_panel.png": "interpretive",
        "srv_quality_panel.png": "technical",
        "srv_stabilized_crop_sequence.mp4": "public_safe",
        "srv_analysis_report.md": "public_safe",
    }
    lines = ["# SRV / Visual Reconstruction Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in {**outputs, "srv_analysis_report.md": report}.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | interpretive visual reconstruction from validated track crops only |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_srv_analysis(case_id: str) -> dict[str, Any]:
    track, validation, rows = _require_valid_inputs(case_id)
    out = output_dir(case_id)
    records, raw_tiles, enhanced_tiles, sizes = _collect_crops(case_id, track, rows)
    outputs: dict[str, Path] = {
        "srv_contact_sheet_raw.png": out / "srv_contact_sheet_raw.png",
        "srv_contact_sheet_enhanced.png": out / "srv_contact_sheet_enhanced.png",
    }
    _make_sheet(raw_tiles, outputs["srv_contact_sheet_raw.png"], cols=4)
    _make_sheet(enhanced_tiles, outputs["srv_contact_sheet_enhanced.png"], cols=4)
    stack_paths = _stack_images(records, case_id)
    outputs.update(stack_paths)
    outputs["srv_stabilized_sequence_panel.png"] = _sequence_panel(case_id, records)
    outputs["srv_comparison_panel.png"] = _comparison_panel(case_id, records, stack_paths)
    outputs["srv_quality_panel.png"] = _quality_panel(case_id, records)
    outputs["srv_stabilized_crop_sequence.mp4"] = _write_video(case_id, records)
    metrics = _metrics(track, records, sizes, outputs)
    outputs["srv_metrics.json"] = out / "srv_metrics.json"
    outputs["srv_timeseries.csv"] = out / "srv_timeseries.csv"
    write_json(outputs["srv_metrics.json"], metrics)
    _write_timeseries(outputs["srv_timeseries.csv"], records)
    report = _write_report(case_id, track, validation, metrics, outputs)
    manifest = _write_manifest(case_id, outputs, report)
    outputs["srv_manifest.md"] = manifest
    status_path = _update_case_status(
        case_id,
        {
            "srv_analysis_status": "complete",
            "srv_analysis_ready": True,
            "last_srv_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "srv_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(outputs["srv_metrics.json"]),
                "csv": str(outputs["srv_timeseries.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "panels": {
                    "raw": str(outputs["srv_contact_sheet_raw.png"]),
                    "enhanced": str(outputs["srv_contact_sheet_enhanced.png"]),
                    "sequence": str(outputs["srv_stabilized_sequence_panel.png"]),
                    "comparison": str(outputs["srv_comparison_panel.png"]),
                    "quality": str(outputs["srv_quality_panel.png"]),
                    "average_stack": str(outputs["srv_stack_average.png"]),
                    "median_stack": str(outputs["srv_stack_median.png"]),
                    "best_sharpness": str(outputs["srv_stack_best_sharpness.png"]),
                },
                "video": str(outputs["srv_stabilized_crop_sequence.mp4"]),
            },
        },
    )
    return {
        "case_id": case_id,
        "srv_analysis_ready": True,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "report": str(report),
        "case_status": str(status_path),
    }
