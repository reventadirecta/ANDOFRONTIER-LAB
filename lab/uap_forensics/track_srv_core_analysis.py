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
from . import track_srv_analysis as srv


CORE_SIZE = 128
REQUIRED_MESSAGE = "SRV analysis requires a human-validated track and dynamic ROIs."


def core_dir(case_id: str) -> Path:
    return ensure_dir(srv.output_dir(case_id) / "object_core")


def core_crops_dir(case_id: str) -> Path:
    return ensure_dir(core_dir(case_id) / "crops_core")


def core_enhanced_dir(case_id: str) -> Path:
    return ensure_dir(core_dir(case_id) / "crops_core_enhanced")


def core_stack_inputs_dir(case_id: str) -> Path:
    return ensure_dir(core_dir(case_id) / "stack_inputs")


def _artifact_mask(crop: np.ndarray) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cyan = ((hsv[..., 0] >= 78) & (hsv[..., 0] <= 105) & (hsv[..., 1] > 55) & (hsv[..., 2] > 55)).astype(np.uint8) * 255
    cyan = cv2.dilate(cyan, np.ones((5, 5), np.uint8), iterations=1)

    dark = ((gray < 8).astype(np.uint8) * 255)
    num, labels, stats, _cent = cv2.connectedComponentsWithStats(dark, connectivity=8)
    dark_art = np.zeros_like(dark)
    h, w = gray.shape
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        touches_border = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        rectangular_overlay = area > 0.015 * h * w and (touches_border or bw > 0.35 * w or bh > 0.35 * h)
        if rectangular_overlay:
            dark_art[labels == idx] = 255
    dark_art = cv2.dilate(dark_art, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.bitwise_or(cyan, dark_art)
    return mask, int(np.count_nonzero(mask))


def _clean_artifacts(crop: np.ndarray) -> tuple[np.ndarray, int]:
    mask, pixels = _artifact_mask(crop)
    if pixels == 0:
        return crop, 0
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, int(np.count_nonzero(mask))


def _detect_core(crop: np.ndarray, prev_core_local: tuple[float, float, float] | None) -> tuple[list[int] | None, str, float, int]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask, mask_pixels = _artifact_mask(crop)
    median = float(np.median(gray[mask == 0])) if np.any(mask == 0) else float(np.median(gray))
    saliency = np.abs(gray.astype(np.float32) - median)
    saliency[mask > 0] = 0
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    nonzero = saliency[saliency > 0]
    if nonzero.size == 0:
        return None, "CORE_NOT_FOUND", 0.0, mask_pixels
    thresh = max(float(np.percentile(nonzero, 97.5)), float(nonzero.mean() + nonzero.std()))
    binary = (saliency >= thresh).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    num, labels, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = gray.shape
    candidates: list[tuple[float, list[int], float]] = []
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        if area < 6 or area > 0.30 * h * w:
            continue
        if (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1) and area > 0.03 * h * w:
            continue
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if aspect > 18:
            continue
        cx, cy = cent[idx]
        local = saliency[labels == idx]
        score = float(local.mean()) * math.sqrt(float(area))
        if prev_core_local:
            px, py, ps = prev_core_local
            dist = math.hypot(cx - px, cy - py)
            score -= dist * 0.45
            size_delta = abs(math.sqrt(area) - ps)
            score -= size_delta * 0.25
        candidates.append((score, [int(x), int(y), int(bw), int(bh)], float(area)))
    if not candidates:
        return None, "CORE_NOT_FOUND", 0.0, mask_pixels
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, bbox, area = candidates[0]
    x, y, bw, bh = bbox
    pad = max(8, int(round(max(bw, bh) * 1.35)))
    cx = x + bw / 2
    cy = y + bh / 2
    side = min(max(pad, 32), max(w, h))
    nx = int(round(cx - side / 2))
    ny = int(round(cy - side / 2))
    nx = max(0, min(w - side, nx))
    ny = max(0, min(h - side, ny))
    confidence = float(max(0.0, score) / (area + 1e-6))
    status = "TRACKING_ACTIVE" if confidence > 12 else "LOW_CONFIDENCE"
    return [nx, ny, int(side), int(side)], status, confidence, mask_pixels


def _smooth_core(current: list[int], prev: list[int] | None) -> list[int]:
    if prev is None:
        return current
    alpha = 0.35
    return [int(round(prev[i] * (1 - alpha) + current[i] * alpha)) for i in range(4)]


def _write_core_timeseries(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "track_bbox_x",
        "track_bbox_y",
        "track_bbox_w",
        "track_bbox_h",
        "core_x",
        "core_y",
        "core_w",
        "core_h",
        "status",
        "confidence",
        "hud_mask_pixels",
        "crop_path",
        "enhanced_crop_path",
        "sharpness_laplacian",
        "contrast_std",
        "luminance_mean",
        "selected_for_stack",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fields})


def _collect_core(case_id: str, track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    cap, fps, _total = srv._open_video(track)
    records: list[dict[str, Any]] = []
    raw_tiles: list[np.ndarray] = []
    enhanced_tiles: list[np.ndarray] = []
    active = [row for row in rows if row["status"] in srv.VALID_STATES and row.get("bbox")]
    pick_every = max(1, len(active) // 12) if active else 1
    prev_local: list[int] | None = None
    prev_descriptor: tuple[float, float, float] | None = None
    for pos, row in enumerate(active):
        frame_idx = int(row["frame"])
        frame = srv._read_frame(cap, frame_idx)
        if frame is None:
            continue
        tx, ty, tw, th = [int(v) for v in row["bbox"]]
        bbox_crop = frame[ty : ty + th, tx : tx + tw]
        detected, status, confidence, mask_pixels = _detect_core(bbox_crop, prev_descriptor)
        if detected is None and prev_local is not None:
            detected = prev_local
            status = "LOW_CONFIDENCE"
        if detected is None:
            records.append(
                {
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps if fps else 0.0,
                    "track_bbox_x": tx,
                    "track_bbox_y": ty,
                    "track_bbox_w": tw,
                    "track_bbox_h": th,
                    "status": "CORE_NOT_FOUND",
                    "confidence": 0.0,
                    "hud_mask_pixels": mask_pixels,
                    "selected_for_stack": False,
                }
            )
            continue
        smoothed = _smooth_core(detected, prev_local)
        cx, cy, cw, ch = smoothed
        cx = max(0, min(bbox_crop.shape[1] - 1, cx))
        cy = max(0, min(bbox_crop.shape[0] - 1, cy))
        cw = max(8, min(bbox_crop.shape[1] - cx, cw))
        ch = max(8, min(bbox_crop.shape[0] - cy, ch))
        core = bbox_crop[cy : cy + ch, cx : cx + cw]
        normalized = cv2.resize(core, (CORE_SIZE, CORE_SIZE), interpolation=cv2.INTER_CUBIC)
        normalized, core_mask_pixels = _clean_artifacts(normalized)
        _clahe, _denoised, enhanced = srv._enhance(normalized)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        crop_path = core_crops_dir(case_id) / f"core_{frame_idx:06d}.png"
        enhanced_path = core_enhanced_dir(case_id) / f"core_{frame_idx:06d}_enhanced.png"
        cv2.imwrite(str(crop_path), normalized)
        cv2.imwrite(str(enhanced_path), enhanced)
        prev_local = [cx, cy, cw, ch]
        prev_descriptor = (cx + cw / 2, cy + ch / 2, math.sqrt(cw * ch))
        record = {
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps else 0.0,
            "track_bbox_x": tx,
            "track_bbox_y": ty,
            "track_bbox_w": tw,
            "track_bbox_h": th,
            "core_x": tx + cx,
            "core_y": ty + cy,
            "core_w": cw,
            "core_h": ch,
            "status": status,
            "confidence": confidence,
            "hud_mask_pixels": mask_pixels + core_mask_pixels,
            "crop_path": str(crop_path),
            "enhanced_crop_path": str(enhanced_path),
            "sharpness_laplacian": srv._sharpness(gray),
            "contrast_std": srv._contrast(gray),
            "luminance_mean": float(gray.mean()),
            "selected_for_stack": False,
        }
        records.append(record)
        if pos % pick_every == 0 and len(raw_tiles) < 12:
            raw_tiles.append(srv._label(cv2.resize(normalized, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{frame_idx} core"))
            enhanced_tiles.append(srv._label(cv2.resize(enhanced, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{frame_idx} enh"))
    cap.release()
    return records, raw_tiles, enhanced_tiles


def _stack_core(case_id: str, records: list[dict[str, Any]]) -> dict[str, Path]:
    valid = [row for row in records if row.get("crop_path") and row.get("status") != "CORE_NOT_FOUND"]
    ranked = sorted(valid, key=lambda row: row["sharpness_laplacian"], reverse=True)
    selected = ranked[: min(64, len(ranked))]
    selected_frames = {row["frame"] for row in selected}
    arrays = []
    for row in selected:
        img = cv2.imread(row["crop_path"])
        if img is None:
            continue
        arrays.append(img.astype(np.float32))
        cv2.imwrite(str(core_stack_inputs_dir(case_id) / Path(row["crop_path"]).name), img)
    for row in records:
        row["selected_for_stack"] = row["frame"] in selected_frames
    paths = {
        "srv_core_stack_average.png": core_dir(case_id) / "srv_core_stack_average.png",
        "srv_core_stack_median.png": core_dir(case_id) / "srv_core_stack_median.png",
        "srv_core_stack_best_sharpness.png": core_dir(case_id) / "srv_core_stack_best_sharpness.png",
    }
    if not arrays:
        return paths
    stack = np.stack(arrays, axis=0)
    cv2.imwrite(str(paths["srv_core_stack_average.png"]), np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8))
    cv2.imwrite(str(paths["srv_core_stack_median.png"]), np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8))
    best_tiles = []
    for row in selected[: min(8, len(selected))]:
        img = cv2.imread(row["crop_path"])
        if img is not None:
            best_tiles.append(srv._label(cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST), f"f{row['frame']} S={row['sharpness_laplacian']:.1f}"))
    srv._make_sheet(best_tiles, paths["srv_core_stack_best_sharpness.png"], cols=len(best_tiles) or 1)
    return paths


def _sequence_panel(case_id: str, records: list[dict[str, Any]]) -> Path:
    path = core_dir(case_id) / "srv_core_stabilized_sequence_panel.png"
    valid = [row for row in records if row.get("crop_path")]
    picks = np.linspace(0, len(valid) - 1, min(12, len(valid)), dtype=int).tolist() if valid else []
    tiles = []
    for pos in picks:
        row = valid[pos]
        img = cv2.imread(row["crop_path"])
        if img is not None:
            tiles.append(srv._label(cv2.resize(img, (180, 180), interpolation=cv2.INTER_NEAREST), f"f{row['frame']} {row['status']}"))
    srv._make_sheet(tiles, path, cols=4)
    return path


def _comparison_panel(case_id: str, records: list[dict[str, Any]], stacks: dict[str, Path]) -> Path:
    path = core_dir(case_id) / "srv_core_comparison_panel.png"
    valid = [row for row in records if row.get("crop_path")]
    if not valid:
        return path
    row = valid[len(valid) // 2]
    raw = cv2.imread(row["crop_path"])
    if raw is None:
        return path
    clahe, denoise, sharpen = srv._enhance(raw)
    avg = cv2.imread(str(stacks.get("srv_core_stack_average.png", "")))
    med = cv2.imread(str(stacks.get("srv_core_stack_median.png", "")))
    tiles = []
    for label, img in [
        ("core raw", raw),
        ("CLAHE", clahe),
        ("denoise", denoise),
        ("sharpen", sharpen),
        ("core average", avg if avg is not None else raw),
        ("core median", med if med is not None else raw),
    ]:
        tiles.append(srv._label(cv2.resize(img, (180, 180), interpolation=cv2.INTER_NEAREST), label))
    srv._make_sheet(tiles, path, cols=len(tiles))
    return path


def _quality_panel(case_id: str, records: list[dict[str, Any]]) -> Path:
    path = core_dir(case_id) / "srv_core_quality_panel.png"
    valid = [row for row in records if row.get("crop_path")]
    frames = np.array([row["frame"] for row in valid])
    sharp = np.array([row["sharpness_laplacian"] for row in valid])
    contrast = np.array([row["contrast_std"] for row in valid])
    mask_pixels = np.array([row["hud_mask_pixels"] for row in valid])
    selected = np.array([bool(row["selected_for_stack"]) for row in valid])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(frames, sharp, color="#2f80ed", linewidth=1)
    if len(frames):
        axes[0].scatter(frames[selected], sharp[selected], color="red", s=12, label="selected")
        axes[0].legend()
    axes[0].set_title("Core sharpness; selected frames marked")
    axes[1].plot(frames, contrast, color="#f28e2b", linewidth=1)
    axes[1].set_title("Core contrast")
    axes[2].plot(frames, mask_pixels, color="#8e44ad", linewidth=1)
    axes[2].set_title("HUD/artifact mask pixels inside tracked bbox")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_core_video(case_id: str, records: list[dict[str, Any]]) -> Path:
    raw = core_dir(case_id) / "srv_core_stabilized_crop_sequence_raw.mp4"
    final = core_dir(case_id) / "srv_core_stabilized_crop_sequence.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24, (CORE_SIZE, CORE_SIZE))
    for row in records:
        img = cv2.imread(row.get("crop_path", ""))
        if img is not None:
            writer.write(img)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(final)]
        subprocess.run(command, cwd=str(core_dir(case_id)), capture_output=True, text=True)
        if final.exists() and final.stat().st_size > 0:
            raw.unlink(missing_ok=True)
            return final
    return raw


def _core_metrics(case_id: str, records: list[dict[str, Any]], outputs: dict[str, Path]) -> dict[str, Any]:
    valid = [row for row in records if row.get("crop_path") and row["status"] != "CORE_NOT_FOUND"]
    low = [row for row in records if row["status"] == "LOW_CONFIDENCE"]
    not_found = [row for row in records if row["status"] == "CORE_NOT_FOUND"]
    sharp = np.array([row["sharpness_laplacian"] for row in valid], dtype=np.float32)
    contrast = np.array([row["contrast_std"] for row in valid], dtype=np.float32)
    lum = np.array([row["luminance_mean"] for row in valid], dtype=np.float32)
    sizes = np.array([[row["core_w"], row["core_h"]] for row in valid], dtype=np.float32) if valid else np.zeros((0, 2))
    hud = np.array([row["hud_mask_pixels"] for row in records], dtype=np.float32) if records else np.zeros(0)
    artifact_score = float(np.mean(hud) / max(1.0, 301 * 204)) if len(hud) else 0.0
    return {
        "case_id": case_id,
        "total_frames": len(records),
        "valid_core_frames": len(valid),
        "low_confidence_frames": len(low),
        "core_not_found_frames": len(not_found),
        "mean_core_crop_size": [float(np.mean(sizes[:, 0])) if len(sizes) else 0.0, float(np.mean(sizes[:, 1])) if len(sizes) else 0.0],
        "normalized_core_crop_size": [CORE_SIZE, CORE_SIZE],
        "mean_sharpness": float(np.mean(sharp)) if len(sharp) else 0.0,
        "mean_contrast": float(np.mean(contrast)) if len(contrast) else 0.0,
        "mean_luminance": float(np.mean(lum)) if len(lum) else 0.0,
        "hud_mask_pixels_mean": float(np.mean(hud)) if len(hud) else 0.0,
        "artifact_contamination_score": artifact_score,
        "generated_outputs": {name: str(path) for name, path in outputs.items()},
        "notes": [
            "Object-core SRV is for visual reconstruction only and does not replace tracking bboxes.",
            "HUD/cyan reticle and large overlay masks are ignored to reduce visual contamination.",
            "No generative model was used and no physical structure or origin is proven.",
        ],
    }


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    path = core_dir(case_id) / "srv_core_manifest.md"
    lines = [
        "# Object-Core SRV Manifest",
        "",
        f"Case: `{case_id}`",
        "",
        "The original bbox-level SRV remains available as context. Bbox-level stacks are marked `superseded_by_object_core_srv` for visual reconstruction if HUD/reticle contamination is present.",
        "",
        "| output | path | classification | caution |",
        "| --- | --- | --- | --- |",
    ]
    for name, out in outputs.items():
        classification = "interpretive" if "stack" in name or "enhanced" in name or "comparison" in name else "technical"
        if "contact_sheet_raw" in name or "sequence" in name:
            classification = "public_safe"
        lines.append(f"| `{name}` | `{out}` | `{classification}` | core crop derived inside validated bbox; does not replace track |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _append_core_report(case_id: str, metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report = case_report_dir(case_id) / "srv_analysis_report.md"
    existing = report.read_text(encoding="utf-8") if report.exists() else f"# SRV / Visual Reconstruction Report: {case_id}\n"
    lines = [
        existing.rstrip(),
        "",
        "## Object-Core SRV",
        "",
        "The bbox-level SRV is retained as tracking context. The recommended visual reconstruction output is now `object_core_srv`, which crops a smaller object-core ROI inside each validated tracking bbox.",
        "",
        "- Tracking was not changed.",
        "- Tracking bboxes were not changed.",
        "- No generative model was used.",
        "- No detail was invented.",
        "- HUD/reticle masking was applied only to avoid visual contamination from cyan reticles, black overlays and artificial frame markings.",
        "- Visual reconstruction is interpretive, not proof of physical structure or origin.",
        "",
        "### Core Metrics",
        "",
        f"- valid core frames: `{metrics['valid_core_frames']}`",
        f"- low confidence frames: `{metrics['low_confidence_frames']}`",
        f"- core not found frames: `{metrics['core_not_found_frames']}`",
        f"- mean core crop size: `{metrics['mean_core_crop_size']}`",
        f"- mean sharpness: `{metrics['mean_sharpness']:.4f}`",
        f"- mean contrast: `{metrics['mean_contrast']:.4f}`",
        f"- mean luminance: `{metrics['mean_luminance']:.4f}`",
        f"- hud mask pixels mean: `{metrics['hud_mask_pixels_mean']:.4f}`",
        f"- artifact contamination score: `{metrics['artifact_contamination_score']:.6f}`",
        "",
        "### Core Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _mark_bbox_manifest(case_id: str) -> None:
    manifest = srv.output_dir(case_id) / "srv_manifest.md"
    if not manifest.exists():
        return
    text = manifest.read_text(encoding="utf-8")
    note = "\n## Object-Core Supersession Note\n\nBbox-level SRV outputs remain as context. If reticle/HUD contamination is visible, `srv_stack_average.png`, `srv_stack_median.png`, `srv_stack_best_sharpness.png`, `srv_comparison_panel.png` and bbox-level stabilized outputs are `superseded_by_object_core_srv` for visual reconstruction.\n"
    if "Object-Core Supersession Note" not in text:
        manifest.write_text(text.rstrip() + note, encoding="utf-8")


def run_track_srv_core_analysis(case_id: str) -> dict[str, Any]:
    track, validation, rows = srv._require_valid_inputs(case_id)
    records, raw_tiles, enhanced_tiles = _collect_core(case_id, track, rows)
    outputs: dict[str, Path] = {
        "object_core_rois.csv": core_dir(case_id) / "object_core_rois.csv",
        "object_core_rois.json": core_dir(case_id) / "object_core_rois.json",
        "srv_core_contact_sheet_raw.png": core_dir(case_id) / "srv_core_contact_sheet_raw.png",
        "srv_core_contact_sheet_enhanced.png": core_dir(case_id) / "srv_core_contact_sheet_enhanced.png",
    }
    _write_core_timeseries(outputs["object_core_rois.csv"], records)
    write_json(outputs["object_core_rois.json"], {"case_id": case_id, "object_core_rois": records})
    srv._make_sheet(raw_tiles, outputs["srv_core_contact_sheet_raw.png"], cols=4)
    srv._make_sheet(enhanced_tiles, outputs["srv_core_contact_sheet_enhanced.png"], cols=4)
    stacks = _stack_core(case_id, records)
    outputs.update(stacks)
    outputs["srv_core_stabilized_sequence_panel.png"] = _sequence_panel(case_id, records)
    outputs["srv_core_comparison_panel.png"] = _comparison_panel(case_id, records, stacks)
    outputs["srv_core_quality_panel.png"] = _quality_panel(case_id, records)
    outputs["srv_core_stabilized_crop_sequence.mp4"] = _write_core_video(case_id, records)
    metrics = _core_metrics(case_id, records, outputs)
    outputs["srv_core_metrics.json"] = core_dir(case_id) / "srv_core_metrics.json"
    write_json(outputs["srv_core_metrics.json"], metrics)
    manifest = _write_manifest(case_id, outputs)
    outputs["srv_core_manifest.md"] = manifest
    report = _append_core_report(case_id, metrics, outputs)
    _mark_bbox_manifest(case_id)
    status_path = srv._update_case_status(
        case_id,
        {
            "srv_object_core_status": "complete",
            "srv_object_core_ready": True,
            "last_srv_object_core_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "srv_bbox_context_note": "bbox-level SRV retained as context; object-core SRV recommended for visual reconstruction",
            "srv_object_core_paths": {
                "output_dir": str(core_dir(case_id)),
                "metrics": str(outputs["srv_core_metrics.json"]),
                "csv": str(outputs["object_core_rois.csv"]),
                "json": str(outputs["object_core_rois.json"]),
                "report": str(report),
                "manifest": str(manifest),
                "video": str(outputs["srv_core_stabilized_crop_sequence.mp4"]),
                "panels": {name: str(path) for name, path in outputs.items() if name.endswith(".png")},
            },
        },
    )
    return {
        "case_id": case_id,
        "srv_object_core_ready": True,
        "output_dir": str(core_dir(case_id)),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "report": str(report),
        "case_status": str(status_path),
    }
