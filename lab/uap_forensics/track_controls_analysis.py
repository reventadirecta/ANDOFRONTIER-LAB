from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


VALID_STATES = {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"}
REQUIRED_MESSAGE = "Controls analysis requires a human-validated track and dynamic ROIs."
CONTROL_TYPES = [
    "near_background",
    "far_background",
    "hud_artifact",
    "dark_region",
    "bright_region",
    "compression_noise",
    "random_background",
]
BACKGROUND_TYPES = [name for name in CONTROL_TYPES if name != "hud_artifact"]
VERSION = "Controls v0.2 clean masked"


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def dynamic_rois_csv(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "controls_analysis")


def controls_root(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "controls")


def artifact_mask_dir(case_id: str) -> Path:
    return ensure_dir(output_dir(case_id) / "artifact_masks")


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

            bbox = [i("bbox_x"), i("bbox_y"), i("bbox_w"), i("bbox_h")]
            expanded = [i("expanded_x"), i("expanded_y"), i("expanded_w"), i("expanded_h")]
            rows.append(
                {
                    "frame": int(raw["frame"]),
                    "status": raw.get("status", ""),
                    "bbox": bbox if all(v is not None for v in bbox) else None,
                    "expanded_bbox": expanded if all(v is not None for v in expanded) else None,
                    "confidence": float(raw.get("confidence") or 0),
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


def _clamp_bbox(box: list[int], width: int, height: int) -> list[int] | None:
    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    return frame[y : y + h, x : x + w]


def _gray(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop


def _overlap(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1, min(aw * ah, bw * bh))


def _hf_ratio(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(small))
    mag = np.log1p(np.abs(fft))
    yy, xx = np.indices(mag.shape)
    center = np.array([(mag.shape[0] - 1) / 2, (mag.shape[1] - 1) / 2])
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    high = radius > (0.35 * radius.max())
    return float(np.sum(mag[high]) / (np.sum(mag) + 1e-9))


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 50, 140)
    return float(np.mean(edges > 0))


def _make_artifact_mask(frame: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    black = (gray < 12).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    black_closed = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=1)
    large_black = np.zeros_like(black_closed)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(black_closed, 8)
    min_area = max(120, int(frame.shape[0] * frame.shape[1] * 0.0025))
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        rectangular = area / max(1, w * hh) > 0.55
        if area >= min_area and rectangular:
            large_black[labels == idx] = 255
    cyan = (((h >= 78) & (h <= 105) & (s > 70) & (v > 70))).astype(np.uint8) * 255
    white = ((gray > 235) & (s < 45)).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 80, 180)
    artificial_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    border = np.zeros_like(gray, dtype=np.uint8)
    border[:4, :] = 255
    border[-4:, :] = 255
    border[:, :4] = 255
    border[:, -4:] = 255
    raw = cv2.bitwise_or(large_black, cyan)
    raw = cv2.bitwise_or(raw, white)
    raw = cv2.bitwise_or(raw, cv2.bitwise_and(artificial_edges, cv2.dilate(raw, kernel, iterations=1)))
    raw = cv2.bitwise_or(raw, border)
    mask = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=1)
    return mask, {"large_black": large_black, "cyan": cyan, "white": white, "border": border}


def _candidate_grid(width: int, height: int, object_box: list[int]) -> list[list[int]]:
    ox, oy, w, h = object_box
    max_x = max(0, width - w)
    max_y = max(0, height - h)
    raw: list[list[int]] = []
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-3, -2, -1, 1, 2, 3]:
            raw.append([ox + dx * max(1, w // 2), oy + dy * max(1, h // 2), w, h])
    raw += [
        [0, 0, w, h],
        [max_x, 0, w, h],
        [0, max_y, w, h],
        [max_x, max_y, w, h],
        [max_x // 2, 0, w, h],
        [max_x // 2, max_y, w, h],
        [0, max_y // 2, w, h],
        [max_x, max_y // 2, w, h],
        [max_x // 2, max_y // 2, w, h],
    ]
    boxes: list[list[int]] = []
    seen = set()
    for x, y, bw, bh in raw:
        box = [max(0, min(max_x, int(x))), max(0, min(max_y, int(y))), bw, bh]
        key = tuple(box)
        if key not in seen:
            seen.add(key)
            boxes.append(box)
    return boxes


def _quality(frame: np.ndarray, artifact_mask: np.ndarray, object_box: list[int], box: list[int]) -> dict[str, Any]:
    crop = _crop(frame, box)
    gray = _gray(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_crop = _crop(artifact_mask, box)
    overlap = _overlap(box, object_box)
    artifact_ratio = float(np.mean(mask_crop > 0)) if mask_crop.size else 1.0
    black_ratio = float(np.mean(gray < 12)) if gray.size else 1.0
    cyan_ratio = float(np.mean((hsv[..., 0] >= 78) & (hsv[..., 0] <= 105) & (hsv[..., 1] > 70) & (hsv[..., 2] > 70))) if crop.size else 1.0
    x, y, w, h = box
    edge_distance = min(x, y, frame.shape[1] - (x + w), frame.shape[0] - (y + h))
    edge_score = float(max(0.0, min(1.0, edge_distance / max(1, min(w, h) * 0.5))))
    texture = float(gray.std()) if gray.size else 0.0
    lum = float(gray.mean()) if gray.size else 0.0
    reasons = []
    if overlap > 0.02:
        reasons.append("overlaps_object")
    if artifact_ratio > 0.005:
        reasons.append("artifact_mask_ratio_high")
    if black_ratio > 0.001:
        reasons.append("black_block_ratio_high")
    if cyan_ratio > 0.0:
        reasons.append("cyan_hud_ratio_high")
    if edge_score < 0.12:
        reasons.append("too_close_to_edge")
    valid = not reasons
    return {
        "box": box,
        "overlap_with_object_bbox": overlap,
        "artifact_mask_ratio": artifact_ratio,
        "black_block_ratio": black_ratio,
        "cyan_hud_ratio": cyan_ratio,
        "edge_distance_score": edge_score,
        "texture_score": texture,
        "luminance_mean": lum,
        "luminance_std": texture,
        "edge_density": _edge_density(gray),
        "valid_for_background": valid,
        "rejection_reason": ";".join(reasons) if reasons else "",
    }


def _pick_controls(frame: np.ndarray, artifact_mask: np.ndarray, object_box: list[int], width: int, height: int, rng: np.random.Generator) -> tuple[dict[str, list[int] | None], list[dict[str, Any]], dict[str, str]]:
    qualities = [_quality(frame, artifact_mask, object_box, box) for box in _candidate_grid(width, height, object_box)]
    valid = [q for q in qualities if q["valid_for_background"]]
    invalid = [q for q in qualities if not q["valid_for_background"]]
    ox, oy, ow, oh = object_box
    oc = np.array([ox + ow / 2, oy + oh / 2], dtype=np.float32)

    def center(q: dict[str, Any]) -> np.ndarray:
        x, y, w, h = q["box"]
        return np.array([x + w / 2, y + h / 2], dtype=np.float32)

    def dist(q: dict[str, Any]) -> float:
        return float(np.linalg.norm(center(q) - oc))

    def choose(items: list[dict[str, Any]], key) -> list[int] | None:
        return min(items, key=key)["box"] if items else None

    controls: dict[str, list[int] | None] = {name: None for name in CONTROL_TYPES}
    status: dict[str, str] = {name: "not_available" for name in CONTROL_TYPES}
    controls["near_background"] = choose(valid, lambda q: abs(dist(q) - max(ow, oh) * 1.2))
    controls["far_background"] = max(valid, key=dist)["box"] if valid else None
    dark_pool = [q for q in valid if q["black_block_ratio"] < 0.08 and q["luminance_mean"] < 95]
    controls["dark_region"] = choose(dark_pool, lambda q: q["luminance_mean"])
    bright_pool = [q for q in valid if q["cyan_hud_ratio"] < 0.001 and q["artifact_mask_ratio"] < 0.04 and q["luminance_mean"] > 90]
    controls["bright_region"] = max(bright_pool, key=lambda q: q["luminance_mean"])["box"] if bright_pool else None
    flat_pool = [q for q in valid if 4 <= q["texture_score"] <= 38]
    controls["compression_noise"] = choose(flat_pool, lambda q: q["texture_score"] + q["edge_density"] * 100)
    controls["random_background"] = valid[int(rng.integers(0, len(valid)))]["box"] if valid else None
    hud_pool = [q for q in invalid if "artifact_mask_ratio_high" in q["rejection_reason"] or "black_block_ratio_high" in q["rejection_reason"] or "cyan_hud_ratio_high" in q["rejection_reason"]]
    controls["hud_artifact"] = max(hud_pool, key=lambda q: q["artifact_mask_ratio"] + q["black_block_ratio"] + q["cyan_hud_ratio"] * 50)["box"] if hud_pool else None
    for name, box in controls.items():
        if box is not None:
            status[name] = "clean" if name != "hud_artifact" else "artifact_isolated"
    return controls, qualities, status


def _make_tile(crop: np.ndarray | None, label: str, size: tuple[int, int] = (190, 140), missing: bool = False) -> np.ndarray:
    if missing or crop is None or crop.size == 0:
        tile = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
        cv2.putText(tile, "MISSING", (22, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        tile = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _write_sheet(tiles: list[np.ndarray], path: Path, cols: int = 4) -> None:
    if not tiles:
        return
    h = max(tile.shape[0] for tile in tiles)
    w = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + tile.shape[0], c * w : c * w + tile.shape[1]] = tile
    cv2.imwrite(str(path), sheet)


def _flow_mean(prev: np.ndarray | None, current: np.ndarray | None) -> float:
    if prev is None or current is None or prev.shape != current.shape:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(prev, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())


def _clean_previous_generated(case_id: str) -> None:
    root = controls_root(case_id)
    for name in CONTROL_TYPES:
        folder = ensure_dir(root / name)
        for file in folder.glob("*.png"):
            file.unlink(missing_ok=True)
    mask_folder = artifact_mask_dir(case_id)
    for file in mask_folder.glob("*.png"):
        file.unlink(missing_ok=True)


def _preserve_v01(case_id: str) -> None:
    out = output_dir(case_id)
    legacy = out / "legacy_controls_v0_1"
    if legacy.exists() or not (out / "controls_metrics.json").exists():
        return
    ensure_dir(legacy)
    for name in [
        "controls_metrics.json",
        "controls_timeseries.csv",
        "controls_summary_panel.png",
        "controls_luminance_panel.png",
        "controls_thermal_panel.png",
        "controls_spectral_panel.png",
        "controls_motion_panel.png",
        "controls_contact_sheet.png",
        "controls_manifest.md",
    ]:
        src = out / name
        if src.exists():
            shutil.copy2(src, legacy / name)


def _collect(track: dict[str, Any], rows: list[dict[str, Any]], case_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[np.ndarray], list[dict[str, Any]], list[np.ndarray]]:
    cap, fps, total, width, height = _open_video(track)
    active = [row for row in rows if row["status"] in VALID_STATES and row.get("expanded_bbox")]
    _preserve_v01(case_id)
    _clean_previous_generated(case_id)
    root = controls_root(case_id)
    mask_root = artifact_mask_dir(case_id)
    series: list[dict[str, Any]] = []
    counts = {name: 0 for name in CONTROL_TYPES}
    clean_counts = {name: 0 for name in BACKGROUND_TYPES}
    fallback_count = 0
    hud_leakage = 0
    artifact_samples: list[float] = []
    contact_tiles: list[np.ndarray] = []
    mask_tiles: list[np.ndarray] = []
    rejection_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    prev_object: np.ndarray | None = None
    prev_bg: np.ndarray | None = None
    pick_every = max(1, len(active) // 8) if active else 1
    save_every = max(1, len(active) // 30) if active else 1
    next_frame_pos: int | None = None
    for pos, row in enumerate(active):
        frame_idx = int(row["frame"])
        if next_frame_pos != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        next_frame_pos = frame_idx + 1
        if not ok or frame is None:
            continue
        object_box = _clamp_bbox(row["expanded_bbox"], width, height)
        if object_box is None:
            continue
        artifact_mask, parts = _make_artifact_mask(frame)
        object_crop = _crop(frame, object_box)
        object_gray = cv2.resize(_gray(object_crop), (96, 96), interpolation=cv2.INTER_AREA)
        controls, qualities, statuses = _pick_controls(frame, artifact_mask, object_box, width, height, rng)
        crop_values: dict[str, np.ndarray | None] = {}
        control_artifacts: dict[str, float] = {}
        for q in qualities:
            rejection_rows.append(
                {
                    "frame": frame_idx,
                    "candidate_x": q["box"][0],
                    "candidate_y": q["box"][1],
                    "candidate_w": q["box"][2],
                    "candidate_h": q["box"][3],
                    "overlap_with_object_bbox": q["overlap_with_object_bbox"],
                    "artifact_mask_ratio": q["artifact_mask_ratio"],
                    "black_block_ratio": q["black_block_ratio"],
                    "cyan_hud_ratio": q["cyan_hud_ratio"],
                    "edge_distance_score": q["edge_distance_score"],
                    "texture_score": q["texture_score"],
                    "luminance_mean": q["luminance_mean"],
                    "luminance_std": q["luminance_std"],
                    "valid_for_background": q["valid_for_background"],
                    "rejection_reason": q["rejection_reason"],
                }
            )
        for control_type, box in controls.items():
            crop_values[control_type] = _crop(frame, box) if box else None
            if box is not None:
                q = _quality(frame, artifact_mask, object_box, box)
                control_artifacts[control_type] = q["artifact_mask_ratio"]
                counts[control_type] += 1
                if control_type in BACKGROUND_TYPES and statuses[control_type] == "clean":
                    clean_counts[control_type] += 1
                if control_type in BACKGROUND_TYPES and q["artifact_mask_ratio"] > 0.08:
                    hud_leakage += 1
                if pos % save_every == 0:
                    cv2.imwrite(str(root / control_type / f"frame_{frame_idx:06d}.png"), crop_values[control_type])
            else:
                fallback_count += 1
        if pos % save_every == 0:
            debug = frame.copy()
            overlay = np.zeros_like(debug)
            overlay[artifact_mask > 0] = (0, 0, 255)
            debug = cv2.addWeighted(debug, 0.72, overlay, 0.28, 0)
            x, y, w, h = object_box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(str(mask_root / f"artifact_mask_frame_{frame_idx:06d}.png"), debug)

        def lum(name: str) -> float:
            crop = crop_values.get(name)
            return float(_gray(crop).mean()) if crop is not None and crop.size else float("nan")

        def hf(name: str) -> float:
            crop = crop_values.get(name)
            return _hf_ratio(_gray(crop)) if crop is not None and crop.size else float("nan")

        def resized_gray(name: str) -> np.ndarray | None:
            crop = crop_values.get(name)
            return cv2.resize(_gray(crop), (96, 96), interpolation=cv2.INTER_AREA) if crop is not None and crop.size else None

        bg_gray = resized_gray("near_background")
        if bg_gray is None:
            bg_gray = resized_gray("far_background")
        object_flow = _flow_mean(prev_object, object_gray)
        bg_flow = _flow_mean(prev_bg, bg_gray)
        prev_object = object_gray
        prev_bg = bg_gray
        near_lum = lum("near_background")
        far_lum = lum("far_background")
        bg_hf_values = [hf("near_background"), hf("far_background"), hf("compression_noise")]
        bg_hf = np.nanmean(bg_hf_values)
        object_hf = _hf_ratio(object_gray)
        object_lum = float(object_gray.mean())
        bg_ir = float(np.nanmean([near_lum, far_lum])) if np.isfinite(np.nanmean([near_lum, far_lum])) else 0.0
        artifact_mean = float(np.nanmean([control_artifacts.get(name, np.nan) for name in BACKGROUND_TYPES]))
        artifact_samples.append(artifact_mean if np.isfinite(artifact_mean) else 1.0)
        control_status = "clean" if all(statuses[name] == "clean" for name in ["near_background", "far_background", "compression_noise"]) else "partial_or_missing"
        row_out = {
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps else 0.0,
            "object_luminance": object_lum,
            "near_background_luminance": near_lum,
            "far_background_luminance": far_lum,
            "hud_luminance": lum("hud_artifact"),
            "dark_region_luminance": lum("dark_region"),
            "bright_region_luminance": lum("bright_region"),
            "compression_noise_luminance": lum("compression_noise"),
            "object_high_frequency_ratio": object_hf,
            "background_high_frequency_ratio": float(bg_hf) if np.isfinite(bg_hf) else 0.0,
            "object_ir_intensity": object_lum,
            "background_ir_intensity": bg_ir,
            "object_flow_mean": object_flow,
            "background_flow_mean": bg_flow,
            "object_vs_near_delta": object_lum - near_lum if np.isfinite(near_lum) else 0.0,
            "object_vs_far_delta": object_lum - far_lum if np.isfinite(far_lum) else 0.0,
            "control_status": control_status,
        }
        series.append(row_out)
        if pos % pick_every == 0 and len(contact_tiles) < 64:
            contact_tiles.append(_make_tile(object_crop, f"object f{frame_idx}"))
            for name in CONTROL_TYPES:
                crop = crop_values.get(name)
                contact_tiles.append(_make_tile(crop, name.replace("_", " "), missing=crop is None))
            mask_vis = cv2.cvtColor(artifact_mask, cv2.COLOR_GRAY2BGR)
            mask_tiles.append(_make_tile(mask_vis, f"artifact mask f{frame_idx}"))
    cap.release()
    metrics = _metrics(track, total, series, counts, clean_counts, artifact_samples, fallback_count, hud_leakage, rejection_rows)
    return series, metrics, contact_tiles, rejection_rows, mask_tiles


def _mean(series: list[dict[str, Any]], key: str) -> float:
    values = np.array([row[key] for row in series if np.isfinite(row.get(key, float("nan")))], dtype=np.float32)
    return float(values.mean()) if len(values) else 0.0


def _similarity(a: float, b: float) -> float:
    return float(max(0.0, 1.0 - abs(a - b) / max(1.0, abs(a), abs(b))))


def _metrics(track: dict[str, Any], total: int, series: list[dict[str, Any]], counts: dict[str, int], clean_counts: dict[str, int], artifact_samples: list[float], fallback_count: int, hud_leakage: int, rejection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    object_lum = _mean(series, "object_luminance")
    near_lum = _mean(series, "near_background_luminance")
    far_lum = _mean(series, "far_background_luminance")
    bg_lum = (near_lum + far_lum) / 2.0 if near_lum or far_lum else 0.0
    object_hf = _mean(series, "object_high_frequency_ratio")
    bg_hf = _mean(series, "background_high_frequency_ratio")
    object_flow = _mean(series, "object_flow_mean")
    bg_flow = _mean(series, "background_flow_mean")
    hud_lum = _mean(series, "hud_luminance")
    comp_lum = _mean(series, "compression_noise_luminance")
    missing = [name for name, count in counts.items() if count == 0]
    background_similarity = _similarity(object_lum, bg_lum)
    compression_similarity = (_similarity(object_lum, comp_lum) + _similarity(object_hf, bg_hf)) / 2.0 if comp_lum else 0.0
    hud_similarity = _similarity(object_lum, hud_lum) if hud_lum else 0.0
    valid_frames = len(series)
    bg_slots = max(1, valid_frames * len(BACKGROUND_TYPES))
    fallback_rate = fallback_count / max(1, valid_frames * len(CONTROL_TYPES))
    hud_leakage_rate = hud_leakage / bg_slots
    artifact_rate = float(np.mean(artifact_samples)) if artifact_samples else 1.0
    clean_near = clean_counts["near_background"] / max(1, valid_frames)
    clean_far = clean_counts["far_background"] / max(1, valid_frames)
    clean_compression = clean_counts["compression_noise"] / max(1, valid_frames)
    accepted = sum(1 for row in rejection_rows if row["valid_for_background"])
    rejected = sum(1 for row in rejection_rows if not row["valid_for_background"])
    completeness = 1.0 - len(missing) / len(CONTROL_TYPES)
    cleanliness = max(0.0, 1.0 - artifact_rate * 2.5 - hud_leakage_rate * 2.0 - fallback_rate)
    key_clean = (clean_near + clean_far + clean_compression) / 3.0
    validity = float(max(0.0, min(0.98, completeness * cleanliness * key_clean)))
    return {
        "case_id": track.get("case_id"),
        "controls_version": VERSION,
        "total_frames": int(total or track.get("summary", {}).get("total_frames") or valid_frames),
        "valid_tracked_frames": valid_frames,
        "controls_generated_per_type": counts,
        "missing_control_types": missing,
        "object_luminance_mean": object_lum,
        "near_background_luminance_mean": near_lum,
        "far_background_luminance_mean": far_lum,
        "object_vs_near_background_delta_luminance": object_lum - near_lum,
        "object_vs_far_background_delta_luminance": object_lum - far_lum,
        "object_ir_intensity_mean": object_lum,
        "background_ir_intensity_mean": bg_lum,
        "object_vs_background_delta_ir_intensity": object_lum - bg_lum,
        "object_high_frequency_ratio": object_hf,
        "background_high_frequency_ratio": bg_hf,
        "object_vs_background_delta_high_frequency_ratio": object_hf - bg_hf,
        "object_motion_flow_mean": object_flow,
        "background_motion_flow_mean": bg_flow,
        "object_vs_background_delta_motion_flow": object_flow - bg_flow,
        "hud_similarity_score": hud_similarity,
        "compression_similarity_score": compression_similarity,
        "background_similarity_score": background_similarity,
        "control_validity_score": validity,
        "artifact_contamination_rate": artifact_rate,
        "clean_near_background_ratio": clean_near,
        "clean_far_background_ratio": clean_far,
        "clean_compression_control_ratio": clean_compression,
        "hud_leakage_rate": hud_leakage_rate,
        "fallback_control_rate": fallback_rate,
        "rejected_control_candidates": rejected,
        "accepted_control_candidates": accepted,
        "notes": [
            "Controls v0.2 uses artifact masks to separate HUD/cyan/black overlays from clean background controls.",
            "HUD artifacts are isolated in hud_artifact_control and excluded from normal background controls.",
            "Controls compare visual/IR signal behavior; they do not determine origin.",
        ],
    }


def _write_timeseries(path: Path, series: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "object_luminance",
        "near_background_luminance",
        "far_background_luminance",
        "hud_luminance",
        "dark_region_luminance",
        "bright_region_luminance",
        "compression_noise_luminance",
        "object_high_frequency_ratio",
        "background_high_frequency_ratio",
        "object_ir_intensity",
        "background_ir_intensity",
        "object_flow_mean",
        "background_flow_mean",
        "object_vs_near_delta",
        "object_vs_far_delta",
        "control_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(series)


def _write_rejections(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "candidate_x",
        "candidate_y",
        "candidate_w",
        "candidate_h",
        "overlap_with_object_bbox",
        "artifact_mask_ratio",
        "black_block_ratio",
        "cyan_hud_ratio",
        "edge_distance_score",
        "texture_score",
        "luminance_mean",
        "luminance_std",
        "valid_for_background",
        "rejection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _series_array(series: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([row.get(key, np.nan) for row in series], dtype=np.float32)


def _line_panel(path: Path, title: str, series: list[dict[str, Any]], items: list[tuple[str, str, str]]) -> None:
    frames = _series_array(series, "frame")
    fig, ax = plt.subplots(figsize=(12, 5))
    for key, label, color in items:
        ax.plot(frames, _series_array(series, key), label=label, color=color)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary_panel(path: Path, metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(["object", "near", "far"], [metrics["object_luminance_mean"], metrics["near_background_luminance_mean"], metrics["far_background_luminance_mean"]], color=["#e15759", "#4e79a7", "#59a14f"])
    axes[0].set_title("Clean luminance controls")
    axes[1].bar(["artifact", "hud leak", "fallback"], [metrics["artifact_contamination_rate"], metrics["hud_leakage_rate"], metrics["fallback_control_rate"]], color=["#e15759", "#af7aa1", "#edc948"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Contamination / fallback")
    axes[2].bar(["near", "far", "compression", "validity"], [metrics["clean_near_background_ratio"], metrics["clean_far_background_ratio"], metrics["clean_compression_control_ratio"], metrics["control_validity_score"]], color=["#4e79a7", "#59a14f", "#76b7b2", "#f28e2b"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Clean control ratios")
    fig.suptitle(f"{VERSION} Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _quality_panel(path: Path, metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["accepted", "rejected"], [metrics["accepted_control_candidates"], metrics["rejected_control_candidates"]], color=["#59a14f", "#e15759"])
    axes[0].set_title("Candidate validation")
    axes[1].bar(["artifact", "hud leakage", "fallback", "validity"], [metrics["artifact_contamination_rate"], metrics["hud_leakage_rate"], metrics["fallback_control_rate"], metrics["control_validity_score"]], color=["#e15759", "#af7aa1", "#edc948", "#59a14f"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Control quality scores")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(case_id: str, validation: dict[str, Any], metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report = case_report_dir(case_id) / "controls_report.md"
    lines = [
        f"# Controls Track-Based Report: {case_id}",
        "",
        f"Version: `{VERSION}`",
        "",
        "## Scope",
        "",
        "Controls compare the human-validated tracked object ROI against same-size controls after masking HUD/cyan/black overlay artifacts.",
        "",
        "Controls compare visual/IR signal behavior; they do not determine origin.",
        "",
        "## Source",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Artifact Masking",
        "",
        "- Large black boxes, cyan reticles, white overlays, borders and adjacent artificial edges are masked.",
        "- HUD/artifact controls may contain these regions.",
        "- Near/far/dark/bright/compression/random background controls reject these regions.",
        "",
        "## Controls Generated",
        "",
    ]
    for name, count in metrics["controls_generated_per_type"].items():
        lines.append(f"- {name}: `{count}`")
    lines += [
        "",
        f"- missing controls: `{', '.join(metrics['missing_control_types']) if metrics['missing_control_types'] else 'none'}`",
        f"- control validity score: `{metrics['control_validity_score']:.4f}`",
        f"- artifact contamination rate: `{metrics['artifact_contamination_rate']:.4f}`",
        f"- HUD leakage rate: `{metrics['hud_leakage_rate']:.4f}`",
        f"- fallback control rate: `{metrics['fallback_control_rate']:.4f}`",
        f"- accepted candidates: `{metrics['accepted_control_candidates']}`",
        f"- rejected candidates: `{metrics['rejected_control_candidates']}`",
        "",
        "## Comparison Metrics",
        "",
        f"- object luminance mean: `{metrics['object_luminance_mean']:.4f}`",
        f"- near background luminance mean: `{metrics['near_background_luminance_mean']:.4f}`",
        f"- far background luminance mean: `{metrics['far_background_luminance_mean']:.4f}`",
        f"- object vs near delta luminance: `{metrics['object_vs_near_background_delta_luminance']:.4f}`",
        f"- object vs far delta luminance: `{metrics['object_vs_far_background_delta_luminance']:.4f}`",
        f"- object vs background delta IR intensity: `{metrics['object_vs_background_delta_ir_intensity']:.4f}`",
        f"- object high-frequency ratio: `{metrics['object_high_frequency_ratio']:.6f}`",
        f"- background high-frequency ratio: `{metrics['background_high_frequency_ratio']:.6f}`",
        f"- object motion flow mean: `{metrics['object_motion_flow_mean']:.6f}`",
        f"- background motion flow mean: `{metrics['background_motion_flow_mean']:.6f}`",
        f"- HUD similarity score: `{metrics['hud_similarity_score']:.4f}`",
        f"- compression similarity score: `{metrics['compression_similarity_score']:.4f}`",
        f"- background similarity score: `{metrics['background_similarity_score']:.4f}`",
        "",
        "## Technical Conclusion",
        "",
    ]
    if metrics["control_validity_score"] < 0.45:
        lines.append("The clean baseline is still weak; use controls as diagnostic context and prefer manual review for interpretation.")
    elif metrics["artifact_contamination_rate"] > 0.08 or metrics["hud_leakage_rate"] > 0.02:
        lines.append("The baseline is usable with caution; some residual artifact contamination may remain.")
    else:
        lines.append("The clean masked baseline is usable for comparing object signal against local/background controls, without making an origin claim.")
    lines += [
        "",
        "## Limitations",
        "",
        "- Artifact masking is heuristic and must be visually reviewed.",
        "- Clean controls may be unavailable in frames dominated by HUD/borders/overlays.",
        "- Brightness is not used to redetect the object.",
        "- No PCA, autoencoder, generative model, ROI automation or origin claim is used.",
        "",
        "## Supersession",
        "",
        "`controls_v1_superseded_by_controls_v0_2_clean_masked`: previous controls are preserved in `legacy_controls_v0_1` when available.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "clean_controls_manifest.md"
    classes = {
        "controls_metrics.json": "technical",
        "controls_timeseries.csv": "technical",
        "controls_summary_panel.png": "public_safe",
        "controls_luminance_panel.png": "technical",
        "controls_thermal_panel.png": "technical",
        "controls_spectral_panel.png": "technical",
        "controls_motion_panel.png": "technical",
        "controls_contact_sheet.png": "public_safe",
        "artifact_mask_debug_panel.png": "debug",
        "controls_rejection_report.csv": "debug",
        "controls_quality_panel.png": "technical",
        "controls_report.md": "interpretive",
    }
    lines = [
        "# Clean Controls Manifest",
        "",
        f"Case: `{case_id}`",
        f"Version: `{VERSION}`",
        "",
        "`controls_v1_superseded_by_controls_v0_2_clean_masked`",
        "",
        "| output | path | classification | caution |",
        "| --- | --- | --- | --- |",
    ]
    for name, path in outputs.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | artifact-masked controls only; no origin claim |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_controls_analysis(case_id: str) -> dict[str, Any]:
    _track, validation, rows = _require_valid_inputs(case_id)
    track = read_json(track_path(case_id))
    out = output_dir(case_id)
    series, metrics, contact_tiles, rejection_rows, mask_tiles = _collect(track, rows, case_id)
    outputs: dict[str, Path] = {
        "controls_metrics.json": out / "controls_metrics.json",
        "controls_timeseries.csv": out / "controls_timeseries.csv",
        "controls_summary_panel.png": out / "controls_summary_panel.png",
        "controls_luminance_panel.png": out / "controls_luminance_panel.png",
        "controls_thermal_panel.png": out / "controls_thermal_panel.png",
        "controls_spectral_panel.png": out / "controls_spectral_panel.png",
        "controls_motion_panel.png": out / "controls_motion_panel.png",
        "controls_contact_sheet.png": out / "controls_contact_sheet.png",
        "artifact_mask_debug_panel.png": out / "artifact_mask_debug_panel.png",
        "controls_rejection_report.csv": out / "controls_rejection_report.csv",
        "controls_quality_panel.png": out / "controls_quality_panel.png",
    }
    write_json(outputs["controls_metrics.json"], metrics)
    _write_timeseries(outputs["controls_timeseries.csv"], series)
    _write_rejections(outputs["controls_rejection_report.csv"], rejection_rows)
    _summary_panel(outputs["controls_summary_panel.png"], metrics)
    _line_panel(outputs["controls_luminance_panel.png"], "Luminance: object vs clean controls", series, [("object_luminance", "object", "#e15759"), ("near_background_luminance", "near bg", "#4e79a7"), ("far_background_luminance", "far bg", "#59a14f")])
    _line_panel(outputs["controls_thermal_panel.png"], "Relative IR intensity: object vs clean background", series, [("object_ir_intensity", "object", "#e15759"), ("background_ir_intensity", "background", "#4e79a7")])
    _line_panel(outputs["controls_spectral_panel.png"], "High-frequency ratio: object vs clean background", series, [("object_high_frequency_ratio", "object", "#e15759"), ("background_high_frequency_ratio", "background", "#4e79a7")])
    _line_panel(outputs["controls_motion_panel.png"], "Optical flow: object vs clean background", series, [("object_flow_mean", "object", "#e15759"), ("background_flow_mean", "background", "#4e79a7")])
    _quality_panel(outputs["controls_quality_panel.png"], metrics)
    _write_sheet(contact_tiles, outputs["controls_contact_sheet.png"], cols=4)
    _write_sheet(mask_tiles, outputs["artifact_mask_debug_panel.png"], cols=4)
    report = _write_report(case_id, validation, metrics, outputs)
    outputs["controls_report.md"] = report
    manifest = _write_manifest(case_id, outputs)
    outputs["clean_controls_manifest.md"] = manifest
    legacy_manifest = out / "controls_manifest.md"
    legacy_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    outputs["controls_manifest.md"] = legacy_manifest
    status_path = _update_case_status(
        case_id,
        {
            "controls_analysis_status": "complete",
            "controls_analysis_ready": True,
            "controls_analysis_version": VERSION,
            "last_controls_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "control_validity_score": metrics["control_validity_score"],
            "artifact_contamination_rate": metrics["artifact_contamination_rate"],
            "hud_leakage_rate": metrics["hud_leakage_rate"],
            "fallback_control_rate": metrics["fallback_control_rate"],
            "controls_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(outputs["controls_metrics.json"]),
                "csv": str(outputs["controls_timeseries.csv"]),
                "rejection_report": str(outputs["controls_rejection_report.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "panels": {
                    "summary": str(outputs["controls_summary_panel.png"]),
                    "luminance": str(outputs["controls_luminance_panel.png"]),
                    "thermal": str(outputs["controls_thermal_panel.png"]),
                    "spectral": str(outputs["controls_spectral_panel.png"]),
                    "motion": str(outputs["controls_motion_panel.png"]),
                    "contact_sheet": str(outputs["controls_contact_sheet.png"]),
                    "artifact_mask": str(outputs["artifact_mask_debug_panel.png"]),
                    "quality": str(outputs["controls_quality_panel.png"]),
                },
            },
        },
    )
    return {
        "case_id": case_id,
        "controls_analysis_ready": True,
        "controls_analysis_version": VERSION,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "case_status": str(status_path),
    }
