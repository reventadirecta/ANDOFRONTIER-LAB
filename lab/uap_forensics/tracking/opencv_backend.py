from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .backends import TrackingBackend


VALID_TRACK_STATUSES = {"tracked", "auto_recovered"}


def _bbox_from_request(request: dict[str, Any]) -> list[int]:
    prompt = request.get("object_prompt") or {}
    box = prompt.get("box")
    if not box:
        raise ValueError("object_prompt.box is required for the OpenCV fallback backend.")
    if isinstance(box, dict):
        return [int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])]
    if len(box) != 4:
        raise ValueError("object_prompt.box must be [x, y, w, h].")
    return [int(v) for v in box]


def _bbox_dict(bbox: list[int] | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    x, y, w, h = [int(v) for v in bbox]
    return {"x": x, "y": y, "w": w, "h": h}


def _bbox_list(bbox: dict[str, Any] | list[int] | tuple[int, int, int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return [int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])]
    return [int(v) for v in bbox]


def _centroid(bbox: list[int]) -> list[float]:
    x, y, w, h = bbox
    return [float(x + w / 2), float(y + h / 2)]


def _tracker() -> tuple[Any, str]:
    factories = [
        ("CSRT", "legacy", "TrackerCSRT_create"),
        ("CSRT", "cv2", "TrackerCSRT_create"),
        ("KCF", "legacy", "TrackerKCF_create"),
        ("KCF", "cv2", "TrackerKCF_create"),
        ("MIL", "legacy", "TrackerMIL_create"),
        ("MIL", "cv2", "TrackerMIL_create"),
    ]
    for label, namespace, name in factories:
        source = cv2.legacy if namespace == "legacy" and hasattr(cv2, "legacy") else cv2
        if hasattr(source, name):
            return getattr(source, name)(), f"opencv-{label.lower()}-{namespace}"
    raise RuntimeError("No OpenCV tracker is available. Tried CSRT, KCF and MIL in legacy/cv2 namespaces.")


def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _center(bbox: list[int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _crop_gray(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)


def _local_contrast(gray: np.ndarray, bbox: list[int]) -> tuple[float, str]:
    x, y, w, h = bbox
    pad = max(4, int(round(max(w, h) * 0.75)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    roi = gray[y : y + h, x : x + w]
    local = gray[y0:y1, x0:x1]
    if roi.size == 0 or local.size == 0:
        return 0.0, "unknown"
    roi_mean = float(np.mean(roi))
    local_mean = float(np.mean(local))
    local_std = float(np.std(local)) + 1.0
    delta = (roi_mean - local_mean) / local_std
    polarity = "white_hot" if delta >= 0 else "black_hot"
    return min(1.0, abs(delta) / 3.0), polarity


def _hud_overlap_score(bbox: list[int], width: int, height: int) -> float:
    x, y, w, h = bbox
    cx, cy = _center(bbox)
    frame_cx, frame_cy = width / 2, height / 2
    center_dist = _distance((cx, cy), (frame_cx, frame_cy)) / max(1.0, math.hypot(width, height))
    center_score = max(0.0, 1.0 - center_dist / 0.16)
    cross_band = 0.0
    if abs(cx - frame_cx) < max(8, width * 0.035) or abs(cy - frame_cy) < max(8, height * 0.035):
        cross_band = 0.65
    border_score = 0.75 if x < width * 0.035 or y < height * 0.035 or x + w > width * 0.965 or y + h > height * 0.965 else 0.0
    return max(center_score, cross_band, border_score)


def _line_hud_score(gray: np.ndarray, bbox: list[int], width: int, height: int) -> tuple[float, str]:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 1.0, "empty_candidate"
    aspect = w / max(1, h)
    line_score = 0.0
    reasons: list[str] = []
    if aspect > 5.5 or aspect < 0.18:
        line_score = max(line_score, 0.78)
        reasons.append("thin_linear_shape")
    crop = gray[y : y + h, x : x + w]
    if crop.size:
        edges = cv2.Canny(crop, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        if edge_density > 0.18 and (aspect > 3.0 or aspect < 0.33):
            line_score = max(line_score, min(1.0, edge_density * 3.0))
            reasons.append("linear_edge_density")
        mid_x = edges[:, edges.shape[1] // 2] if edges.shape[1] else np.array([])
        mid_y = edges[edges.shape[0] // 2, :] if edges.shape[0] else np.array([])
        if mid_x.size and mid_y.size:
            cross_density = (np.count_nonzero(mid_x) / mid_x.size + np.count_nonzero(mid_y) / mid_y.size) / 2
            if cross_density > 0.20 and _hud_overlap_score(bbox, width, height) > 0.55:
                line_score = max(line_score, 0.86)
                reasons.append("reticle_cross_pattern")
    return line_score, "+".join(reasons) or "shape_ok"


def _black_overlay_score(gray: np.ndarray, bbox: list[int], width: int, height: int) -> tuple[float, str]:
    x, y, w, h = bbox
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return 1.0, "empty_candidate"
    dark_ratio = float(np.mean(crop < 18))
    very_dark_ratio = float(np.mean(crop < 8))
    std = float(np.std(crop))
    aspect = w / max(1, h)
    area_ratio = (w * h) / max(1.0, width * height)
    near_video_overlay_band = (
        y < height * 0.18
        or y + h > height * 0.88
        or x < width * 0.08
        or x + w > width * 0.92
    )
    rectangular_label_shape = 1.7 <= aspect <= 9.5 and area_ratio >= 0.00035
    uniform_black_block = dark_ratio > 0.62 and std < 34.0
    score = 0.0
    reasons: list[str] = []
    if uniform_black_block:
        score = max(score, 0.72 + min(0.22, (dark_ratio - 0.62) * 0.6))
        reasons.append("uniform_black_block")
    if very_dark_ratio > 0.55 and rectangular_label_shape:
        score = max(score, 0.86)
        reasons.append("black_rectangular_label")
    if near_video_overlay_band and (dark_ratio > 0.45 or very_dark_ratio > 0.30):
        score = max(score, 0.82)
        reasons.append("overlay_band_dark_region")
    if near_video_overlay_band and rectangular_label_shape and dark_ratio > 0.35:
        score = max(score, 0.90)
        reasons.append("fixed_hud_label_geometry")
    return min(1.0, score), "+".join(reasons) or "not_black_overlay"


def _hud_rejection(
    frame: np.ndarray,
    bbox: list[int],
    previous: list[int],
    predicted: list[int],
    initial: list[int],
    recent_centers: list[tuple[float, float]],
    template_score: float | None = None,
) -> tuple[bool, float, str]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hud_overlap = _hud_overlap_score(bbox, width, height)
    line_score, line_reason = _line_hud_score(gray, bbox, width, height)
    black_score, black_reason = _black_overlay_score(gray, bbox, width, height)
    jump = _distance(_center(bbox), _center(predicted))
    previous_jump = _distance(_center(bbox), _center(previous))
    max_reasonable_jump = max(width, height) * 0.14
    area_ratio = (bbox[2] * bbox[3]) / max(1, initial[2] * initial[3])
    frame_center = (width / 2, height / 2)
    candidate_center = _center(bbox)
    initial_center = _center(initial)
    candidate_on_center_hud = (
        abs(candidate_center[0] - frame_center[0]) < max(14, width * 0.07)
        and abs(candidate_center[1] - frame_center[1]) < max(14, height * 0.07)
    )
    initial_was_not_center_hud = _distance(initial_center, frame_center) > max(width, height) * 0.12
    fixed_score = 0.0
    if len(recent_centers) >= 6:
        recent_motion = np.mean([_distance(recent_centers[i], recent_centers[i - 1]) for i in range(1, len(recent_centers))])
        if recent_motion > 1.5 and max(_distance(candidate_center, c) for c in recent_centers[-5:]) < 1.2:
            fixed_score = 0.75
    template_value = 1.0 if template_score is None else float(template_score)
    template_mismatch = template_score is not None and template_value < 0.16
    score = max(hud_overlap * 0.85, line_score, black_score, fixed_score)
    if black_score >= 0.86 and (previous_jump > max_reasonable_jump * 0.35 or hud_overlap > 0.35 or template_mismatch):
        return True, max(score, 0.90), black_reason
    if black_score >= 0.78 and template_mismatch and previous_jump > max_reasonable_jump * 0.18:
        return True, max(score, 0.86), f"{black_reason}+template_mismatch"
    if candidate_on_center_hud and initial_was_not_center_hud and hud_overlap > 0.72:
        return True, max(score, 0.88), "reticle_hud_overlap"
    if jump > max_reasonable_jump and hud_overlap > 0.45:
        score = max(score, 0.92)
        return True, score, "drift_to_hud_jump"
    if previous_jump > max_reasonable_jump * 1.25:
        return True, max(score, 0.82), "candidate_jump_too_large"
    if area_ratio < 0.25 or area_ratio > 4.0:
        return True, max(score, 0.75), "size_drift"
    if score >= 0.80:
        if black_score >= hud_overlap and black_score >= line_score:
            reason = black_reason
        else:
            reason = "reticle_hud_overlap" if hud_overlap >= line_score else line_reason
        return True, score, reason
    return False, score, "accepted_candidate"


def _prediction(previous: list[int], velocity: tuple[float, float], width: int, height: int) -> list[int]:
    x, y, w, h = previous
    return _clip_bbox([int(round(x + velocity[0])), int(round(y + velocity[1])), w, h], width, height)


def _template_score(frame_gray: np.ndarray, bbox: list[int], templates: list[np.ndarray]) -> float:
    if not templates:
        return 0.0
    crop = frame_gray[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    if crop.size == 0:
        return 0.0
    scores = []
    for template in templates[-4:]:
        resized = cv2.resize(crop, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        if float(np.std(resized)) < 1.0 or float(np.std(template)) < 1.0:
            scores.append(0.0)
        else:
            scores.append(float(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)[0][0]))
    return max(scores) if scores else 0.0


def _candidate_components(gray: np.ndarray, window: list[int], initial: list[int]) -> list[list[int]]:
    x, y, w, h = window
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return []
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    local_mean = float(np.mean(blur))
    local_std = float(np.std(blur)) + 1.0
    masks = [
        cv2.threshold(blur, local_mean + local_std * 1.2, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, local_mean - local_std * 1.2, 255, cv2.THRESH_BINARY_INV)[1],
    ]
    candidates: list[list[int]] = []
    init_area = max(1, initial[2] * initial[3])
    init_aspect = initial[2] / max(1, initial[3])
    for mask in masks:
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < init_area * 0.20 or area > init_area * 5.0:
                continue
            aspect = bw / max(1, bh)
            if aspect < init_aspect / 4.0 or aspect > init_aspect * 4.0:
                continue
            candidates.append(_clip_bbox([x + bx, y + by, bw, bh], gray.shape[1], gray.shape[0]))
    return candidates


def _recover_candidate(
    frame: np.ndarray,
    predicted: list[int],
    previous: list[int],
    initial: list[int],
    templates: list[np.ndarray],
    recent_centers: list[tuple[float, float]],
    expand: float,
) -> tuple[list[int] | None, float, str, dict[str, Any] | None]:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    px, py, pw, ph = predicted
    margin = int(round(max(pw, ph) * expand))
    window = _clip_bbox([px - margin, py - margin, pw + margin * 2, ph + margin * 2], width, height)
    best: tuple[float, list[int], str, float] | None = None
    for candidate in _candidate_components(gray, window, initial):
        template_match = _template_score(gray, candidate, templates)
        rejected, hud_score, reason = _hud_rejection(frame, candidate, previous, predicted, initial, recent_centers, template_match)
        if rejected:
            return None, 0.0, "recovery_candidate_rejected_hud", {
                "candidate_rejected": True,
                "reason": reason,
                "hud_overlap_score": round(hud_score, 3),
                "template_score": round(template_match, 3),
                "bbox": _bbox_dict(candidate),
            }
        contrast, polarity = _local_contrast(gray, candidate)
        proximity = max(0.0, 1.0 - _distance(_center(candidate), _center(predicted)) / max(1.0, max(width, height) * 0.25))
        area_ratio = (candidate[2] * candidate[3]) / max(1, initial[2] * initial[3])
        size_score = max(0.0, 1.0 - abs(math.log(max(0.05, area_ratio))))
        template = max(0.0, template_match)
        score = 0.32 * contrast + 0.24 * proximity + 0.24 * size_score + 0.20 * template
        if best is None or score > best[0]:
            best = (score, candidate, polarity, template)
    if best and best[0] >= 0.52:
        return best[1], min(0.82, best[0]), f"auto_recovered contrast/template polarity={best[2]} template={best[3]:.2f}", None
    return None, 0.0, "no_recovery_candidate", None


def _make_item(
    frame_idx: int,
    bbox: list[int] | None,
    confidence: float,
    status: str,
    method: str,
    reason: str,
    mask_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "frame": int(frame_idx),
        "bbox": _bbox_dict(bbox),
        "bbox_xywh": bbox,
        "centroid": _centroid(bbox) if bbox else None,
        "confidence": float(round(confidence, 4)),
        "status": status,
        "state": status,
        "method": method,
        "reason": reason,
        "mask_path": mask_path,
    }
    if extra:
        item.update(extra)
    return item


def _quality(track: list[dict[str, Any]], total: int, recovery_count: int, hud_rejected: int, reticle_rejected: int, template_skips: int, drift: bool) -> dict[str, Any]:
    confidence_values = [float(item.get("confidence", 0.0)) for item in track if item.get("status") != "waiting_for_object"]
    counts = {status: sum(1 for item in track if item.get("status") == status) for status in ["tracked", "auto_recovered", "low_confidence", "predicted_only", "lost"]}
    valid = counts["tracked"] + counts["auto_recovered"]
    usable_ratio = valid / max(1, total)
    lost_ratio = (counts["lost"] + counts["predicted_only"]) / max(1, total)
    if valid == 0 or lost_ratio > 0.50:
        recommendation = "tracking_failed"
    elif drift or counts["low_confidence"] > valid * 0.35 or counts["auto_recovered"] > valid * 0.35:
        recommendation = "tracking_needs_review"
    elif counts["auto_recovered"] or counts["low_confidence"]:
        recommendation = "tracking_usable_with_review"
    elif usable_ratio > 0.65:
        recommendation = "tracking_good"
    else:
        recommendation = "tracking_usable_with_review"
    return {
        "total_frames": total,
        "tracked_frames": counts["tracked"],
        "auto_recovered_frames": counts["auto_recovered"],
        "low_confidence_frames": counts["low_confidence"],
        "predicted_only_frames": counts["predicted_only"],
        "lost_frames": counts["lost"],
        "mean_confidence": round(float(np.mean(confidence_values)) if confidence_values else 0.0, 4),
        "min_confidence": round(float(np.min(confidence_values)) if confidence_values else 0.0, 4),
        "recovery_count": recovery_count,
        "hud_rejected_candidates": hud_rejected,
        "reticle_rejected_candidates": reticle_rejected,
        "template_updates_skipped_due_to_hud": template_skips,
        "drift_to_hud_detected": bool(drift),
        "recommendation": recommendation,
    }


class OpenCVBackend(TrackingBackend):
    name = "opencv"

    def track(
        self,
        case: dict[str, Any],
        request: dict[str, Any],
        output_dir: Path,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        first_frame = request.get("first_object_frame")
        if first_frame is None:
            raise ValueError("first_object_frame is required. Do not start tracking before the object appears.")
        first_frame = int(first_frame)
        initial_box = _bbox_from_request(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "track_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(case["video_path"]))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {case['video_path']}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or case.get("fps") or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        initial_box = _clip_bbox(initial_box, width, height)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read first_object_frame={first_frame}")

        tracker, tracker_name = _tracker()
        tracker.init(frame, tuple(initial_box))
        initial_template = _crop_gray(frame, initial_box)
        templates = [cv2.resize(initial_template, (max(8, initial_box[2]), max(8, initial_box[3])), interpolation=cv2.INTER_AREA)]
        items: list[dict[str, Any]] = []
        previous = initial_box
        velocity = (0.0, 0.0)
        recent_centers: list[tuple[float, float]] = []
        lost_streak = 0
        recovery_count = 0
        hud_rejected = 0
        reticle_rejected = 0
        template_skips = 0
        drift_to_hud = False
        first_lost_frame: int | None = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        progress_step = max(1, total // 100) if total else 1
        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if progress and (frame_idx == first_frame or frame_idx % progress_step == 0 or frame_idx == total - 1):
                progress({"stage": "tracking", "frame": frame_idx, "total": total, "percent": round((frame_idx / max(1, total - 1)) * 100, 1)})
            if frame_idx < first_frame:
                items.append(_make_item(frame_idx, None, 0.0, "waiting_for_object", "none", "before user-marked first object frame"))
                continue

            predicted = _prediction(previous, velocity, width, height)
            current: list[int] | None = None
            status = "tracked"
            confidence = 0.96 if frame_idx == first_frame else 0.72
            method = "initial_user_box" if frame_idx == first_frame else "opencv_tracker"
            reason = "initial user box" if frame_idx == first_frame else "tracker update"
            rejection: dict[str, Any] | None = None

            if frame_idx == first_frame:
                current = initial_box
            else:
                tracked, raw_box = tracker.update(frame)
                if tracked:
                    raw_current = _clip_bbox([int(round(v)) for v in raw_box], width, height)
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    template_match = _template_score(frame_gray, raw_current, templates)
                    rejected, hud_score, reject_reason = _hud_rejection(frame, raw_current, previous, predicted, initial_box, recent_centers, template_match)
                    if rejected:
                        hud_rejected += 1
                        if "reticle" in reject_reason or "cross" in reject_reason:
                            reticle_rejected += 1
                        if "hud" in reject_reason or "reticle" in reject_reason or "overlay" in reject_reason or "black" in reject_reason:
                            drift_to_hud = True
                        rejection = {
                            "candidate_rejected": True,
                            "reason": reject_reason,
                            "hud_overlap_score": round(hud_score, 3),
                            "template_score": round(template_match, 3),
                            "rejected_bbox": _bbox_dict(raw_current),
                        }
                    else:
                        contrast, polarity = _local_contrast(frame_gray, raw_current)
                        if contrast < 0.06:
                            rejection = {
                                "candidate_rejected": True,
                                "reason": "insufficient_local_contrast",
                                "hud_overlap_score": round(hud_score, 3),
                                "template_score": round(template_match, 3),
                                "rejected_bbox": _bbox_dict(raw_current),
                            }
                        else:
                            current = raw_current
                            confidence = min(0.92, 0.52 + contrast * 0.28 + max(0.0, template_match) * 0.12)
                            reason = f"tracker update polarity={polarity} contrast={contrast:.2f} template={template_match:.2f}"
                else:
                    rejection = {"candidate_rejected": True, "reason": "OpenCV tracker update failed", "hud_overlap_score": 0.0}

                if current is None:
                    expand = 2.0 + min(4.0, lost_streak * 0.75)
                    recovered, rec_conf, rec_reason, rec_rejection = _recover_candidate(frame, predicted, previous, initial_box, templates, recent_centers, expand)
                    if rec_rejection:
                        hud_rejected += 1
                        if "reticle" in rec_rejection.get("reason", ""):
                            reticle_rejected += 1
                        rec_reason_text = rec_rejection.get("reason", "")
                        drift_to_hud = drift_to_hud or any(marker in rec_reason_text for marker in ["hud", "reticle", "overlay", "black"])
                        rejection = {**(rejection or {}), "recovery_rejection": rec_rejection}
                    if recovered is not None:
                        current = recovered
                        status = "auto_recovered"
                        confidence = rec_conf
                        method = "contrast_template_recovery"
                        reason = rec_reason
                        recovery_count += 1
                        lost_streak = 0
                        tracker, tracker_name = _tracker()
                        tracker.init(frame, tuple(current))
                    elif lost_streak < 8:
                        current = predicted
                        status = "predicted_only" if lost_streak >= 2 else "low_confidence"
                        confidence = max(0.18, 0.40 - lost_streak * 0.05)
                        method = "motion_prediction"
                        reason = (rejection or {}).get("reason", "candidate rejected; using prediction")
                        lost_streak += 1
                    else:
                        current = None
                        status = "lost"
                        confidence = 0.0
                        method = "none"
                        reason = (rejection or {}).get("reason", "tracker lost; no safe recovery candidate")
                        lost_streak += 1
                        first_lost_frame = frame_idx if first_lost_frame is None else first_lost_frame
                else:
                    lost_streak = 0

            mask_path = None
            if current is not None:
                mask = np.zeros((height, width), dtype=np.uint8)
                x, y, w, h = current
                mask[y : y + h, x : x + w] = 255
                mask_path_obj = masks_dir / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path_obj), mask)
                mask_path = str(mask_path_obj)
                if status in {"tracked", "auto_recovered"}:
                    old_center = _center(previous)
                    new_center = _center(current)
                    velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                    previous = current
                    recent_centers.append(new_center)
                    recent_centers = recent_centers[-12:]
                    if status == "tracked" and confidence >= 0.68:
                        templates.append(_crop_gray(frame, current))
                        templates = templates[-6:]
                    else:
                        template_skips += 1
                else:
                    template_skips += 1

            extra = rejection or {}
            item = _make_item(frame_idx, current, confidence, status, method, reason, mask_path, extra)
            items.append(item)

        cap.release()
        quality = _quality(items, total, recovery_count, hud_rejected, reticle_rejected, template_skips, drift_to_hud)
        track_valid_for_review = quality["recommendation"] in {"tracking_good", "tracking_usable_with_review"} and quality["tracked_frames"] + quality["auto_recovered_frames"] > 0
        return {
            "case_id": case["case_id"],
            "backend": "opencv",
            "backend_requested": request.get("backend_requested"),
            "backend_used": "opencv",
            "tracker_backend": tracker_name,
            "backend_note": "OpenCV tracker initialized from the user box, with HUD/reticle rejection, drift prevention and local recovery. No HUD candidate is accepted as tracked.",
            "first_object_frame": first_frame,
            "initial_box": {"x": initial_box[0], "y": initial_box[1], "w": initial_box[2], "h": initial_box[3]},
            "do_not_track_before_first_frame": True,
            "track_status": "unvalidated",
            "frames_tracked": len(items),
            "summary": {
                **quality,
                "first_lost_frame": first_lost_frame,
                "track_valid_for_review": track_valid_for_review,
            },
            "tracking_quality": quality,
            "track": items,
            "warnings": ["HUD/reticle candidates are rejected. Tracks with low-confidence or recovered frames require human review."],
            "video": {
                "path": case["video_path"],
                "fps": fps,
                "frame_count": total,
                "width": width,
                "height": height,
            },
        }


def write_centroids_csv(track: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "confidence", "status"])
        writer.writeheader()
        for item in track.get("track", []):
            centroid = item.get("centroid")
            cx, cy = centroid if centroid else ("", "")
            writer.writerow({"frame": item["frame"], "x": cx, "y": cy, "confidence": item["confidence"], "status": item.get("status")})
