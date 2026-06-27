from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bbox_xywh(bbox: Any) -> list[int] | None:
    if not bbox:
        return None
    if isinstance(bbox, dict):
        return [int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])]
    return [int(v) for v in bbox]


def resize_for_grid(frame: np.ndarray, width: int = 360) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / max(1, w)
    return cv2.resize(frame, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def contact_sheet(frames: list[np.ndarray], labels: list[str], output: Path, cols: int = 4) -> None:
    if not frames:
        return
    thumbs = [resize_for_grid(frame) for frame in frames]
    cell_w = max(t.shape[1] for t in thumbs)
    cell_h = max(t.shape[0] for t in thumbs) + 34
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h + 28
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(sheet, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def draw_track_frame(frame: np.ndarray, frame_idx: int, track_by_frame: dict[int, dict[str, Any]], backend: str, first_frame: int) -> np.ndarray:
    out = frame.copy()
    if frame_idx < first_frame:
        state = "waiting for user-marked object"
        color = (160, 160, 160)
    elif frame_idx not in track_by_frame:
        state = "lost"
        color = (0, 0, 255)
    else:
        item = track_by_frame[frame_idx]
        state = item.get("status") or item.get("state", "tracked")
        colors = {
            "tracked": (0, 220, 0),
            "auto_recovered": (255, 160, 0),
            "low_confidence": (0, 220, 255),
            "predicted_only": (0, 220, 255),
            "lost": (0, 0, 255),
            "waiting_for_object": (160, 160, 160),
            "TRACKING_ACTIVE": (0, 220, 0),
            "LOW_CONFIDENCE": (0, 220, 255),
            "TRACK_LOST": (0, 0, 255),
            "WAITING_FOR_OBJECT": (160, 160, 160),
        }
        color = colors.get(state, (0, 0, 255))
        if state in {"WAITING_FOR_OBJECT", "waiting_for_object"}:
            color = (160, 160, 160)
        bbox = _bbox_xywh(item.get("bbox") or item.get("bbox_xywh"))
        if bbox is None:
            label = f"frame {frame_idx}  {backend.upper()}  {state}"
            cv2.putText(out, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            return out
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(3, out.shape[1] // 600))
        conf = float(item.get("confidence", 0.0))
        details = [f"conf {conf:.2f}"]
        if state == "auto_recovered":
            details.append("auto recovered")
        if state in {"low_confidence", "predicted_only"}:
            details.append("low confidence")
        if state == "lost":
            details.append("lost")
        if item.get("candidate_rejected") or item.get("recovery_rejection"):
            details.append("HUD rejected")
        cv2.putText(out, " | ".join(details), (max(4, x), max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    label = f"frame {frame_idx}  {backend.upper()}  {state}"
    cv2.putText(out, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return out


def write_overlay_video(case: dict[str, Any], track: dict[str, Any], output: Path) -> None:
    cap = cv2.VideoCapture(str(case["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {case['video_path']}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), min(30, fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not write overlay video: {output}")
    first_frame = int(track["first_object_frame"])
    track_by_frame = {int(item["frame"]): item for item in track.get("track", [])}
    backend = track.get("tracker_backend") or track.get("backend_used", "unknown")
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_track_frame(frame, frame_idx, track_by_frame, backend, first_frame))
    writer.release()
    cap.release()


def write_track_contact_sheet(case: dict[str, Any], track: dict[str, Any], output: Path) -> None:
    cap = cv2.VideoCapture(str(case["video_path"]))
    if not cap.isOpened():
        return
    track_frames = [int(item["frame"]) for item in track.get("track", [])]
    if not track_frames:
        cap.release()
        return
    picks = sorted(set(np.linspace(min(track_frames), max(track_frames), min(12, len(track_frames)), dtype=int).tolist()))
    track_by_frame = {int(item["frame"]): item for item in track.get("track", [])}
    frames = []
    labels = []
    for frame_idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(draw_track_frame(frame, frame_idx, track_by_frame, track.get("backend_used", "unknown"), int(track["first_object_frame"])))
            labels.append(f"f{frame_idx}")
    cap.release()
    contact_sheet(frames, labels, output, cols=3)
