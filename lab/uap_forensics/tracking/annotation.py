from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from uap_forensics.io import read_json, write_json

from .interactive_config import interactive_output_dir, interactive_request_path, prepare_interactive_track, _load_case


WINDOW_NAME = "interactive track annotation"


def _load_or_prepare_request(case_id: str) -> dict[str, Any]:
    path = interactive_request_path(case_id)
    if not path.exists():
        prepare_interactive_track(case_id)
    return read_json(path)


def _hint_frame(case_id: str, request: dict[str, Any], total_frames: int) -> int:
    if request.get("first_object_frame") is not None:
        return max(0, min(total_frames - 1, int(request["first_object_frame"])))
    hint_path = Path("data") / "cases" / case_id / "tracking_target_hint.json"
    if hint_path.exists():
        hint = read_json(hint_path)
        if hint.get("frame") is not None:
            return max(0, min(total_frames - 1, int(hint["frame"])))
    candidates = request.get("possible_appearance_frames") or []
    if candidates:
        frames = [int(item["frame"]) for item in candidates if item.get("frame") is not None]
        if frames:
            return max(0, min(total_frames - 1, frames[len(frames) // 2]))
    return 0


def _scale_frame(frame, max_width: int = 1280, max_height: int = 860):
    h, w = frame.shape[:2]
    scale = min(max_width / max(1, w), max_height / max(1, h), 1.0)
    if scale == 1.0:
        return frame.copy(), scale
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _box_to_display(box: dict[str, int] | None, scale: float) -> tuple[int, int, int, int] | None:
    if not box:
        return None
    return (
        int(box["x"] * scale),
        int(box["y"] * scale),
        int((box["x"] + box["w"]) * scale),
        int((box["y"] + box["h"]) * scale),
    )


def _display_to_box(p0: tuple[int, int], p1: tuple[int, int], scale: float, width: int, height: int) -> dict[str, int]:
    x0, y0 = p0
    x1, y1 = p1
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    inv = 1.0 / max(scale, 1e-6)
    x = max(0, min(width - 1, int(round(x0 * inv))))
    y = max(0, min(height - 1, int(round(y0 * inv))))
    w = max(1, min(width - x, int(round((x1 - x0) * inv))))
    h = max(1, min(height - y, int(round((y1 - y0) * inv))))
    return {"x": x, "y": y, "w": w, "h": h}


def _draw_overlay(frame, frame_idx: int, total: int, saved_frame: int | None, saved_box: dict[str, int] | None, current_box: dict[str, int] | None, scale: float):
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"frame {frame_idx}/{max(0, total - 1)}",
        "n next | p previous | + +10 | - -10 | drag box | s save | q quit",
        "Mark the real object only. Do not mark background texture.",
    ]
    if saved_frame is not None and saved_box:
        lines.append(f"saved: frame {saved_frame} box x={saved_box['x']} y={saved_box['y']} w={saved_box['w']} h={saved_box['h']}")
    for i, line in enumerate(lines):
        y = 30 + i * 26
        cv2.putText(out, line, (16, y), font, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if saved_frame == frame_idx and saved_box:
        box = _box_to_display(saved_box, scale)
        if box:
            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 220, 255), 2)
            cv2.putText(out, "saved box", (box[0], max(20, box[1] - 8)), font, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    if current_box:
        box = _box_to_display(current_box, scale)
        if box:
            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (255, 220, 0), 2)
            cv2.putText(out, "current box", (box[0], max(20, box[1] - 8)), font, 0.55, (255, 220, 0), 2, cv2.LINE_AA)
    return out


def _read_frame(cap, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def _save_preview(case_id: str, frame, frame_idx: int, box: dict[str, int]) -> Path:
    out_dir = interactive_output_dir(case_id)
    preview = frame.copy()
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 220, 0), max(2, preview.shape[1] // 600))
    cv2.drawMarker(preview, (x + w // 2, y + h // 2), (255, 220, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(preview, f"manual annotation frame {frame_idx}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2, cv2.LINE_AA)
    path = out_dir / "manual_annotation_preview.png"
    cv2.imwrite(str(path), preview)
    return path


def annotate_interactive_track(case_id: str) -> dict[str, Any]:
    case = _load_case(case_id)
    request = _load_or_prepare_request(case_id)
    request_path = interactive_request_path(case_id)
    cap = cv2.VideoCapture(str(case["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {case['video_path']}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_idx = _hint_frame(case_id, request, total)
    prompt = request.setdefault("object_prompt", {"type": "box_or_points", "box": None, "positive_points": [], "negative_points": []})
    saved_frame = request.get("first_object_frame")
    saved_box = prompt.get("box") if isinstance(prompt.get("box"), dict) else None
    current_box = dict(saved_box) if saved_box else None
    mouse = {"dragging": False, "start": None, "end": None, "scale": 1.0, "current_frame": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["dragging"] = True
            mouse["start"] = (x, y)
            mouse["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["dragging"]:
            mouse["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["dragging"] = False
            mouse["end"] = (x, y)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    saved = False
    preview_path: Path | None = None
    try:
        while True:
            frame = _read_frame(cap, frame_idx)
            display, scale = _scale_frame(frame)
            mouse["scale"] = scale
            if mouse["start"] and mouse["end"]:
                current_box = _display_to_box(mouse["start"], mouse["end"], scale, width, height)
            shown = _draw_overlay(display, frame_idx, total, saved_frame, saved_box, current_box, scale)
            cv2.imshow(WINDOW_NAME, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                frame_idx = min(total - 1, frame_idx + 1)
                mouse["start"] = mouse["end"] = None
            elif key == ord("p"):
                frame_idx = max(0, frame_idx - 1)
                mouse["start"] = mouse["end"] = None
            elif key in (ord("+"), ord("=")):
                frame_idx = min(total - 1, frame_idx + 10)
                mouse["start"] = mouse["end"] = None
            elif key in (ord("-"), ord("_")):
                frame_idx = max(0, frame_idx - 10)
                mouse["start"] = mouse["end"] = None
            elif key == ord("s"):
                if not current_box:
                    continue
                cx = current_box["x"] + current_box["w"] / 2
                cy = current_box["y"] + current_box["h"] / 2
                request["first_object_frame"] = frame_idx
                request["do_not_track_before_first_frame"] = True
                request["object_prompt"] = {
                    "type": "box_or_points",
                    "box": current_box,
                    "positive_points": [[cx, cy]],
                    "negative_points": [],
                }
                write_json(request_path, request)
                preview_path = _save_preview(case_id, frame, frame_idx, current_box)
                saved_frame = frame_idx
                saved_box = current_box
                saved = True
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_NAME)
    return {
        "case_id": case_id,
        "saved": saved,
        "request": str(request_path),
        "first_object_frame": saved_frame,
        "box": saved_box,
        "preview": str(preview_path) if preview_path else None,
    }
