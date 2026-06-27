from pathlib import Path

import cv2
import numpy as np

from .frames import load_frames
from .io import write_json
from .paths import case_roi_dir, ensure_dir
from .visuals import save_image


def detect_roi_by_brightness(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = max(float(np.percentile(blur, 99.5)), float(blur.mean() + 2.5 * blur.std()))
    mask = (blur >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        side = min(w, h) // 4
        return {"mode": "auto-brightness-fallback", "x": (w - side) // 2, "y": (h - side) // 2, "width": side, "height": side}
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    pad = max(8, int(max(w, h) * 0.75))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return {"mode": "auto-brightness", "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def resolve_roi(config: dict) -> dict:
    roi = dict(config.get("roi") or {})
    mode = roi.get("mode", "manual")
    if mode == "manual" and all(int(roi.get(key, 0) or 0) > 0 for key in ("width", "height")):
        return roi
    frame = load_frames(config["case_id"])[0]
    if mode in {"auto", "auto-brightness"} or not roi.get("width") or not roi.get("height"):
        return detect_roi_by_brightness(frame)
    return roi


def crop_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    x = max(0, int(roi["x"]))
    y = max(0, int(roi["y"]))
    w = max(1, int(roi["width"]))
    h = max(1, int(roi["height"]))
    return frame[y : y + h, x : x + w]


def save_roi_frames(config: dict) -> dict:
    case_id = config["case_id"]
    roi = resolve_roi(config)
    out_dir = ensure_dir(case_roi_dir(case_id))
    frames = load_frames(case_id)
    for idx, frame in enumerate(frames):
        save_image(out_dir / f"roi_{idx:06d}.png", crop_roi(frame, roi))
    write_json(out_dir / "roi.json", {"case_id": case_id, "roi": roi})
    return {"case_id": case_id, "roi": roi, "roi_frames": len(frames), "roi_dir": str(out_dir)}


def load_roi_frames(case_id: str, grayscale: bool = False) -> list[np.ndarray]:
    out_dir = case_roi_dir(case_id)
    paths = sorted(out_dir.glob("roi_*.png"))
    if not paths:
        raise FileNotFoundError(f"No ROI frames found for case {case_id}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return [cv2.imread(str(path), flag) for path in paths]
