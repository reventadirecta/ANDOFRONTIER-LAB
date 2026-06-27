from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .paths import case_frames_dir, ensure_dir
from .visuals import make_contact_sheet


def frame_paths(case_id: str) -> list[Path]:
    return sorted(case_frames_dir(case_id).glob("frame_*.png"))


def extract_frames(config: dict) -> dict:
    case_id = config["case_id"]
    video_path = Path(config["video_path"]).expanduser().resolve()
    window = config.get("frame_window", {})
    start = int(window.get("start") or 0)
    end = window.get("end")
    end = int(end) if end is not None else None
    step = max(1, int(window.get("step") or 1))

    output_dir = ensure_dir(case_frames_dir(case_id))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected_end = end if end is not None else total
    saved = 0
    contact_images: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for index in tqdm(range(start, selected_end, step), desc=f"extract {case_id}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        out = output_dir / f"frame_{index:06d}.png"
        cv2.imwrite(str(out), frame)
        saved += 1
        if len(contact_images) < 25:
            thumb = cv2.resize(frame, (240, int(240 * frame.shape[0] / frame.shape[1])))
            contact_images.append(thumb)
    cap.release()
    make_contact_sheet(contact_images, output_dir / "contact_sheet.png")
    return {"case_id": case_id, "frames_saved": saved, "frames_dir": str(output_dir)}


def load_frames(case_id: str, grayscale: bool = False) -> list[np.ndarray]:
    paths = frame_paths(case_id)
    if not paths:
        raise FileNotFoundError(f"No extracted frames found for case {case_id}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    frames = [cv2.imread(str(path), flag) for path in paths]
    return [frame for frame in frames if frame is not None]
