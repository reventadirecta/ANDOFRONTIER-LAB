from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import case_output_dir, ensure_dir
from .roi import load_roi_frames
from .visuals import save_image, save_panel


def _gray_stack(case_id: str) -> np.ndarray:
    frames = load_roi_frames(case_id, grayscale=True)
    return np.stack(frames).astype(np.float32)


def run_base_analysis(config: dict) -> dict:
    case_id = config["case_id"]
    out = ensure_dir(case_output_dir(case_id) / "base")
    stack = _gray_stack(case_id)
    mean_projection = stack.mean(axis=0)
    max_projection = stack.max(axis=0)
    min_projection = stack.min(axis=0)
    minimal_resistance = max_projection - min_projection
    diffs = np.abs(np.diff(stack, axis=0))
    diff_mean = diffs.mean(axis=0) if len(diffs) else np.zeros_like(mean_projection)

    save_image(out / "mean_projection.png", mean_projection)
    save_image(out / "max_projection.png", max_projection)
    save_image(out / "min_projection.png", min_projection)
    save_image(out / "minimal_resistance.png", minimal_resistance)
    save_image(out / "frame_difference_mean.png", diff_mean)

    frames_bgr = load_roi_frames(case_id)
    first = frames_bgr[0]
    hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)
    clahe_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    edges = cv2.Canny(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), 50, 150)
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(mean_projection))))
    fft_norm = 255 * (fft - fft.min()) / (np.ptp(fft) + 1e-6)

    flow_mag = np.zeros_like(mean_projection)
    if len(stack) > 1:
        prev = stack[0].astype(np.uint8)
        mags = []
        for current in stack[1:].astype(np.uint8):
            flow = cv2.calcOpticalFlowFarneback(prev, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(mag)
            prev = current
        flow_mag = np.mean(mags, axis=0)
    save_image(out / "optical_flow_magnitude.png", 255 * flow_mag / (flow_mag.max() + 1e-6))

    luminance_profile = pd.DataFrame(
        {
            "frame_index": np.arange(stack.shape[0]),
            "mean_luminance": stack.mean(axis=(1, 2)),
            "max_luminance": stack.max(axis=(1, 2)),
            "std_luminance": stack.std(axis=(1, 2)),
        }
    )
    luminance_profile.to_csv(out / "luminance_profiles.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(luminance_profile["frame_index"], luminance_profile["mean_luminance"], label="mean")
    ax.plot(luminance_profile["frame_index"], luminance_profile["max_luminance"], label="max")
    ax.set_xlabel("frame")
    ax.set_ylabel("luminance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "luminance_profile.png", dpi=150)
    plt.close(fig)

    save_panel(
        out / "base_panel.png",
        {
            "mean": mean_projection,
            "max": max_projection,
            "diff mean": diff_mean,
            "flow magnitude": 255 * flow_mag / (flow_mag.max() + 1e-6),
            "clahe": clahe_l,
            "edges": edges,
            "fft spatial": fft_norm,
            "hue": hsv[:, :, 0],
        },
    )
    return {"case_id": case_id, "output_dir": str(out), "frames_analyzed": int(stack.shape[0])}
