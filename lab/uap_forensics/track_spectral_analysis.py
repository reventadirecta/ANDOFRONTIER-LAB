from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import read_json, write_json
from .paths import DATA_DIR, ensure_dir


VALID_STATES = {"TRACKING_ACTIVE", "TRACKING ACTIVE", "tracked", "auto_recovered"}
REQUIRED_MESSAGE = "Spectral analysis requires a human-validated track and dynamic ROIs."


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def dynamic_rois_csv(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "spectral_analysis")


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
    out: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            def i(name: str) -> int | None:
                value = raw.get(name, "")
                return int(float(value)) if value not in {"", None} else None

            bbox = [i("expanded_x"), i("expanded_y"), i("expanded_w"), i("expanded_h")]
            tight = [i("bbox_x"), i("bbox_y"), i("bbox_w"), i("bbox_h")]
            out.append(
                {
                    "frame": int(raw["frame"]),
                    "status": raw.get("status", ""),
                    "bbox": bbox if all(v is not None for v in bbox) else None,
                    "tight_bbox": tight if all(v is not None for v in tight) else None,
                    "area": float(raw.get("area") or 0),
                }
            )
    return sorted(out, key=lambda row: row["frame"])


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


def _spatial_frequency(gray: np.ndarray) -> tuple[float, float, float]:
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(small))
    mag = np.log1p(np.abs(fft))
    yy, xx = np.indices(mag.shape)
    center = np.array([(mag.shape[0] - 1) / 2, (mag.shape[1] - 1) / 2])
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    high_mask = radius > (0.35 * radius.max())
    total = float(np.sum(mag) + 1e-9)
    high = float(np.sum(mag[high_mask]))
    lap = cv2.Laplacian(small, cv2.CV_32F)
    return float(np.mean(mag)), high / total, float(np.var(lap))


def _collect_series(track: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[np.ndarray]]:
    cap, fps, total = _open_video(track)
    valid = [row for row in rows if row["status"] in VALID_STATES and row.get("bbox")]
    series: list[dict[str, Any]] = []
    contact: list[np.ndarray] = []
    spatial_energy = []
    high_ratio = []
    noise_proxy = []
    pick_every = max(1, len(valid) // 12) if valid else 1
    for pos, row in enumerate(valid):
        frame_idx = int(row["frame"])
        frame = _read_frame(cap, frame_idx)
        if frame is None:
            continue
        x, y, w, h = [int(v) for v in row["bbox"]]
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(crop)
        se, hf, npv = _spatial_frequency(gray)
        spatial_energy.append(se)
        high_ratio.append(hf)
        noise_proxy.append(npv)
        lum_mean = float(gray.mean())
        row_out = {
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps else 0.0,
            "luminance_mean": lum_mean,
            "luminance_std": float(gray.std()),
            "red_mean": float(r.mean()),
            "green_mean": float(g.mean()),
            "blue_mean": float(b.mean()),
            "hue_mean": float(hsv[..., 0].mean()),
            "saturation_mean": float(hsv[..., 1].mean()),
            "value_mean": float(hsv[..., 2].mean()),
            "bbox_area": float(row.get("area") or w * h),
            "track_status": row["status"],
            "spatial_frequency_energy": se,
            "high_frequency_energy_ratio": hf,
            "compression_noise_proxy": npv,
        }
        series.append(row_out)
        if pos % pick_every == 0 and len(contact) < 12:
            tile = cv2.resize(crop, (220, 150), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"f{frame_idx} L={lum_mean:.1f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, f"HF={hf:.3f}", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            contact.append(tile)
    cap.release()
    metrics = _summarize(track, fps, total, series, spatial_energy, high_ratio, noise_proxy)
    return series, metrics, contact


def _temporal_fft(luminance: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    if len(luminance) < 4 or fps <= 0:
        return np.zeros(0), np.zeros(0), 0.0, 0.0, 0.0
    signal = luminance.astype(np.float32) - float(np.mean(luminance))
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    if len(spectrum) > 1:
        spectrum[0] = 0.0
        idx = int(np.argmax(spectrum))
        peak_freq = float(freqs[idx])
        peak_strength = float(spectrum[idx] / (np.sum(spectrum) + 1e-9))
    else:
        peak_freq = 0.0
        peak_strength = 0.0
    p = spectrum / (np.sum(spectrum) + 1e-9)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])) / max(1.0, np.log2(len(p)))) if len(p) else 0.0
    return freqs, spectrum, peak_freq, peak_strength, entropy


def _summarize(track: dict[str, Any], fps: float, total: int, series: list[dict[str, Any]], spatial: list[float], high: list[float], noise: list[float]) -> dict[str, Any]:
    lum = np.array([row["luminance_mean"] for row in series], dtype=np.float32)
    r = np.array([row["red_mean"] for row in series], dtype=np.float32)
    g = np.array([row["green_mean"] for row in series], dtype=np.float32)
    b = np.array([row["blue_mean"] for row in series], dtype=np.float32)
    h = np.array([row["hue_mean"] for row in series], dtype=np.float32)
    s = np.array([row["saturation_mean"] for row in series], dtype=np.float32)
    v = np.array([row["value_mean"] for row in series], dtype=np.float32)
    _freqs, _spec, dom, strength, entropy = _temporal_fft(lum, fps)
    duration = (series[-1]["timestamp"] - series[0]["timestamp"]) if len(series) > 1 else 0.0
    flicker = float(np.std(lum) / max(1e-6, float(np.mean(lum)))) if len(lum) else 0.0
    return {
        "case_id": track.get("case_id"),
        "total_frames": int(total or track.get("summary", {}).get("total_frames") or len(series)),
        "valid_tracked_frames": len(series),
        "fps": fps,
        "duration_analyzed": duration,
        "mean_luminance": float(np.mean(lum)) if len(lum) else 0.0,
        "luminance_std": float(np.std(lum)) if len(lum) else 0.0,
        "luminance_min": float(np.min(lum)) if len(lum) else 0.0,
        "luminance_max": float(np.max(lum)) if len(lum) else 0.0,
        "mean_red_channel": float(np.mean(r)) if len(r) else 0.0,
        "mean_green_channel": float(np.mean(g)) if len(g) else 0.0,
        "mean_blue_channel": float(np.mean(b)) if len(b) else 0.0,
        "mean_hue": float(np.mean(h)) if len(h) else 0.0,
        "mean_saturation": float(np.mean(s)) if len(s) else 0.0,
        "mean_value": float(np.mean(v)) if len(v) else 0.0,
        "luminance_flicker_index": flicker,
        "dominant_temporal_frequency_hz": dom,
        "temporal_frequency_peak_strength": strength,
        "spectral_entropy_temporal": entropy,
        "mean_spatial_frequency_energy": float(np.mean(spatial)) if spatial else 0.0,
        "high_frequency_energy_ratio": float(np.mean(high)) if high else 0.0,
        "compression_noise_proxy": float(np.mean(noise)) if noise else 0.0,
        "tracker_used": track.get("tracker_backend") or track.get("backend_used"),
        "notes": [
            "Spectral analysis is computed inside human-validated dynamic track ROIs.",
            "Temporal frequency is based on luminance over tracked frames, not the full frame.",
            "This does not measure physical composition, real temperature, origin, or material.",
        ],
    }


def _write_timeseries(path: Path, series: list[dict[str, Any]]) -> None:
    fields = [
        "frame",
        "timestamp",
        "luminance_mean",
        "luminance_std",
        "red_mean",
        "green_mean",
        "blue_mean",
        "hue_mean",
        "saturation_mean",
        "value_mean",
        "bbox_area",
        "track_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in series:
            writer.writerow({key: row[key] for key in fields})


def _luminance_panel(path: Path, series: list[dict[str, Any]]) -> None:
    frames = np.array([row["frame"] for row in series])
    lum = np.array([row["luminance_mean"] for row in series])
    std = np.array([row["luminance_std"] for row in series])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, lum, color="#2f80ed")
    if len(lum) > 4:
        peaks = lum >= np.percentile(lum, 95)
        axes[0].scatter(frames[peaks], lum[peaks], color="red", s=18, label="top 5%")
        axes[0].legend()
    axes[0].set_title("Mean luminance inside tracked ROI")
    axes[0].set_ylabel("luminance")
    axes[1].plot(frames, std, color="#f28e2b")
    axes[1].set_title("Luminance standard deviation")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("std")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _color_panel(path: Path, series: list[dict[str, Any]]) -> None:
    frames = np.array([row["frame"] for row in series])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, [row["red_mean"] for row in series], color="red", label="R")
    axes[0].plot(frames, [row["green_mean"] for row in series], color="green", label="G")
    axes[0].plot(frames, [row["blue_mean"] for row in series], color="blue", label="B")
    axes[0].set_title("Mean RGB channels inside tracked ROI")
    axes[0].legend()
    axes[1].plot(frames, [row["hue_mean"] for row in series], color="#8e44ad", label="Hue")
    axes[1].plot(frames, [row["saturation_mean"] for row in series], color="#e67e22", label="Saturation")
    axes[1].plot(frames, [row["value_mean"] for row in series], color="#34495e", label="Value")
    axes[1].set_title("Mean HSV channels")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fft_panel(path: Path, series: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    lum = np.array([row["luminance_mean"] for row in series], dtype=np.float32)
    freqs, spectrum, dom, strength, _entropy = _temporal_fft(lum, float(metrics.get("fps") or 0))
    fig, ax = plt.subplots(figsize=(12, 5))
    if len(freqs):
        ax.plot(freqs, spectrum / (np.max(spectrum) + 1e-9), color="#17a589")
        ax.axvline(dom, color="red", linestyle="--", label=f"dominant {dom:.3f} Hz")
        ax.legend()
    warning = "short/unstable signal caution" if metrics.get("duration_analyzed", 0) < 5 else "tracked ROI temporal FFT"
    ax.set_title(f"Temporal luminance FFT ({warning}, peak strength {strength:.3f})")
    ax.set_xlabel("Hz")
    ax.set_ylabel("relative power")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _spatial_panel(path: Path, series: list[dict[str, Any]]) -> None:
    frames = np.array([row["frame"] for row in series])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frames, [row["spatial_frequency_energy"] for row in series], color="#4e79a7")
    axes[0].set_title("Mean spatial frequency energy")
    axes[1].plot(frames, [row["high_frequency_energy_ratio"] for row in series], color="#e15759")
    axes[1].set_title("High-frequency energy ratio")
    axes[2].plot(frames, [row["compression_noise_proxy"] for row in series], color="#59a14f")
    axes[2].set_title("Compression/noise proxy")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _contact_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    if not tiles:
        return
    h = max(tile.shape[0] for tile in tiles)
    w = max(tile.shape[1] for tile in tiles)
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + tile.shape[0], c * w : c * w + tile.shape[1]] = tile
    cv2.imwrite(str(path), sheet)


def _write_report(case_id: str, track: dict[str, Any], validation: dict[str, Any], metrics: dict[str, Any], paths: dict[str, Path]) -> Path:
    report = output_dir(case_id) / "spectral_analysis_report.md"
    lines = [
        f"# Spectral Analysis Track-Based Report: {case_id}",
        "",
        "## Scope",
        "",
        "This module describes temporal luminance, color, temporal FFT and spatial-frequency behavior inside the human-validated dynamic track ROI only.",
        "",
        "## Source Track",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- tracker used: `{metrics.get('tracker_used')}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Summary",
        "",
        f"- frames analyzed: `{metrics['valid_tracked_frames']}`",
        f"- fps: `{metrics['fps']:.4f}`",
        f"- duration analyzed: `{metrics['duration_analyzed']:.4f}` seconds",
        f"- mean luminance: `{metrics['mean_luminance']:.4f}`",
        f"- luminance std: `{metrics['luminance_std']:.4f}`",
        f"- mean RGB: `R={metrics['mean_red_channel']:.4f}, G={metrics['mean_green_channel']:.4f}, B={metrics['mean_blue_channel']:.4f}`",
        f"- mean HSV: `H={metrics['mean_hue']:.4f}, S={metrics['mean_saturation']:.4f}, V={metrics['mean_value']:.4f}`",
        f"- flicker index: `{metrics['luminance_flicker_index']:.6f}`",
        f"- dominant temporal frequency: `{metrics['dominant_temporal_frequency_hz']:.6f}` Hz",
        f"- temporal frequency peak strength: `{metrics['temporal_frequency_peak_strength']:.6f}`",
        f"- temporal spectral entropy: `{metrics['spectral_entropy_temporal']:.6f}`",
        f"- mean spatial frequency energy: `{metrics['mean_spatial_frequency_energy']:.6f}`",
        f"- high-frequency energy ratio: `{metrics['high_frequency_energy_ratio']:.6f}`",
        f"- compression/noise proxy: `{metrics['compression_noise_proxy']:.6f}`",
        "",
        "## Interpretation",
        "",
        "The results describe the tracked object's image signal: luminance variation, channel balance, temporal frequency content and texture/noise proxies inside the ROI.",
        "",
        "## Limitations",
        "",
        "- This does not measure physical composition, material, true temperature, distance, or origin.",
        "- It depends on tracking precision, source compression, camera motion, focus, exposure and frame rate.",
        "- Temporal FFT can be weak or unstable if the analyzed duration is short or the luminance signal is non-stationary.",
        "- No automatic ROI, brightness redetection, full-frame primary analysis, autoencoder, SRV or Thermal/IR module was used.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "spectral_manifest.md"
    classes = {
        "spectral_metrics.json": "technical",
        "spectral_timeseries.csv": "technical",
        "spectral_luminance_panel.png": "public_safe",
        "spectral_color_panel.png": "technical",
        "spectral_fft_panel.png": "technical",
        "spectral_spatial_frequency_panel.png": "technical",
        "spectral_contact_sheet.png": "public_safe",
        "spectral_analysis_report.md": "public_safe",
    }
    lines = ["# Spectral Analysis Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in outputs.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | track-based image signal only; no physical composition or origin claim |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_spectral_analysis(case_id: str) -> dict[str, Any]:
    track, validation, rows = _require_valid_inputs(case_id)
    out = output_dir(case_id)
    series, metrics, contact_tiles = _collect_series(track, rows)
    paths = {
        "spectral_metrics.json": out / "spectral_metrics.json",
        "spectral_timeseries.csv": out / "spectral_timeseries.csv",
        "spectral_luminance_panel.png": out / "spectral_luminance_panel.png",
        "spectral_color_panel.png": out / "spectral_color_panel.png",
        "spectral_fft_panel.png": out / "spectral_fft_panel.png",
        "spectral_spatial_frequency_panel.png": out / "spectral_spatial_frequency_panel.png",
        "spectral_contact_sheet.png": out / "spectral_contact_sheet.png",
    }
    write_json(paths["spectral_metrics.json"], metrics)
    _write_timeseries(paths["spectral_timeseries.csv"], series)
    _luminance_panel(paths["spectral_luminance_panel.png"], series)
    _color_panel(paths["spectral_color_panel.png"], series)
    _fft_panel(paths["spectral_fft_panel.png"], series, metrics)
    _spatial_panel(paths["spectral_spatial_frequency_panel.png"], series)
    _contact_sheet(paths["spectral_contact_sheet.png"], contact_tiles)
    report = _write_report(case_id, track, validation, metrics, paths)
    paths["spectral_analysis_report.md"] = report
    manifest = _write_manifest(case_id, paths)
    paths["spectral_manifest.md"] = manifest
    status_path = _update_case_status(
        case_id,
        {
            "spectral_analysis_status": "complete",
            "spectral_analysis_ready": True,
            "last_spectral_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "spectral_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(paths["spectral_metrics.json"]),
                "csv": str(paths["spectral_timeseries.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "panels": {
                    "luminance": str(paths["spectral_luminance_panel.png"]),
                    "color": str(paths["spectral_color_panel.png"]),
                    "fft": str(paths["spectral_fft_panel.png"]),
                    "spatial_frequency": str(paths["spectral_spatial_frequency_panel.png"]),
                    "contact_sheet": str(paths["spectral_contact_sheet.png"]),
                },
            },
        },
    )
    return {
        "case_id": case_id,
        "spectral_analysis_ready": True,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in paths.items()},
        "case_status": str(status_path),
    }
