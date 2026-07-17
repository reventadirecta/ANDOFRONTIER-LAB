from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


FORMATS = {
    "16x9": {"size": (1280, 720), "fps": 30},
    "9x16": {"size": (720, 1280), "fps": 30},
    "1x1": {"size": (1080, 1080), "fps": 30},
}

SAFE_TITLE = "AndoFrontier Lab - Evidence Video Export"
NO_ORIGIN_CLAIM = "No origin claim - technical review only"


@dataclass
class Scene:
    kind: str
    title: str
    source: Path | None = None
    subtitle: str = ""
    duration: float = 3.0


def export_evidence_videos(case_id: str, fmt: str = "all") -> dict[str, Any]:
    _validate_case_id(case_id)
    formats = list(FORMATS) if fmt == "all" else [fmt]
    unknown = [item for item in formats if item not in FORMATS]
    if unknown:
        raise ValueError(f"Unsupported format(s): {', '.join(unknown)}")

    out_dir = ensure_dir(case_report_dir(case_id) / "video_exports")
    scenes = _build_scenes(case_id)
    outputs: dict[str, Any] = {}
    for item in formats:
        output = out_dir / f"{_safe_name(case_id)}_{item}.mp4"
        outputs[item] = _render_format(item, scenes, output)

    manifest = {
        "case_id": case_id,
        "formats": formats,
        "output_dir": str(out_dir),
        "outputs": outputs,
        "source_policy": "Uses existing local analysis outputs only; no ROI redetection and no origin claim.",
        "missing_assets": [scene.title for scene in scenes if scene.source and not scene.source.exists()],
    }
    write_json(out_dir / "video_export_manifest.json", manifest)
    return manifest


def _validate_case_id(value: str) -> None:
    if value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("case_id must contain only letters, numbers, dots, underscores, or hyphens")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120] or "case"


def _json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path) if path.exists() else {}
    except Exception:
        return {}


def _build_scenes(case_id: str) -> list[Scene]:
    outputs = DATA_DIR / "outputs" / case_id
    reports = DATA_DIR / "reports" / case_id
    summary = _json(reports / "unified_report" / "unified_case_summary.json")
    status = summary.get("module_status", {}) if isinstance(summary, dict) else {}
    assessment = summary.get(
        "overall_public_safe_assessment",
        "Traceable technical review with limitations.",
    ) if isinstance(summary, dict) else ""

    return [
        Scene("title", "Evidence video export", subtitle=f"{case_id}\n{NO_ORIGIN_CLAIM}", duration=2.5),
        Scene("image", "Unified scorecard", reports / "unified_report" / "unified_case_scorecard.png", "Public-safe metrics summary", 3.5),
        Scene("video", "Human-validated tracking", outputs / "interactive_tracking" / "track_overlay_preview_web.mp4", "Manual box -> tracker follows selected object", 8.0),
        Scene("image", "Motion / optical flow", outputs / "motion_analysis" / "motion_trajectory_panel.png", "Trajectory, velocity, stability context", 3.5),
        Scene("image", "Motion field", outputs / "motion_analysis" / "motion_optical_flow_panel.png", "Optical flow inside tracked ROI", 3.0),
        Scene("image", "Spectral analysis", outputs / "spectral_analysis" / "spectral_fft_panel.png", "FFT / frequency panel", 3.5),
        Scene("image", "Thermal / IR relative intensity", outputs / "thermal_analysis" / "thermal_intensity_panel.png", "Relative FLIR/IR intensity only", 3.5),
        Scene("image", "Clean controls baseline", outputs / "controls_analysis" / "controls_summary_panel.png", "Object ROI vs masked clean controls", 3.5),
        Scene("image", "PCA baseline", outputs / "pca_analysis" / "pca_scatter_panel.png", "PCA comparison against baselines", 3.5),
        Scene("image", "Autoencoder baseline", outputs / "autoencoder_analysis" / "autoencoder_summary_panel.png", "Reconstruction-error comparison", 3.5),
        Scene("image", "SRV visual reconstruction", outputs / "srv_analysis" / "srv_comparison_panel.png", "Non-generative stabilization/enhancement", 3.5),
        Scene("image", "Object-core SRV", outputs / "srv_analysis" / "object_core" / "srv_core_comparison_panel.png", "Tracked object-core context", 3.5),
        Scene("metrics", "Workflow status", subtitle=_status_text(status, assessment), duration=4.0),
        Scene("title", "Conclusion", subtitle="Technical measurements only\nSource/context must be reviewed separately\nNo origin, intent, identity, or provenance claim", duration=3.0),
    ]


def _status_text(status: dict[str, Any], assessment: str) -> str:
    if not status:
        return f"{assessment}\nNo origin claim."
    lines = []
    for key in ("tracking", "dynamic_roi", "motion", "spectral", "thermal_ir", "controls", "pca", "autoencoder", "srv"):
        if key in status:
            lines.append(f"{key.replace('_', ' ')}: {status[key]}")
    lines.extend(["", assessment])
    return "\n".join(lines[:12])


def _render_format(fmt: str, scenes: list[Scene], output: Path) -> dict[str, Any]:
    width, height = FORMATS[fmt]["size"]
    fps = FORMATS[fmt]["fps"]
    temp = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {temp}")

    rendered_scenes: list[dict[str, Any]] = []
    try:
        for scene in scenes:
            count = _write_scene(writer, scene, width, height, fps)
            rendered_scenes.append(
                {
                    "title": scene.title,
                    "kind": scene.kind,
                    "source": str(scene.source) if scene.source else None,
                    "source_exists": bool(scene.source and scene.source.exists()),
                    "frames_written": count,
                    "duration_seconds": round(count / fps, 3),
                }
            )
    finally:
        writer.release()

    final = _web_encode(temp, output)
    if temp.exists() and temp != final:
        temp.unlink(missing_ok=True)
    return {
        "format": fmt,
        "path": str(final),
        "exists": final.exists(),
        "size": final.stat().st_size if final.exists() else 0,
        "canvas": {"width": width, "height": height, "fps": fps},
        "scenes": rendered_scenes,
    }


def _write_scene(writer: cv2.VideoWriter, scene: Scene, width: int, height: int, fps: int) -> int:
    if scene.kind == "video" and scene.source and scene.source.exists():
        return _write_video_scene(writer, scene, width, height, fps)
    frame = _scene_frame(scene, width, height)
    count = max(1, int(round(scene.duration * fps)))
    for idx in range(count):
        writer.write(_animate_frame(frame, idx, count, width, height))
    return count


def _write_video_scene(writer: cv2.VideoWriter, scene: Scene, width: int, height: int, fps: int) -> int:
    cap = cv2.VideoCapture(str(scene.source))
    max_frames = max(1, int(round(scene.duration * fps)))
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = _fit_image(frame, width, height)
        _draw_header(canvas, scene.title, scene.subtitle)
        _draw_footer(canvas, NO_ORIGIN_CLAIM)
        writer.write(canvas)
        written += 1
    cap.release()
    if written == 0:
        return _write_scene(
            writer,
            Scene("missing", scene.title, scene.source, "Video asset missing or unreadable", 2.0),
            width,
            height,
            fps,
        )
    return written


def _scene_frame(scene: Scene, width: int, height: int) -> np.ndarray:
    if scene.kind in {"title", "metrics"}:
        canvas = np.full((height, width, 3), (17, 22, 24), dtype=np.uint8)
        _draw_center_text(canvas, SAFE_TITLE if scene.kind == "title" else scene.title, scene.subtitle)
        _draw_footer(canvas, NO_ORIGIN_CLAIM)
        return canvas

    if scene.source and scene.source.exists():
        image = cv2.imread(str(scene.source), cv2.IMREAD_COLOR)
        if image is not None:
            canvas = _fit_image(image, width, height)
            _draw_header(canvas, scene.title, scene.subtitle)
            _draw_footer(canvas, NO_ORIGIN_CLAIM)
            return canvas

    canvas = np.full((height, width, 3), (21, 26, 28), dtype=np.uint8)
    subtitle = scene.subtitle or "Asset not available for this case"
    _draw_center_text(canvas, scene.title, f"{subtitle}\nSkipped/missing asset")
    _draw_footer(canvas, NO_ORIGIN_CLAIM)
    return canvas


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (10, 12, 13), dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0:
        return canvas
    margin_x = int(width * 0.055)
    margin_top = int(height * 0.12)
    margin_bottom = int(height * 0.11)
    max_width = max(1, width - 2 * margin_x)
    max_height = max(1, height - margin_top - margin_bottom)
    scale = min(max_width / image_width, max_height / image_height)
    new_width = max(1, int(image_width * scale))
    new_height = max(1, int(image_height * scale))
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    x = (width - new_width) // 2
    y = margin_top + (max_height - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    return canvas


def _animate_frame(frame: np.ndarray, idx: int, count: int, width: int, height: int) -> np.ndarray:
    if count <= 1:
        return frame.copy()
    progress = idx / max(1, count - 1)
    zoom = 1.0 + 0.035 * math.sin(progress * math.pi)
    if zoom <= 1.001:
        return frame.copy()
    new_width = int(width * zoom)
    new_height = int(height * zoom)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    x = (new_width - width) // 2
    y = (new_height - height) // 2
    return resized[y : y + height, x : x + width].copy()


def _draw_header(canvas: np.ndarray, title: str, subtitle: str = "") -> None:
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, int(height * 0.095)), (8, 11, 12), -1)
    cv2.putText(canvas, title[:64], (int(width * 0.04), int(height * 0.055)), cv2.FONT_HERSHEY_SIMPLEX, _font_scale(width, 0.78), (235, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle.splitlines()[0][:80], (int(width * 0.04), int(height * 0.084)), cv2.FONT_HERSHEY_SIMPLEX, _font_scale(width, 0.42), (120, 225, 160), 1, cv2.LINE_AA)


def _draw_footer(canvas: np.ndarray, text: str) -> None:
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, int(height * 0.92)), (width, height), (8, 11, 12), -1)
    cv2.putText(canvas, text[:100], (int(width * 0.04), int(height * 0.965)), cv2.FONT_HERSHEY_SIMPLEX, _font_scale(width, 0.48), (210, 220, 224), 1, cv2.LINE_AA)


def _draw_center_text(canvas: np.ndarray, title: str, subtitle: str) -> None:
    height, width = canvas.shape[:2]
    y = int(height * 0.28)
    for line in _wrap(title, max(18, int(width / 34))):
        cv2.putText(canvas, line, (int(width * 0.08), y), cv2.FONT_HERSHEY_SIMPLEX, _font_scale(width, 1.05), (117, 255, 159), 3, cv2.LINE_AA)
        y += int(height * 0.07)
    y += int(height * 0.03)
    subtitle_lines: list[str] = []
    for part in subtitle.splitlines():
        subtitle_lines.extend(_wrap(part, max(24, int(width / 26))))
    for line in subtitle_lines:
        cv2.putText(canvas, line, (int(width * 0.08), y), cv2.FONT_HERSHEY_SIMPLEX, _font_scale(width, 0.56), (230, 236, 238), 1, cv2.LINE_AA)
        y += int(height * 0.042)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _font_scale(width: int, base: float) -> float:
    return max(0.38, base * width / 1280)


def _web_encode(raw: Path, final: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raw.replace(final)
        return final
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return final
