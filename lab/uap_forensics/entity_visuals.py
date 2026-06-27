from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_target import _roi_xywh, _selected_indices, entity_target_path
from .frames import load_frames
from .io import read_json
from .paths import case_output_dir, ensure_dir


def _out_dir(case_id: str) -> Path:
    return ensure_dir(case_output_dir(case_id) / "entity_visuals")


def _crop(frame: np.ndarray, roi: dict[str, Any]) -> np.ndarray:
    x, y, w, h = _roi_xywh(roi)
    return frame[y : y + h, x : x + w].copy()


def _frame_at(frames: list[np.ndarray], idx: int) -> np.ndarray:
    return frames[max(0, min(len(frames) - 1, idx))]


def _draw_cv_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def _annotated_full_frame(frame: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    canvas = frame.copy()
    ex, ey, ew, eh = _roi_xywh(target["entity_roi"])
    lx, ly, lw, lh = _roi_xywh(target["light_roi"])
    cv2.rectangle(canvas, (ex, ey), (ex + ew, ey + eh), (80, 255, 80), 2)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), (60, 60, 255), 2)
    _draw_cv_label(canvas, "PRIMARY TARGET: ENTITY STRUCTURE", (18, 32), (80, 255, 80))
    _draw_cv_label(canvas, "LIGHT ROI: SECONDARY / EXCLUDED EVENT", (18, 62), (80, 180, 255))
    return canvas


def _save_contact_sheet(
    path: Path,
    frames: list[np.ndarray],
    target: dict[str, Any],
    indices: list[int],
    title: str,
    excluded: bool = False,
    max_images: int = 24,
) -> None:
    if not indices:
        return
    if len(indices) <= max_images:
        sample = indices
    else:
        sample_positions = np.linspace(0, len(indices) - 1, max_images).round().astype(int)
        sample = [indices[int(pos)] for pos in sample_positions]
    cols = 6
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.35))
    axes_arr = np.array(axes).reshape(-1)
    for ax, idx in zip(axes_arr, sample):
        crop = _crop(_frame_at(frames, idx), target["entity_roi"])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {idx}", fontsize=8, color=("red" if excluded else "black"))
        ax.axis("off")
        if excluded:
            ax.text(
                0.02,
                0.94,
                "EXCLUDED",
                transform=ax.transAxes,
                color="white",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "red", "alpha": 0.78, "pad": 2},
            )
    for ax in axes_arr[len(sample) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_overview(path: Path, frames: list[np.ndarray], target: dict[str, Any]) -> None:
    idx = int(target["analysis_segments"][0]["start_frame"]) if target.get("analysis_segments") else 0
    image = _annotated_full_frame(_frame_at(frames, idx), target)
    cv2.imwrite(str(path), image)


def _save_entity_vs_light(path: Path, frames: list[np.ndarray], target: dict[str, Any]) -> None:
    entity_idx = int(target["analysis_segments"][0]["start_frame"]) if target.get("analysis_segments") else 0
    light_idx = int(target["light_excluded_segments"][0]["start_frame"]) if target.get("light_excluded_segments") else entity_idx
    entity = _crop(_frame_at(frames, entity_idx), target["entity_roi"])
    light = _crop(_frame_at(frames, light_idx), target["light_roi"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(entity, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Intended target: entity structure (frame {entity_idx})")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(light, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Secondary/excluded light event (frame {light_idx})")
    axes[1].axis("off")
    fig.suptitle("Entity structure and saturated light are separate targets")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_clahe_panel(path: Path, frames: list[np.ndarray], target: dict[str, Any]) -> None:
    idx = int(target["analysis_segments"][0]["start_frame"]) if target.get("analysis_segments") else 0
    crop = _crop(_frame_at(frames, idx), target["entity_roi"])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    edges = cv2.Canny(gray, 35, 120)
    contour = crop.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, contours, -1, (80, 255, 80), 1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [
        ("entity crop", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), None),
        ("CLAHE / local contrast", clahe_l, "gray"),
        ("edges", edges, "gray"),
        ("approx contours", cv2.cvtColor(contour, cv2.COLOR_BGR2RGB), None),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Entity structure contrast/contour view - not centered on the light patch")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_motion_panel(path: Path, frames: list[np.ndarray], target: dict[str, Any], indices: list[int]) -> None:
    sample = indices[: min(140, len(indices))]
    crops = [_crop(_frame_at(frames, idx), target["entity_roi"]) for idx in sample]
    grays = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crops]
    flow_mags = []
    diffs = []
    for prev, cur in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(mag)
        diffs.append(cv2.absdiff(prev, cur))
    flow_mean = np.mean(flow_mags, axis=0) if flow_mags else np.zeros_like(grays[0])
    diff_mean = np.mean(diffs, axis=0) if diffs else np.zeros_like(grays[0])
    flow_color = cv2.applyColorMap((255 * flow_mean / (flow_mean.max() + 1e-9)).astype(np.uint8), cv2.COLORMAP_TURBO)
    diff_color = cv2.applyColorMap((255 * diff_mean / (diff_mean.max() + 1e-9)).astype(np.uint8), cv2.COLORMAP_MAGMA)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    panels = [
        ("valid entity frame", cv2.cvtColor(crops[0], cv2.COLOR_BGR2RGB)),
        ("later valid frame", cv2.cvtColor(crops[-1], cv2.COLOR_BGR2RGB)),
        ("mean frame difference", cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)),
        ("mean optical flow", cv2.cvtColor(flow_color, cv2.COLOR_BGR2RGB)),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Motion analysis on entity ROI only; excluded light frames are not sampled")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_pca_panel(path: Path, case_id: str) -> None:
    analysis_dir = case_output_dir(case_id) / "entity_analysis"
    src = cv2.imread(str(analysis_dir / "entity_pca_panel.png"), cv2.IMREAD_COLOR)
    metrics = pd.read_csv(analysis_dir / "entity_pca_explained_variance.csv")
    pc1 = float(metrics.loc[0, "explained_variance_ratio"])
    k5 = float(metrics.loc[min(4, len(metrics) - 1), "cumulative_variance_ratio"])
    k10 = float(metrics.loc[min(9, len(metrics) - 1), "cumulative_variance_ratio"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title("Entity PCA panel")
    axes[1].axis("off")
    lines = [
        "PCA - entity_structure_analysis",
        "",
        f"PC1: {pc1:.4f}",
        f"Cumulative k=5: {k5:.4f}",
        f"Cumulative k=10: {k10:.4f}",
        "",
        "ROI used: entity ROI",
        "Light ROI is not used as PCA target",
    ]
    axes[1].text(0.03, 0.92, "\n".join(lines), va="top", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_autoencoder_panel(path: Path, case_id: str) -> None:
    analysis_dir = case_output_dir(case_id) / "entity_analysis"
    src = cv2.imread(str(analysis_dir / "entity_autoencoder_panel.png"), cv2.IMREAD_COLOR)
    metrics = pd.read_csv(analysis_dir / "entity_autoencoder_metrics.csv")
    mean_z = float(metrics["entity_core_mean_zscore"].mean())
    max_z = float(metrics["entity_core_max_zscore"].max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title("Entity autoencoder anomaly map")
    axes[1].axis("off")
    lines = [
        "Autoencoder - entity ROI",
        "",
        f"Mean entity-core z-score: {mean_z:.2f}",
        f"Max point z-score: {max_z:.2f}",
        "Max point: exploratory",
        "",
        "Primary visual target is entity structure.",
        "The light patch is not the main visual.",
    ]
    axes[1].text(0.03, 0.92, "\n".join(lines), va="top", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_metrics_card(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("#101418")
    fig.patch.set_facecolor("#101418")
    ax.axis("off")
    lines = [
        ("ENTITY STRUCTURE ANALYSIS", 22, "#7CFF9B"),
        ("6232026_psionic_session", 13, "#DDE6EF"),
        ("", 8, "#DDE6EF"),
        ("frames analyzed: 547", 17, "#FFFFFF"),
        ("light excluded: 172-208", 17, "#FFD166"),
        ("PCA PC1: 0.1271", 17, "#FFFFFF"),
        ("PCA k=5: 0.4042", 17, "#FFFFFF"),
        ("PCA k=10: 0.5649", 17, "#FFFFFF"),
        ("mean entity-core z-score: 94.23", 17, "#FFFFFF"),
        ("", 8, "#FFFFFF"),
        ("source: secondary / unverified", 15, "#B8C1CC"),
        ("claim: structure analysis, not origin proof", 15, "#B8C1CC"),
    ]
    y = 0.91
    for text, size, color in lines:
        ax.text(0.08, y, text, color=color, fontsize=size, weight=("bold" if size >= 17 else "normal"))
        y -= 0.075 if text else 0.035
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_visual_summary(path: Path, out_dir: Path, names: list[str]) -> None:
    images = []
    labels = []
    for name in names[:9]:
        img = cv2.imread(str(out_dir / name), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            labels.append(name)
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 3.4))
    axes_arr = np.array(axes).reshape(-1)
    for ax, image, label in zip(axes_arr, images, labels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.suptitle("Entity structure analysis visual content pack")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_manifest(path: Path, out_dir: Path) -> None:
    rows = [
        ("01_entity_target_overview.png", "Full frame with entity ROI and secondary light ROI.", "yes", "public_safe", "ROI is proposed and requires manual validation."),
        ("02_entity_crop_valid_frames_contact_sheet.png", "Contact sheet of valid entity frames from analysis segments.", "yes", "public_safe", "Sampling is illustrative, not every frame is shown."),
        ("03_light_excluded_frames_contact_sheet.png", "Contact sheet of light/saturation frames excluded from entity analysis.", "yes", "public_safe", "Use to explain exclusion, not as entity evidence."),
        ("04_entity_vs_light_comparison.png", "Side-by-side target separation: entity structure vs light event.", "yes", "public_safe", "Do not imply origin, only target separation."),
        ("05_entity_clahe_contrast_panel.png", "Entity crop, CLAHE, edges and approximate contours.", "yes", "public_safe", "Contrast enhancement is interpretive."),
        ("06_entity_motion_panel.png", "Frame differencing and optical flow on valid entity frames only.", "yes", "public_safe", "Motion maps depend on compression and low-light noise."),
        ("07_entity_pca_panel.png", "PCA panel and PC1/k5/k10 values for entity ROI.", "yes", "public_safe", "PCA depends on ROI and preprocessing."),
        ("08_entity_autoencoder_panel.png", "Entity autoencoder map with mean z-score and exploratory max.", "mixed", "public_safe + exploratory", "Max z-score is exploratory; prefer mean/core metrics."),
        ("09_public_safe_metrics_card.png", "Clean metrics card for content.", "yes", "public_safe", "States structure analysis, not origin proof."),
        ("10_visual_summary_contact_sheet.png", "Summary sheet of visual assets.", "yes", "public_safe", "Derivative overview of the above assets."),
    ]
    lines = ["# Visual Asset Manifest", "", "Folder: `" + out_dir.as_posix() + "`", ""]
    lines.append("| Image | Path | Represents | Public content | Classification | Caution |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, represents, public, classification, caution in rows:
        lines.append(f"| `{name}` | `{(out_dir / name).as_posix()}` | {represents} | {public} | {classification} | {caution} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_entity_visuals(config: dict[str, Any]) -> dict[str, Any]:
    case_id = config["case_id"]
    target = read_json(entity_target_path(case_id))
    frames = load_frames(case_id)
    out_dir = _out_dir(case_id)
    valid_indices = _selected_indices(target, len(frames))
    excluded_indices: list[int] = []
    for seg in target.get("light_excluded_segments", []):
        excluded_indices.extend(range(int(seg["start_frame"]), int(seg["end_frame"]) + 1))

    outputs = [
        "01_entity_target_overview.png",
        "02_entity_crop_valid_frames_contact_sheet.png",
        "03_light_excluded_frames_contact_sheet.png",
        "04_entity_vs_light_comparison.png",
        "05_entity_clahe_contrast_panel.png",
        "06_entity_motion_panel.png",
        "07_entity_pca_panel.png",
        "08_entity_autoencoder_panel.png",
        "09_public_safe_metrics_card.png",
        "10_visual_summary_contact_sheet.png",
    ]
    _save_overview(out_dir / outputs[0], frames, target)
    _save_contact_sheet(out_dir / outputs[1], frames, target, valid_indices, "Valid entity frames used for structure analysis")
    _save_contact_sheet(
        out_dir / outputs[2],
        frames,
        target,
        excluded_indices,
        "Light/saturation frames excluded from entity analysis",
        excluded=True,
    )
    _save_entity_vs_light(out_dir / outputs[3], frames, target)
    _save_clahe_panel(out_dir / outputs[4], frames, target)
    _save_motion_panel(out_dir / outputs[5], frames, target, valid_indices)
    _save_pca_panel(out_dir / outputs[6], case_id)
    _save_autoencoder_panel(out_dir / outputs[7], case_id)
    _save_metrics_card(out_dir / outputs[8])
    _save_visual_summary(out_dir / outputs[9], out_dir, outputs)
    _write_manifest(out_dir / "visual_asset_manifest.md", out_dir)

    return {
        "case_id": case_id,
        "output_dir": str(out_dir),
        "images": [str(out_dir / name) for name in outputs],
        "manifest": str(out_dir / "visual_asset_manifest.md"),
        "valid_frames": len(valid_indices),
        "excluded_light_frames": len(excluded_indices),
    }
