from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .frames import frame_paths, load_frames
from .io import read_json, write_json
from .paths import DATA_DIR, case_output_dir, case_report_dir, ensure_dir
from .visuals import save_image, save_panel


def _case_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "cases" / case_id)


def entity_target_path(case_id: str) -> Path:
    return _case_dir(case_id) / "entity_target.json"


def _roi_xywh(roi: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(roi.get("x", 0)),
        int(roi.get("y", 0)),
        int(roi.get("w", roi.get("width", 0))),
        int(roi.get("h", roi.get("height", 0))),
    )


def _segments_from_mask(mask: np.ndarray, min_len: int = 2) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start = None
    for idx, value in enumerate(mask.astype(bool).tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            end = idx - 1
            if end - start + 1 >= min_len:
                segments.append({"start_frame": int(start), "end_frame": int(end)})
            start = None
    return segments


def _detect_light_segments(frames: list[np.ndarray], entity_roi: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    x, y, w, h = _roi_xywh(entity_roi)
    saturated_pct = []
    high_percentile = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[y : y + h, x : x + w]
        saturated_pct.append(float((crop >= 245).mean()))
        high_percentile.append(float(np.percentile(crop, 99.5)))
    sat_arr = np.array(saturated_pct)
    p995_arr = np.array(high_percentile)
    sat_threshold = max(0.004, float(np.percentile(sat_arr, 98)))
    p995_threshold = max(230.0, float(np.percentile(p995_arr, 98)))
    light_mask = (sat_arr >= sat_threshold) | (p995_arr >= p995_threshold)
    margin = 8
    expanded = light_mask.copy()
    for idx, value in enumerate(light_mask):
        if value:
            expanded[max(0, idx - margin) : min(len(expanded), idx + margin + 1)] = True
    light_mask = expanded
    segments = _segments_from_mask(light_mask, min_len=2)
    for seg in segments:
        seg["reason"] = "saturated or high-brightness light event"
    thresholds = {
        "saturated_pixel_fraction_threshold": sat_threshold,
        "p99_5_luminance_threshold": p995_threshold,
        "temporal_margin_frames": float(margin),
    }
    return segments, thresholds


def _invert_segments(total: int, excluded: list[dict[str, Any]], min_len: int = 5) -> list[dict[str, Any]]:
    mask = np.ones(total, dtype=bool)
    for seg in excluded:
        start = max(0, int(seg["start_frame"]))
        end = min(total - 1, int(seg["end_frame"]))
        mask[start : end + 1] = False
    segments = _segments_from_mask(mask, min_len=min_len)
    for idx, seg in enumerate(segments):
        seg["reason"] = "entity visible in non-saturated frames" if idx else "entity visible before light"
    return segments


def _load_pipeline_roi(case_id: str) -> dict[str, Any] | None:
    path = DATA_DIR / "roi" / case_id / "roi.json"
    if not path.exists():
        return None
    return read_json(path).get("roi")


def _default_entity_roi(frames: list[np.ndarray]) -> dict[str, Any]:
    h, w = frames[0].shape[:2]
    return {
        "x": int(w * 0.15),
        "y": int(h * 0.06),
        "w": int(w * 0.70),
        "h": int(h * 0.88),
        "note": "Proposed broad ROI for full entity / structure; validate manually.",
    }


def build_entity_target(case_id: str, frames: list[np.ndarray]) -> dict[str, Any]:
    pipeline_roi = _load_pipeline_roi(case_id)
    light_roi = {
        "x": int(pipeline_roi.get("x", 278)) if pipeline_roi else 278,
        "y": int(pipeline_roi.get("y", 140)) if pipeline_roi else 140,
        "w": int(pipeline_roi.get("width", 81)) if pipeline_roi else 81,
        "h": int(pipeline_roi.get("height", 100)) if pipeline_roi else 100,
        "note": "Secondary light event reference only; not primary entity target.",
    }
    entity_roi = _default_entity_roi(frames)
    light_segments, thresholds = _detect_light_segments(frames, entity_roi)
    analysis_segments = _invert_segments(len(frames), light_segments)
    target = {
        "case_id": case_id,
        "primary_target": "entity_structure",
        "exclude_saturated_light": True,
        "analysis_pauses_on_light": True,
        "entity_roi": entity_roi,
        "light_roi": light_roi,
        "analysis_segments": analysis_segments,
        "light_excluded_segments": light_segments,
        "thresholds": thresholds,
        "notes": [
            "Entity ROI is a conservative broad proposal, not a final manual annotation.",
            "Pipeline ROI by brightness is retained only as light/event reference.",
            "Saturated-light frames are excluded from primary entity analysis.",
        ],
    }
    write_json(entity_target_path(case_id), target)
    return target


def load_or_create_entity_target(case_id: str) -> dict[str, Any]:
    path = entity_target_path(case_id)
    if path.exists():
        return read_json(path)
    return build_entity_target(case_id, load_frames(case_id))


def _frame_at(frames: list[np.ndarray], idx: int) -> np.ndarray:
    return frames[max(0, min(len(frames) - 1, idx))]


def _draw_rect(ax: Any, roi: dict[str, Any], label: str, color: str) -> None:
    x, y, w, h = _roi_xywh(roi)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.5))
    ax.text(x, max(0, y - 5), label, color=color, fontsize=9, weight="bold")


def _save_target_panel(case_id: str, frames: list[np.ndarray], target: dict[str, Any], out_dir: Path) -> Path:
    first_seg = target["analysis_segments"][0] if target["analysis_segments"] else {"start_frame": 0}
    first_light = target["light_excluded_segments"][0] if target["light_excluded_segments"] else {"start_frame": 0}
    entity_idx = int(first_seg["start_frame"])
    light_idx = int(first_light["start_frame"])
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(cv2.cvtColor(_frame_at(frames, entity_idx), cv2.COLOR_BGR2RGB))
    _draw_rect(ax0, target["entity_roi"], "intended target: entity structure", "lime")
    _draw_rect(ax0, target["light_roi"], "previous target: light/saturation", "red")
    ax0.set_title(f"Entity-visible frame {entity_idx}")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(cv2.cvtColor(_frame_at(frames, light_idx), cv2.COLOR_BGR2RGB))
    _draw_rect(ax1, target["light_roi"], "light ROI", "red")
    ax1.set_title(f"Light/fogonazo frame {light_idx}")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(cv2.cvtColor(_frame_at(frames, entity_idx), cv2.COLOR_BGR2RGB))
    _draw_rect(ax2, target["entity_roi"], "entity ROI", "lime")
    ax2.set_title("Proposed full-structure ROI")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis("off")
    lines = [
        "The saturated light is not the primary target",
        "",
        "Previous target: light / saturation",
        "- auto-brightness ROI",
        "- max luminance",
        "- patch-white z-scores",
        "",
        "Intended target: entity structure",
        "- silhouette / mass / contours",
        "- texture / persistence",
        "- only non-saturated frames",
        "",
        f"Entity segments: {len(target['analysis_segments'])}",
        f"Light excluded: {len(target['light_excluded_segments'])}",
    ]
    ax3.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12)
    fig.tight_layout()
    path = out_dir / "target_audit_panel.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _classify_path(path: Path) -> tuple[str, list[str]]:
    text = str(path).lower()
    reasons = []
    target = "unknown"
    light_terms = ["debug_zscore", "central_brightness", "brightness", "auto_current", "max_projection"]
    entity_terms = ["manual_center", "base", "pca", "motion", "optical_flow", "mean_projection"]
    if any(term in text for term in light_terms):
        target = "light"
        reasons.append("name/path references brightness, auto-brightness ROI, or z-score debug")
    if any(term in text for term in entity_terms):
        if target == "light":
            target = "mixed"
        else:
            target = "entity"
        reasons.append("name/path can describe structure, PCA, motion, or non-max aggregate")
    if "autoencoder" in text:
        target = "mixed" if target in {"light", "unknown"} else target
        reasons.append("autoencoder validity depends on ROI and background definition")
    if "zscore_debug_grid.csv" in text:
        target = "light"
        reasons.append("contains patch-extreme/light-sensitive z-score variants")
    return target, reasons or ["insufficient metadata for confident classification"]


def _scan_outputs(case_id: str) -> pd.DataFrame:
    roots = [case_output_dir(case_id), case_report_dir(case_id), DATA_DIR / "roi" / case_id]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            classification, reasons = _classify_path(path)
            rows.append(
                {
                    "path": str(path.relative_to(DATA_DIR)),
                    "classification": classification,
                    "dependency_flags": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _summarize_existing_metrics(case_id: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    roi = _load_pipeline_roi(case_id)
    if roi:
        summary["pipeline_roi"] = roi
    pca_csv = case_output_dir(case_id) / "pca" / "pca_explained_variance.csv"
    if pca_csv.exists():
        df = pd.read_csv(pca_csv)
        summary["pca_k5"] = float(df.loc[min(4, len(df) - 1), "cumulative_variance_ratio"])
        summary["pca_k10"] = float(df.loc[min(9, len(df) - 1), "cumulative_variance_ratio"])
    ae_csv = case_output_dir(case_id) / "autoencoder" / "autoencoder_metrics.csv"
    if ae_csv.exists():
        df = pd.read_csv(ae_csv)
        summary["autoencoder_zscore_mean"] = float(df["object_error_zscore"].mean())
        summary["autoencoder_zscore_max"] = float(df["object_error_zscore"].max())
    debug_csv = case_output_dir(case_id) / "debug_zscore" / "zscore_debug_grid.csv"
    if debug_csv.exists():
        df = pd.read_csv(debug_csv)
        summary["debug_top_zscore"] = float(df["zscore"].max())
        summary["debug_unstable_rows"] = int((df["metric_classification"] == "unstable").sum())
    return summary


def audit_analysis_target(config: dict[str, Any]) -> dict[str, Any]:
    case_id = config["case_id"]
    frames = load_frames(case_id)
    target = build_entity_target(case_id, frames)
    out_dir = ensure_dir(case_output_dir(case_id) / "target_audit")
    report_dir = ensure_dir(case_report_dir(case_id))
    panel = _save_target_panel(case_id, frames, target, out_dir)
    classifications = _scan_outputs(case_id)
    classifications.to_csv(out_dir / "output_target_classification.csv", index=False)
    summary = _summarize_existing_metrics(case_id)

    source_path = DATA_DIR / "sources" / f"{case_id}.source.json"
    source = read_json(source_path) if source_path.exists() else {}
    counts = classifications["classification"].value_counts().to_dict() if not classifications.empty else {}
    report = report_dir / "analysis_target_audit.md"
    content = f"""# Analysis Target Audit: {case_id}

## Objetivo

Esta auditoria separa formalmente `light_event_analysis` de `entity_structure_analysis`. El objetivo principal del caso pasa a ser la entidad/estructura visible en frames no saturados. La luz/fogonazo queda como evento secundario o depuracion.

## Fuente

- Tipo: `{source.get("source_type", config.get("source_type", "unknown"))}`
- SHA256: `{source.get("sha256", "pendiente")}`
- Frames revisados: `{len(frames)}`
- Resolucion: `{frames[0].shape[1]}x{frames[0].shape[0]}`

## Configuracion creada

- Archivo: `{entity_target_path(case_id).as_posix()}`
- Primary target: `entity_structure`
- Entity ROI propuesta: `{target["entity_roi"]}`
- Light ROI secundaria: `{target["light_roi"]}`
- Segmentos de entidad: `{target["analysis_segments"]}`
- Segmentos excluidos por luz: `{target["light_excluded_segments"]}`

La ROI de entidad es una propuesta amplia y editable. Debe validarse manualmente antes de usarla como anotacion final.

## Clasificacion de outputs anteriores

Conteo: `{counts}`

CSV completo: `{(out_dir / "output_target_classification.csv").as_posix()}`

## Que estaba centrado en luz/brillo

- La ROI automatica anterior se obtuvo por brillo: `{summary.get("pipeline_roi", "sin ROI previa")}`.
- La depuracion `debug_zscore` contiene variantes patch-extreme y fondos oscuros que pueden dispararse por patch blanco, saturacion o desviacion de fondo casi cero.
- Cualquier metrica basada en `central_brightness_peak`, maximo brillo, patch saturado o fondo negro estricto debe reclasificarse como `light_event_analysis` o `mixed`, no como prueba principal de entidad.

## Que puede seguir siendo util para entidad

- PCA y correlaciones pueden seguir siendo utiles si se recalculan sobre una ROI de entidad y frames no saturados.
- Optical flow, frame differencing, contornos y textura son utiles si excluyen segmentos `light_on`.
- Autoencoder puede usarse para entidad si sus patches de fondo excluyen la ROI de entidad y los frames saturados.

## Que debe rehacerse

- PCA principal del caso.
- Autoencoder principal del caso.
- Motion/frame differencing principal.
- Paneles y reporte principal centrados en entidad.

## Por que diferia del analisis visual manual

El analisis automatizado inicial persiguio el maximo de brillo y por eso sus ROI, z-score y algunos paneles quedaron sesgados hacia la luz/fogonazo. El analisis visual manual estaba intentando seguir la estructura completa. La diferencia no implica que una ruta sea falsa; indica que estaban midiendo objetivos distintos.

## Conclusion

Los outputs anteriores deben conservarse como trazabilidad, pero el claim principal del caso debe migrar a `entity_structure_analysis`. La luz/fogonazo es un evento excluido o secundario. La fuente sigue siendo secundaria/no verificada y no se debe afirmar origen.

## Artefactos

- Panel: `{panel.as_posix()}`
- Config entity target: `{entity_target_path(case_id).as_posix()}`
"""
    report.write_text(content, encoding="utf-8")
    return {
        "case_id": case_id,
        "report": str(report),
        "panel": str(panel),
        "entity_target": str(entity_target_path(case_id)),
        "classification_csv": str(out_dir / "output_target_classification.csv"),
        "classification_counts": counts,
        "light_bias_confirmed": bool(summary.get("pipeline_roi")),
    }


def _selected_indices(target: dict[str, Any], total: int) -> list[int]:
    indices: list[int] = []
    for seg in target.get("analysis_segments", []):
        start = max(0, int(seg["start_frame"]))
        end = min(total - 1, int(seg["end_frame"]))
        indices.extend(range(start, end + 1))
    excluded = set()
    for seg in target.get("light_excluded_segments", []):
        excluded.update(range(max(0, int(seg["start_frame"])), min(total - 1, int(seg["end_frame"])) + 1))
    return [idx for idx in sorted(set(indices)) if idx not in excluded]


def _crop_entity_frames(frames: list[np.ndarray], target: dict[str, Any]) -> tuple[list[np.ndarray], list[int]]:
    x, y, w, h = _roi_xywh(target["entity_roi"])
    indices = _selected_indices(target, len(frames))
    crops = [frames[idx][y : y + h, x : x + w].copy() for idx in indices]
    return crops, indices


def _gray_resized(crop: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _run_entity_base(case_id: str, crops: list[np.ndarray], indices: list[int], out_dir: Path) -> dict[str, Any]:
    grays = np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crops]).astype(np.float32)
    mean_projection = grays.mean(axis=0)
    max_projection = grays.max(axis=0)
    diff_mean = np.abs(np.diff(grays, axis=0)).mean(axis=0) if len(grays) > 1 else np.zeros_like(mean_projection)
    first = crops[0]
    lab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    edges = cv2.Canny(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), 35, 120)
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(mean_projection))))
    fft = 255 * (fft - fft.min()) / (np.ptp(fft) + 1e-6)
    save_panel(
        out_dir / "entity_base_panel.png",
        {
            "mean entity": mean_projection,
            "max entity": max_projection,
            "frame difference": diff_mean,
            "clahe": clahe,
            "edges": edges,
            "fft": fft,
        },
    )
    pd.DataFrame(
        {
            "source_frame": indices,
            "mean_luminance": grays.mean(axis=(1, 2)),
            "max_luminance": grays.max(axis=(1, 2)),
            "std_luminance": grays.std(axis=(1, 2)),
        }
    ).to_csv(out_dir / "entity_luminance_metrics.csv", index=False)
    return {"frames": len(crops), "mean_luminance": float(grays.mean())}


def _run_entity_motion(crops: list[np.ndarray], indices: list[int], out_dir: Path) -> dict[str, Any]:
    grays = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crops]
    flow_mags = []
    diff_mags = []
    for prev, cur in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(mag)
        diff_mags.append(cv2.absdiff(prev, cur))
    flow_mean = np.mean(flow_mags, axis=0) if flow_mags else np.zeros_like(grays[0], dtype=np.float32)
    diff_mean = np.mean(diff_mags, axis=0) if diff_mags else np.zeros_like(grays[0], dtype=np.float32)
    save_panel(
        out_dir / "entity_motion_panel.png",
        {
            "first entity": crops[0],
            "mean frame difference": diff_mean,
            "mean optical flow": 255 * flow_mean / (flow_mean.max() + 1e-9),
        },
    )
    rows = []
    for idx, (flow_mag, diff_mag) in enumerate(zip(flow_mags, diff_mags), start=1):
        rows.append(
            {
                "source_frame": indices[idx],
                "mean_frame_difference": float(np.mean(diff_mag)),
                "mean_optical_flow_magnitude": float(np.mean(flow_mag)),
                "max_optical_flow_magnitude": float(np.max(flow_mag)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "entity_motion_metrics.csv", index=False)
    return {"motion_rows": len(rows), "mean_flow": float(np.mean(flow_mean))}


def _run_entity_pca(case_id: str, crops: list[np.ndarray], indices: list[int], out_dir: Path) -> dict[str, Any]:
    matrix = np.stack([_gray_resized(crop).reshape(-1) for crop in crops]).astype(np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    normalized = scaler.fit_transform(matrix)
    n_components = min(20, normalized.shape[0], normalized.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(normalized)
    recon = scaler.inverse_transform(pca.inverse_transform(scores))
    residual = np.abs(matrix - recon)
    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(out_dir / "entity_pca_explained_variance.csv", index=False)
    metrics = pd.DataFrame(
        {
            "source_frame": indices,
            "entity_mean": matrix.mean(axis=1),
            "entity_std": matrix.std(axis=1),
            "pca_residual_mean": residual.mean(axis=1),
            "pca_residual_max": residual.max(axis=1),
        }
    )
    mean_signature = matrix.mean(axis=0)
    metrics["corr_entity_vs_mean_signature"] = [
        np.corrcoef(row, mean_signature)[0, 1] if np.std(row) and np.std(mean_signature) else np.nan
        for row in matrix
    ]
    metrics["corr_frame_to_previous"] = np.nan
    for i in range(1, len(matrix)):
        metrics.loc[i, "corr_frame_to_previous"] = np.corrcoef(matrix[i - 1], matrix[i])[0, 1]
    metrics.to_csv(out_dir / "entity_pca_metrics.csv", index=False)
    panels: dict[str, np.ndarray] = {
        "mean entity": matrix.mean(axis=0).reshape(64, 64),
        "mean residual": residual.mean(axis=0).reshape(64, 64) * 4,
    }
    for idx in range(min(3, n_components)):
        comp = pca.components_[idx].reshape(64, 64)
        panels[f"PC{idx + 1}"] = 255 * (comp - comp.min()) / (np.ptp(comp) + 1e-6)
    save_panel(out_dir / "entity_pca_panel.png", panels)
    return {
        "components": int(n_components),
        "pc1": float(explained.loc[0, "explained_variance_ratio"]),
        "k5": float(explained.loc[min(4, len(explained) - 1), "cumulative_variance_ratio"]),
        "k10": float(explained.loc[min(9, len(explained) - 1), "cumulative_variance_ratio"]),
    }


def _extract_entity_background_patches(crops: list[np.ndarray], patch_size: int = 16, max_patches: int = 3500) -> np.ndarray:
    records = []
    for frame_idx, crop in enumerate(crops):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        cx0, cx1 = int(w * 0.28), int(w * 0.72)
        cy0, cy1 = int(h * 0.22), int(h * 0.78)
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                overlaps_entity_core = x < cx1 and x + patch_size > cx0 and y < cy1 and y + patch_size > cy0
                patch = gray[y : y + patch_size, x : x + patch_size]
                if not overlaps_entity_core and patch.max() < 0.94:
                    records.append(patch)
    if not records:
        raise RuntimeError("No entity-background patches available")
    if len(records) > max_patches:
        rng = np.random.default_rng(42)
        selected = rng.choice(len(records), size=max_patches, replace=False)
        records = [records[int(i)] for i in selected]
    return np.stack(records).astype(np.float32)


def _run_entity_autoencoder(crops: list[np.ndarray], indices: list[int], out_dir: Path, epochs: int = 6) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    np.random.seed(42)
    patch_size = 16
    patches = _extract_entity_background_patches(crops, patch_size=patch_size)
    x_train = torch.tensor(patches.reshape(len(patches), -1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train), batch_size=min(128, len(x_train)), shuffle=True)
    dim = x_train.shape[1]
    model = nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, dim),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(x_train))
    bg_errors = []
    with torch.no_grad():
        for patch in patches[: min(2000, len(patches))]:
            vector = torch.tensor(patch.reshape(1, -1), dtype=torch.float32)
            recon = model(vector).numpy().reshape(patch_size, patch_size)
            bg_errors.append(float(np.mean((patch - recon) ** 2)))
    bg_errors_arr = np.array(bg_errors)
    bg_mean = float(bg_errors_arr.mean())
    bg_std = float(bg_errors_arr.std() + 1e-12)
    anomaly_maps = []
    rows = []
    for frame_idx, crop in zip(indices, crops):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        err_map = np.zeros_like(gray)
        counts = np.zeros_like(gray)
        with torch.no_grad():
            for y in range(0, h - patch_size + 1, patch_size // 2):
                for x in range(0, w - patch_size + 1, patch_size // 2):
                    patch = gray[y : y + patch_size, x : x + patch_size]
                    vector = torch.tensor(patch.reshape(1, -1), dtype=torch.float32)
                    recon = model(vector).numpy().reshape(patch_size, patch_size)
                    err = (patch - recon) ** 2
                    err_map[y : y + patch_size, x : x + patch_size] += err
                    counts[y : y + patch_size, x : x + patch_size] += 1
        err_map = err_map / np.maximum(counts, 1)
        anomaly_maps.append(err_map)
        core = err_map[int(h * 0.22) : int(h * 0.78), int(w * 0.28) : int(w * 0.72)]
        obj_mean = float(core.mean())
        obj_max = float(core.max())
        rows.append(
            {
                "source_frame": frame_idx,
                "background_mean_error": bg_mean,
                "background_std_error": bg_std,
                "entity_core_mean_error": obj_mean,
                "entity_core_max_error": obj_max,
                "entity_core_mean_zscore": (obj_mean - bg_mean) / bg_std,
                "entity_core_max_zscore": (obj_max - bg_mean) / bg_std,
                "entity_core_mean_percentile": float((bg_errors_arr < obj_mean).mean() * 100),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "entity_autoencoder_metrics.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, epochs + 1), "loss": losses}).to_csv(out_dir / "entity_autoencoder_training_curve.csv", index=False)
    mean_map = np.mean(anomaly_maps, axis=0)
    save_image(out_dir / "entity_anomaly_map_mean.png", 255 * mean_map / (mean_map.max() + 1e-12))
    save_panel(
        out_dir / "entity_autoencoder_panel.png",
        {
            "mean entity": np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops], axis=0),
            "mean anomaly": 255 * mean_map / (mean_map.max() + 1e-12),
            "max anomaly": 255 * np.max(anomaly_maps, axis=0) / (np.max(anomaly_maps) + 1e-12),
        },
    )
    return {
        "patches": int(len(patches)),
        "epochs": epochs,
        "background_mean_error": bg_mean,
        "background_std_error": bg_std,
        "mean_entity_zscore": float(pd.DataFrame(rows)["entity_core_mean_zscore"].mean()),
        "max_mean_entity_zscore": float(pd.DataFrame(rows)["entity_core_mean_zscore"].max()),
        "max_entity_zscore": float(pd.DataFrame(rows)["entity_core_max_zscore"].max()),
        "mean_entity_percentile": float(pd.DataFrame(rows)["entity_core_mean_percentile"].mean()),
    }


def run_entity_analysis(config: dict[str, Any]) -> dict[str, Any]:
    case_id = config["case_id"]
    frames = load_frames(case_id)
    target = load_or_create_entity_target(case_id)
    out_dir = ensure_dir(case_output_dir(case_id) / "entity_analysis")
    crops, indices = _crop_entity_frames(frames, target)
    if len(crops) < 5:
        raise RuntimeError("Not enough non-saturated entity frames for entity analysis")
    base = _run_entity_base(case_id, crops, indices, out_dir)
    motion = _run_entity_motion(crops, indices, out_dir)
    pca = _run_entity_pca(case_id, crops, indices, out_dir)
    ae = _run_entity_autoencoder(crops, indices, out_dir)
    report_path = ensure_dir(case_report_dir(case_id)) / "entity_analysis_report.md"
    source_path = DATA_DIR / "sources" / f"{case_id}.source.json"
    source = read_json(source_path) if source_path.exists() else {}
    content = f"""# Entity Structure Analysis: {case_id}

## Scope

This report is the primary `entity_structure_analysis` pass. It uses `entity_target.json`, excludes saturated/light-on frames, and treats the light/fogonazo as a secondary event rather than the target.

## Source

- Source type: `{source.get("source_type", config.get("source_type", "unknown"))}`
- SHA256: `{source.get("sha256", "pending")}`
- Verification: secondary/unverified source; no origin claim is made.

## Target Configuration

- Entity ROI: `{target["entity_roi"]}`
- Light ROI, secondary only: `{target["light_roi"]}`
- Analysis segments: `{target["analysis_segments"]}`
- Excluded light segments: `{target["light_excluded_segments"]}`
- Frames analyzed: `{len(indices)}`

## Results

### Base / Structure

- Mean luminance over entity crops: `{base["mean_luminance"]:.4f}`
- Panel: `{(out_dir / "entity_base_panel.png").as_posix()}`

### Motion

- Motion rows: `{motion["motion_rows"]}`
- Mean optical-flow magnitude: `{motion["mean_flow"]:.6f}`
- Panel: `{(out_dir / "entity_motion_panel.png").as_posix()}`

### PCA

- PC1: `{pca["pc1"]:.6f}`
- Cumulative k=5: `{pca["k5"]:.6f}`
- Cumulative k=10: `{pca["k10"]:.6f}`
- Panel: `{(out_dir / "entity_pca_panel.png").as_posix()}`

### Autoencoder

- Background patches used: `{ae["patches"]}`
- Background mean error: `{ae["background_mean_error"]:.8f}`
- Background std error: `{ae["background_std_error"]:.8f}`
- Mean entity-core z-score, averaged over valid frames: `{ae["mean_entity_zscore"]:.4f}`
- Max mean entity-core z-score: `{ae["max_mean_entity_zscore"]:.4f}`
- Mean entity-core percentile vs background: `{ae["mean_entity_percentile"]:.4f}`
- Max entity-core z-score: `{ae["max_entity_zscore"]:.4f}` (`exploratory`; sensitive to single-pixel/patch extrema)
- Panel: `{(out_dir / "entity_autoencoder_panel.png").as_posix()}`

## Interpretation

The earlier automated analysis remains useful as `light_event / mixed target analysis` where it depends on brightness, saturation, patch-white behavior or low-variance dark backgrounds. This pass is the primary entity-centered analysis. It measures persistence, structure, contours, motion and reconstruction behavior in non-saturated frames only.

## Limitations

- The entity ROI is broad and should be manually validated.
- The source is secondary/unverified.
- Metrics are method-dependent and do not prove origin.
- Light/fogonazo frames are excluded from the primary entity claim.
"""
    report_path.write_text(content, encoding="utf-8")
    return {
        "case_id": case_id,
        "output_dir": str(out_dir),
        "report": str(report_path),
        "frames_analyzed": len(indices),
        "entity_roi": target["entity_roi"],
        "pca": pca,
        "autoencoder": ae,
    }
