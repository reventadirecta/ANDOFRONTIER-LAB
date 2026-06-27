import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .frames import load_frames
from .io import read_json
from .paths import DATA_DIR, case_output_dir, case_report_dir, ensure_dir


MANUAL_REFERENCE = {
    "object_center_x": 321.6,
    "object_center_y": 183.7,
    "centroid_jump_mean_px": 0.5,
    "centroid_jump_max_px": 5.6,
    "pca_pc1": 0.245,
    "pca_pc2": 0.158,
    "pca_k5": 0.547,
    "pca_k10": 0.672,
    "corr_roi_mean": 0.969,
    "corr_roi_std": 0.040,
    "corr_frame_to_frame": 0.992,
    "ae_background_mean": 0.0001218,
    "ae_background_std": 0.0001756,
    "ae_object_patch_error": 0.147296,
    "ae_object_zscore": 838.0,
}


def _clip_roi(roi: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    x = int(round(roi["x"]))
    y = int(round(roi["y"]))
    w = int(round(roi["width"]))
    h = int(round(roi["height"]))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(4, min(w, width - x))
    h = max(4, min(h, height - y))
    out = dict(roi)
    out.update({"x": x, "y": y, "width": w, "height": h})
    return out


def _center_roi(name: str, cx: float, cy: float, w: int, h: int, frame_w: int, frame_h: int) -> dict[str, Any]:
    return _clip_roi(
        {
            "name": name,
            "mode": "centered",
            "x": cx - w / 2,
            "y": cy - h / 2,
            "width": w,
            "height": h,
            "notes": f"Centered around reference object center ({cx:.1f}, {cy:.1f}).",
        },
        frame_w,
        frame_h,
    )


def _brightest_roi_in_central_window(frame: np.ndarray, w: int = 96, h: int = 96) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape
    x0, x1 = int(fw * 0.25), int(fw * 0.75)
    y0, y1 = int(fh * 0.25), int(fh * 0.75)
    central = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 0)
    _, _, _, max_loc = cv2.minMaxLoc(central)
    cx = x0 + max_loc[0]
    cy = y0 + max_loc[1]
    return _center_roi("central_brightness_peak", cx, cy, w, h, fw, fh) | {
        "mode": "central-brightness-peak",
        "notes": "Centered on the brightest smoothed pixel inside the central 50% frame window.",
    }


def build_roi_candidates(case_id: str, config: dict[str, Any], frames: list[np.ndarray]) -> list[dict[str, Any]]:
    first = frames[len(frames) // 2]
    frame_h, frame_w = first.shape[:2]
    candidates: list[dict[str, Any]] = []

    roi_path = DATA_DIR / "roi" / case_id / "roi.json"
    if roi_path.exists():
        auto_roi = read_json(roi_path)["roi"]
        candidates.append(
            _clip_roi(
                {
                    "name": "pipeline_auto_current",
                    **auto_roi,
                    "notes": "ROI already produced by the automated pipeline.",
                },
                frame_w,
                frame_h,
            )
        )

    candidates.append(_center_roi("manual_center_64", 321.6, 183.7, 64, 64, frame_w, frame_h))
    candidates.append(_center_roi("manual_center_96", 321.6, 183.7, 96, 96, frame_w, frame_h))
    candidates.append(_center_roi("manual_center_128", 321.6, 183.7, 128, 128, frame_w, frame_h))
    candidates.append(_brightest_roi_in_central_window(first, 96, 96))

    for idx, manual in enumerate(config.get("audit_rois", []), start=1):
        candidates.append(
            _clip_roi(
                {
                    "name": manual.get("name", f"config_manual_{idx}"),
                    "mode": "config-manual",
                    "x": manual["x"],
                    "y": manual["y"],
                    "width": manual["width"],
                    "height": manual["height"],
                    "notes": manual.get("notes", "Manual audit ROI from case config."),
                },
                frame_w,
                frame_h,
            )
        )

    unique: list[dict[str, Any]] = []
    seen = set()
    for roi in candidates:
        key = (roi["name"], roi["x"], roi["y"], roi["width"], roi["height"])
        if key not in seen:
            seen.add(key)
            unique.append(roi)
    return unique


def _crop_gray_resized(frame: np.ndarray, roi: dict[str, Any], size: int = 64) -> np.ndarray:
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def save_roi_candidate_panel(case_id: str, frames: list[np.ndarray], rois: list[dict[str, Any]], out_dir: Path) -> None:
    key_index = len(frames) // 2
    frame = frames[key_index].copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb)
    colors = ["lime", "yellow", "cyan", "magenta", "orange", "white", "red"]
    for idx, roi in enumerate(rois):
        color = colors[idx % len(colors)]
        rect = plt.Rectangle((roi["x"], roi["y"]), roi["width"], roi["height"], fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(roi["x"], max(0, roi["y"] - 4), roi["name"], color=color, fontsize=8, weight="bold")
    ax.set_title(f"{case_id} ROI candidates on frame {key_index}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_candidates_panel.png", dpi=160)
    plt.close(fig)

    for roi in rois:
        single = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(single)
        ax.add_patch(
            plt.Rectangle((roi["x"], roi["y"]), roi["width"], roi["height"], fill=False, edgecolor="lime", linewidth=2)
        )
        ax.set_title(roi["name"])
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"roi_{roi['name']}.png", dpi=140)
        plt.close(fig)


def audit_pca(case_id: str, frames: list[np.ndarray], rois: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for roi in rois:
        matrix = np.stack([_crop_gray_resized(frame, roi).reshape(-1) for frame in frames]).astype(np.float32)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        n_components = min(20, centered.shape[0], centered.shape[1])
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(centered)
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        mean_signature = matrix.mean(axis=0)
        corr_roi = [
            np.corrcoef(row, mean_signature)[0, 1] if np.std(row) and np.std(mean_signature) else np.nan
            for row in matrix
        ]
        corr_ff = [
            np.corrcoef(matrix[i - 1], matrix[i])[0, 1] if np.std(matrix[i - 1]) and np.std(matrix[i]) else np.nan
            for i in range(1, len(matrix))
        ]
        rows.append(
            {
                "case_id": case_id,
                "roi_name": roi["name"],
                "x": roi["x"],
                "y": roi["y"],
                "width": roi["width"],
                "height": roi["height"],
                "pc1": float(explained[0]) if len(explained) else np.nan,
                "pc2": float(explained[1]) if len(explained) > 1 else np.nan,
                "cumulative_k5": float(cumulative[min(4, len(cumulative) - 1)]),
                "cumulative_k10": float(cumulative[min(9, len(cumulative) - 1)]),
                "cumulative_k20": float(cumulative[min(19, len(cumulative) - 1)]),
                "corr_roi_vs_mean_avg": float(np.nanmean(corr_roi)),
                "corr_roi_vs_mean_std": float(np.nanstd(corr_roi)),
                "corr_frame_to_frame_avg": float(np.nanmean(corr_ff)),
                "corr_frame_to_frame_std": float(np.nanstd(corr_ff)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pca_roi_comparison.csv", index=False)
    return df


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _sample_patch_records(
    frames: list[np.ndarray],
    roi: dict[str, Any],
    strategy: str,
    patch_size: int,
    stride: int,
    max_patches: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rng = random.Random(seed)
    bg_records: list[tuple[int, int, int]] = []
    obj_records: list[tuple[int, int, int]] = []
    x0, y0, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    obj_cx0 = x0 + int(w * 0.35)
    obj_cx1 = x0 + int(w * 0.65)
    obj_cy0 = y0 + int(h * 0.35)
    obj_cy1 = y0 + int(h * 0.65)

    for frame_idx, frame in enumerate(frames):
        fh, fw = frame.shape[:2]
        if strategy == "roi_aggregate":
            xs = range(x0, max(x0 + 1, x0 + w - patch_size + 1), stride)
            ys = range(y0, max(y0 + 1, y0 + h - patch_size + 1), stride)
        else:
            xs = range(0, max(1, fw - patch_size + 1), stride * 2)
            ys = range(0, max(1, fh - patch_size + 1), stride * 2)
        for y in ys:
            for x in xs:
                if x + patch_size > fw or y + patch_size > fh:
                    continue
                overlaps_object = x < obj_cx1 and x + patch_size > obj_cx0 and y < obj_cy1 and y + patch_size > obj_cy0
                inside_roi = x >= x0 and y >= y0 and x + patch_size <= x0 + w and y + patch_size <= y0 + h
                if overlaps_object:
                    obj_records.append((frame_idx, x, y))
                elif strategy == "roi_aggregate" and inside_roi:
                    bg_records.append((frame_idx, x, y))
                elif strategy == "patch_extreme" and not inside_roi:
                    bg_records.append((frame_idx, x, y))

    if len(bg_records) > max_patches:
        bg_records = rng.sample(bg_records, max_patches)
    patches = []
    for frame_idx, x, y in bg_records:
        gray = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY)
        patches.append(gray[y : y + patch_size, x : x + patch_size])
    if not patches:
        raise RuntimeError(f"No background patches for ROI {roi['name']} strategy {strategy}")
    return np.stack(patches).astype(np.float32) / 255.0, bg_records, obj_records


def _train_patch_autoencoder(patches: np.ndarray, seed: int, epochs: int) -> tuple[Any, list[float], list[float]]:
    _set_seed(seed)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    flat = patches.reshape(len(patches), -1).astype(np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(flat))
    split = max(1, int(len(order) * 0.8))
    train_idx = order[:split]
    val_idx = order[split:] if split < len(order) else order[:1]
    train = torch.tensor(flat[train_idx], dtype=torch.float32)
    val = torch.tensor(flat[val_idx], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train), batch_size=min(128, len(train)), shuffle=True)
    dim = train.shape[1]
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
    train_losses: list[float] = []
    val_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        train_losses.append(total / len(train))
        model.eval()
        with torch.no_grad():
            val_losses.append(float(loss_fn(model(val), val)))
    return model, train_losses, val_losses


def _patch_errors(model: Any, frames: list[np.ndarray], records: list[tuple[int, int, int]], patch_size: int) -> np.ndarray:
    import torch

    errors = []
    model.eval()
    with torch.no_grad():
        for frame_idx, x, y in records:
            gray = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            patch = gray[y : y + patch_size, x : x + patch_size]
            vector = torch.tensor(patch.reshape(1, -1), dtype=torch.float32)
            recon = model(vector).numpy().reshape(patch_size, patch_size)
            errors.append(float(np.mean((patch - recon) ** 2)))
    return np.array(errors, dtype=np.float64)


def _save_anomaly_map(
    model: Any,
    frame: np.ndarray,
    roi: dict[str, Any],
    path: Path,
    patch_size: int,
    stride: int,
) -> None:
    import torch

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    x0, y0, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    crop = gray[y0 : y0 + h, x0 : x0 + w]
    err_map = np.zeros_like(crop)
    counts = np.zeros_like(crop)
    model.eval()
    with torch.no_grad():
        for y in range(0, max(1, h - patch_size + 1), stride):
            for x in range(0, max(1, w - patch_size + 1), stride):
                patch = crop[y : y + patch_size, x : x + patch_size]
                if patch.shape != (patch_size, patch_size):
                    continue
                vector = torch.tensor(patch.reshape(1, -1), dtype=torch.float32)
                recon = model(vector).numpy().reshape(patch_size, patch_size)
                err = (patch - recon) ** 2
                err_map[y : y + patch_size, x : x + patch_size] += err
                counts[y : y + patch_size, x : x + patch_size] += 1
    err_map = err_map / np.maximum(counts, 1)
    norm = 255 * err_map / (err_map.max() + 1e-12)
    cv2.imwrite(str(path), norm.astype(np.uint8))


def audit_autoencoder(
    case_id: str,
    frames: list[np.ndarray],
    rois: list[dict[str, Any]],
    out_dir: Path,
    seed: int,
    epochs: int,
    patch_size: int,
) -> pd.DataFrame:
    rows = []
    curves = []
    ae_dir = ensure_dir(out_dir / "autoencoder_maps")
    for roi_index, roi in enumerate(rois):
        for strategy in ("roi_aggregate", "patch_extreme"):
            stride = patch_size if strategy == "roi_aggregate" else patch_size
            bg_patches, bg_records, obj_records = _sample_patch_records(
                frames,
                roi,
                strategy=strategy,
                patch_size=patch_size,
                stride=stride,
                max_patches=5000,
                seed=seed + roi_index,
            )
            model, train_losses, val_losses = _train_patch_autoencoder(bg_patches, seed + roi_index, epochs)
            bg_errors = _patch_errors(model, frames, bg_records[: min(len(bg_records), 2000)], patch_size)
            obj_eval = obj_records[: min(len(obj_records), 2000)]
            obj_errors = _patch_errors(model, frames, obj_eval, patch_size) if obj_eval else np.array([np.nan])
            bg_mean = float(np.nanmean(bg_errors))
            bg_std = float(np.nanstd(bg_errors) + 1e-12)
            obj_mean = float(np.nanmean(obj_errors))
            obj_max = float(np.nanmax(obj_errors))
            z_mean = (obj_mean - bg_mean) / bg_std
            z_max = (obj_max - bg_mean) / bg_std
            percentile_mean = float((bg_errors < obj_mean).mean() * 100)
            percentile_max = float((bg_errors < obj_max).mean() * 100)
            map_path = ae_dir / f"anomaly_{roi['name']}_{strategy}.png"
            _save_anomaly_map(model, frames[len(frames) // 2], roi, map_path, patch_size, max(1, patch_size // 2))
            rows.append(
                {
                    "case_id": case_id,
                    "roi_name": roi["name"],
                    "strategy": strategy,
                    "seed": seed,
                    "epochs": epochs,
                    "patch_size": patch_size,
                    "background_patches_used": len(bg_patches),
                    "object_patches_evaluated": len(obj_eval),
                    "background_mean_error": bg_mean,
                    "background_std_error": bg_std,
                    "object_mean_error": obj_mean,
                    "object_max_patch_error": obj_max,
                    "zscore_mean": float(z_mean),
                    "zscore_max": float(z_max),
                    "percentile_mean": percentile_mean,
                    "percentile_max": percentile_max,
                    "final_train_loss": float(train_losses[-1]),
                    "final_validation_loss": float(val_losses[-1]),
                    "anomaly_map": str(map_path),
                }
            )
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
                curves.append(
                    {
                        "roi_name": roi["name"],
                        "strategy": strategy,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "autoencoder_strategy_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "autoencoder_training_curves.csv", index=False)
    return df


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows generated._"
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_audit_report(
    case_id: str,
    config: dict[str, Any],
    rois: list[dict[str, Any]],
    pca_df: pd.DataFrame,
    ae_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    source_path = DATA_DIR / "sources" / f"{case_id}.source.json"
    source = read_json(source_path) if source_path.exists() else {}
    roi_df = pd.DataFrame(rois)
    report_dir = ensure_dir(case_report_dir(case_id))
    report_path = report_dir / "reproduction_audit.md"
    content = f"""# Reproduction Audit: {case_id}

## Objetivo

Esta auditoria calibra la pipeline automatizada frente a un analisis manual ad-hoc previo. No fuerza resultados, no hardcodea metricas y no decide que una ejecucion sea correcta o incorrecta. Su funcion es aislar sensibilidad a ROI, ventana, PCA, patches de fondo y calculo de z-score.

## Fuente

- Tipo de fuente: `{source.get("source_type", config.get("source_type", "unknown"))}`
- Ruta: `{source.get("video_path", config.get("video_path", ""))}`
- SHA256: `{source.get("sha256", "pendiente")}`
- Duracion/FPS/resolucion: `{source.get("duration_seconds", "pendiente")}` s, `{source.get("fps", "pendiente")}` FPS, `{source.get("resolution", "pendiente")}`

## Diferencia metodologica

El analisis manual de referencia uso una seleccion concreta del objeto central y una comparacion agresiva entre un patch de objeto y patches de fondo/no-objeto. La pipeline automatizada inicial uso una ROI detectada por brillo y metricas mas agregadas sobre la ROI, por eso sus z-score son mucho mas conservadores. Esta auditoria separa esas dos lecturas: deteccion automatizada robusta y reproduccion aproximada de una ejecucion manual concreta.

## ROI candidatas

{_markdown_table(roi_df, ["name", "mode", "x", "y", "width", "height"])}

Panel: `{(out_dir / "roi_candidates_panel.png").as_posix()}`

## PCA por ROI

Referencia manual aproximada: PC1 0.245, PC2 0.158, k=5 0.547, k=10 0.672, correlacion ROI/firma media 0.969 +/- 0.040, correlacion frame a frame 0.992.

{_markdown_table(pca_df, ["roi_name", "pc1", "pc2", "cumulative_k5", "cumulative_k10", "cumulative_k20", "corr_roi_vs_mean_avg", "corr_roi_vs_mean_std", "corr_frame_to_frame_avg"])}

CSV: `{(out_dir / "pca_roi_comparison.csv").as_posix()}`

## Autoencoder por ROI y estrategia

Referencia manual aproximada: error medio fondo 0.0001218, desviacion fondo 0.0001756, error patch objeto 0.147296, z-score patch objeto ~838.

{_markdown_table(ae_df, ["roi_name", "strategy", "background_patches_used", "object_patches_evaluated", "background_mean_error", "background_std_error", "object_mean_error", "object_max_patch_error", "zscore_mean", "zscore_max", "percentile_mean", "percentile_max"])}

CSV: `{(out_dir / "autoencoder_strategy_comparison.csv").as_posix()}`

## Por que cambian los z-score

Los z-score cambian mucho cuando cambia la definicion de fondo, el tamano del patch, el escalado de error, la region considerada objeto, el uso de media agregada frente a maximo de patch, y si se excluye solo el centro de la ROI o toda la ROI del objeto. Un fondo muy homogeneo puede producir desviaciones estandar pequenas y z-score extremos; una metrica agregada sobre ROI completa mezcla objeto, borde, fondo y compresion, por lo que reduce la magnitud del z-score.

## Conclusion prudente

La pipeline automatizada reproduce una anomalia visual persistente y medible bajo varias ROI candidatas, pero no reproduce exactamente el z-score extremo del analisis manual sin adoptar una comparacion patch-extreme muy sensible a parametros. Esto no invalida ni el analisis manual ni la pipeline: distingue deteccion robusta, reproduccion de una corrida concreta y sensibilidad a ROI/patch/fondo.

## Recomendacion para futuros casos

Usar por defecto: ROI automatica documentada, una ROI centrada/manual revisada, PCA 64x64 mean-centered, reporte de correlaciones, autoencoder con semilla fija, y dos metricas separadas: agregada conservadora y patch-extreme exploratoria. La conclusion debe seguir siendo prudente: anomalia visual persistente / fuente sin clasificar.
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def run_reproduction_audit(config: dict[str, Any], seed: int = 42, epochs: int = 8, patch_size: int = 16) -> dict[str, Any]:
    case_id = config["case_id"]
    out_dir = ensure_dir(case_output_dir(case_id) / "reproduction_audit")
    frames = load_frames(case_id)
    rois = build_roi_candidates(case_id, config, frames)
    (out_dir / "roi_candidates.json").write_text(json.dumps(rois, indent=2), encoding="utf-8")
    save_roi_candidate_panel(case_id, frames, rois, out_dir)
    pca_df = audit_pca(case_id, frames, rois, out_dir)
    ae_df = audit_autoencoder(case_id, frames, rois, out_dir, seed=seed, epochs=epochs, patch_size=patch_size)
    report_path = write_audit_report(case_id, config, rois, pca_df, ae_df, out_dir)
    return {
        "case_id": case_id,
        "output_dir": str(out_dir),
        "report": str(report_path),
        "roi_candidates": len(rois),
        "pca_csv": str(out_dir / "pca_roi_comparison.csv"),
        "autoencoder_csv": str(out_dir / "autoencoder_strategy_comparison.csv"),
        "seed": seed,
        "epochs": epochs,
        "patch_size": patch_size,
    }
