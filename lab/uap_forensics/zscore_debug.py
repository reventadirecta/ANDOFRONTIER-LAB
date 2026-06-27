import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .frames import load_frames
from .io import read_json
from .paths import DATA_DIR, case_output_dir, case_report_dir, ensure_dir
from .reproduction_audit import MANUAL_REFERENCE, build_roi_candidates


PATCH_SIZES = [16, 24, 32, 48, 64]
BACKGROUND_STRATEGIES = [
    "dark_strict",
    "corners",
    "outside_roi",
    "outside_central_zone",
    "low_activity_frames",
    "all_frames_excluding_roi",
    "exclude_saturated",
    "brightness_threshold_loose",
]


def _gray_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for frame in frames]


def _clip_patch(x: int, y: int, patch_size: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(x, width - patch_size)), max(0, min(y, height - patch_size))


def _patch(gray: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    return gray[y : y + patch_size, x : x + patch_size]


def _mse_to_template(patches: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.mean((patches - template[None, :, :]) ** 2, axis=(1, 2))


def _activity_scores(grays: list[np.ndarray]) -> np.ndarray:
    scores = [0.0]
    for idx in range(1, len(grays)):
        scores.append(float(np.mean(np.abs(grays[idx] - grays[idx - 1]))))
    return np.array(scores)


def infer_key_frame(grays: list[np.ndarray], roi: dict[str, Any]) -> int:
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    means = [float(gray[y : y + h, x : x + w].mean()) for gray in grays]
    return int(np.argmax(means))


def object_patch_candidates(
    grays: list[np.ndarray],
    rois: list[dict[str, Any]],
    patch_size: int,
    key_frame: int,
) -> list[dict[str, Any]]:
    h, w = grays[0].shape
    candidates: list[dict[str, Any]] = []
    ref_x, ref_y = MANUAL_REFERENCE["object_center_x"], MANUAL_REFERENCE["object_center_y"]
    manual_x, manual_y = _clip_patch(int(ref_x - patch_size / 2), int(ref_y - patch_size / 2), patch_size, w, h)
    candidates.append({"name": "manual_reference_center", "frame": key_frame, "x": manual_x, "y": manual_y})

    for roi in rois:
        rx, ry, rw, rh = roi["x"], roi["y"], roi["width"], roi["height"]
        cx, cy = _clip_patch(rx + rw // 2 - patch_size // 2, ry + rh // 2 - patch_size // 2, patch_size, w, h)
        candidates.append({"name": f"{roi['name']}_center", "frame": key_frame, "x": cx, "y": cy})

        roi_img = grays[key_frame][ry : ry + rh, rx : rx + rw]
        if roi_img.size:
            blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
            _, _, _, loc = cv2.minMaxLoc(blurred)
            bx, by = _clip_patch(rx + loc[0] - patch_size // 2, ry + loc[1] - patch_size // 2, patch_size, w, h)
            candidates.append({"name": f"{roi['name']}_brightest", "frame": key_frame, "x": bx, "y": by})

            mask = (roi_img > max(float(np.percentile(roi_img, 95)), float(roi_img.mean() + roi_img.std()))).astype(np.uint8)
            moments = cv2.moments(mask)
            if moments["m00"]:
                mx = int(rx + moments["m10"] / moments["m00"])
                my = int(ry + moments["m01"] / moments["m00"])
                mx, my = _clip_patch(mx - patch_size // 2, my - patch_size // 2, patch_size, w, h)
                candidates.append({"name": f"{roi['name']}_mass_centroid", "frame": key_frame, "x": mx, "y": my})

    # A coarse multi-frame brightest candidate approximates a possible ad-hoc frame pick.
    sample_indices = sorted(set([0, len(grays) // 4, len(grays) // 2, 3 * len(grays) // 4, len(grays) - 1, key_frame]))
    best = None
    for frame_idx in sample_indices:
        gray = grays[frame_idx]
        central = gray[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
        _, max_val, _, loc = cv2.minMaxLoc(central)
        if best is None or max_val > best[0]:
            best = (max_val, frame_idx, int(w * 0.25) + loc[0], int(h * 0.25) + loc[1])
    if best:
        _, frame_idx, bx, by = best
        bx, by = _clip_patch(bx - patch_size // 2, by - patch_size // 2, patch_size, w, h)
        candidates.append({"name": "sampled_frames_brightest", "frame": frame_idx, "x": bx, "y": by})

    unique = []
    seen = set()
    for candidate in candidates:
        key = (candidate["name"], candidate["frame"], candidate["x"], candidate["y"])
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _overlaps(rect: tuple[int, int, int, int], roi: dict[str, Any]) -> bool:
    x, y, w, h = rect
    return x < roi["x"] + roi["width"] and x + w > roi["x"] and y < roi["y"] + roi["height"] and y + h > roi["y"]


def background_records(
    grays: list[np.ndarray],
    roi: dict[str, Any],
    patch_size: int,
    strategy: str,
    max_patches: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    h, w = grays[0].shape
    stride = patch_size
    activity = _activity_scores(grays)
    low_activity_frames = set(np.argsort(activity)[: max(10, len(grays) // 10)].tolist())
    records = []
    for frame_idx, gray in enumerate(grays):
        if strategy == "low_activity_frames" and frame_idx not in low_activity_frames:
            continue
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = gray[y : y + patch_size, x : x + patch_size]
                mean = float(patch.mean())
                maxv = float(patch.max())
                rect = (x, y, patch_size, patch_size)
                in_corner = (x < w * 0.18 or x > w * 0.82 - patch_size) and (
                    y < h * 0.18 or y > h * 0.82 - patch_size
                )
                in_central = x < w * 0.70 and x + patch_size > w * 0.30 and y < h * 0.70 and y + patch_size > h * 0.30
                outside_roi = not _overlaps(rect, roi)
                keep = False
                if strategy == "dark_strict":
                    keep = mean < 0.055 and maxv < 0.20
                elif strategy == "corners":
                    keep = in_corner
                elif strategy == "outside_roi":
                    keep = outside_roi
                elif strategy == "outside_central_zone":
                    keep = not in_central
                elif strategy == "low_activity_frames":
                    keep = outside_roi
                elif strategy == "all_frames_excluding_roi":
                    keep = outside_roi
                elif strategy == "exclude_saturated":
                    keep = outside_roi and maxv < 0.92
                elif strategy == "brightness_threshold_loose":
                    keep = outside_roi and mean < 0.25
                if keep:
                    records.append((frame_idx, x, y))
    if len(records) > max_patches:
        idx = rng.choice(len(records), size=max_patches, replace=False)
        records = [records[int(i)] for i in idx]
    return records


def records_to_patches(grays: list[np.ndarray], records: list[tuple[int, int, int]], patch_size: int) -> np.ndarray:
    return np.stack([_patch(grays[frame_idx], x, y, patch_size) for frame_idx, x, y in records]).astype(np.float32)


def classify_metric(zscore: float, bg_std: float, bg_count: int, percentile: float, strategy: str) -> str:
    if bg_count < 30 or not np.isfinite(zscore):
        return "discard"
    if bg_std < 1e-5:
        return "unstable"
    if zscore > 100 or (zscore > 50 and strategy in {"dark_strict", "brightness_threshold_loose"}):
        return "unstable"
    if zscore > 20 or percentile >= 99.5:
        return "exploratory"
    return "public_safe"


def run_zscore_debug(config: dict[str, Any], seed: int = 42, max_background_patches: int = 4000) -> dict[str, Any]:
    case_id = config["case_id"]
    out_dir = ensure_dir(case_output_dir(case_id) / "debug_zscore")
    frames = load_frames(case_id)
    grays = _gray_frames(frames)
    rois = build_roi_candidates(case_id, config, frames)
    roi_by_name = {roi["name"]: roi for roi in rois}
    key_roi = roi_by_name.get("manual_center_128", rois[0])
    key_frame = infer_key_frame(grays, key_roi)
    rows = []
    best_payload: dict[str, Any] | None = None

    for roi_idx, roi in enumerate(rois):
        for patch_size in PATCH_SIZES:
            object_candidates = object_patch_candidates(grays, [roi], patch_size, key_frame)
            for strategy in BACKGROUND_STRATEGIES:
                bg_records = background_records(
                    grays,
                    roi,
                    patch_size,
                    strategy,
                    max_patches=max_background_patches,
                    seed=seed + roi_idx + patch_size,
                )
                if len(bg_records) < 5:
                    continue
                bg_patches = records_to_patches(grays, bg_records, patch_size)
                template = bg_patches.mean(axis=0)
                bg_errors = _mse_to_template(bg_patches, template)
                bg_mean = float(bg_errors.mean())
                bg_std = float(bg_errors.std(ddof=0))
                bg_min = float(bg_errors.min())
                bg_max = float(bg_errors.max())
                for candidate in object_candidates:
                    obj_patch = _patch(grays[candidate["frame"]], candidate["x"], candidate["y"], patch_size)
                    object_error = float(np.mean((obj_patch - template) ** 2))
                    zscore = (object_error - bg_mean) / (bg_std + 1e-12)
                    percentile = float((bg_errors < object_error).mean() * 100)
                    ratio = float(object_error / (bg_mean + 1e-12))
                    std_warning = bool(bg_std < 1e-5 or (abs(zscore) > 100 and bg_std < 0.001))
                    classification = classify_metric(zscore, bg_std, len(bg_errors), percentile, strategy)
                    row = {
                        "case_id": case_id,
                        "error_model": "background_mean_template",
                        "roi_name": roi["name"],
                        "roi_x": roi["x"],
                        "roi_y": roi["y"],
                        "roi_width": roi["width"],
                        "roi_height": roi["height"],
                        "patch_size": patch_size,
                        "object_patch_variant": candidate["name"],
                        "object_frame": candidate["frame"],
                        "object_x": candidate["x"],
                        "object_y": candidate["y"],
                        "background_strategy": strategy,
                        "background_patch_count": len(bg_errors),
                        "background_mean_error": bg_mean,
                        "background_std_error": bg_std,
                        "background_min_error": bg_min,
                        "background_max_error": bg_max,
                        "object_error": object_error,
                        "zscore": float(zscore),
                        "percentile": percentile,
                        "object_error_to_background_mean_ratio": ratio,
                        "std_too_small_warning": std_warning,
                        "metric_classification": classification,
                    }
                    rows.append(row)
                    if best_payload is None or zscore > best_payload["row"]["zscore"]:
                        best_payload = {
                            "row": row,
                            "bg_errors": bg_errors,
                            "bg_records": bg_records[:8],
                            "template": template,
                            "object_patch": obj_patch,
                        }

    df = pd.DataFrame(rows).sort_values("zscore", ascending=False)
    csv_path = out_dir / "zscore_debug_grid.csv"
    df.to_csv(csv_path, index=False)
    if best_payload:
        save_debug_panel(case_id, frames, grays, rois, df, best_payload, out_dir)
    report_path = write_debug_report(case_id, config, df, out_dir)
    return {
        "case_id": case_id,
        "output_dir": str(out_dir),
        "csv": str(csv_path),
        "panel": str(out_dir / "zscore_debug_panel.png"),
        "report": str(report_path),
        "rows": int(len(df)),
        "top_zscore": float(df.iloc[0]["zscore"]) if not df.empty else None,
        "top_classification": str(df.iloc[0]["metric_classification"]) if not df.empty else None,
    }


def save_debug_panel(
    case_id: str,
    frames: list[np.ndarray],
    grays: list[np.ndarray],
    rois: list[dict[str, Any]],
    df: pd.DataFrame,
    best_payload: dict[str, Any],
    out_dir: Path,
) -> None:
    row = best_payload["row"]
    frame = frames[int(row["object_frame"])].copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[0:2, 0:2])
    ax0.imshow(rgb)
    for roi in rois:
        ax0.add_patch(plt.Rectangle((roi["x"], roi["y"]), roi["width"], roi["height"], fill=False, edgecolor="cyan", linewidth=1.5))
    ax0.add_patch(
        plt.Rectangle(
            (row["object_x"], row["object_y"]),
            row["patch_size"],
            row["patch_size"],
            fill=False,
            edgecolor="red",
            linewidth=2.5,
        )
    )
    for frame_idx, x, y in best_payload["bg_records"]:
        if frame_idx == row["object_frame"]:
            ax0.add_patch(plt.Rectangle((x, y), row["patch_size"], row["patch_size"], fill=False, edgecolor="lime", linewidth=1))
    ax0.set_title("Frame clave con ROI, patch objeto y fondos del mismo frame")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(best_payload["object_patch"], cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Patch objeto extremo")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 2])
    bg_examples = []
    for frame_idx, x, y in best_payload["bg_records"][:6]:
        bg_examples.append(_patch(grays[frame_idx], x, y, int(row["patch_size"])))
    if bg_examples:
        bg_grid = np.concatenate(bg_examples, axis=1)
        ax2.imshow(bg_grid, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Ejemplos de patches de fondo")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(best_payload["bg_errors"], bins=60, color="steelblue", alpha=0.85)
    ax3.axvline(row["object_error"], color="red", linewidth=2, label="object error")
    ax3.set_title("Histograma de errores de fondo")
    ax3.set_xlabel("MSE vs template fondo")
    ax3.legend()

    ax4 = fig.add_subplot(gs[2, 1:])
    ax4.axis("off")
    top = df.head(10)[
        ["roi_name", "patch_size", "object_patch_variant", "background_strategy", "zscore", "metric_classification"]
    ].copy()
    top["zscore"] = top["zscore"].map(lambda x: f"{x:.2f}")
    table = ax4.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    warning = "WARNING: z-score may be dominated by tiny background std." if bool(row["std_too_small_warning"]) else ""
    ax4.set_title(f"Top 10 por z-score. {warning}")

    fig.suptitle(f"{case_id} z-score debug: top z={row['zscore']:.2f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "zscore_debug_panel.png", dpi=160)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: list[str], rows: int = 12) -> str:
    view = df.head(rows)[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.6g}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_debug_report(case_id: str, config: dict[str, Any], df: pd.DataFrame, out_dir: Path) -> Path:
    source_path = DATA_DIR / "sources" / f"{case_id}.source.json"
    source = read_json(source_path) if source_path.exists() else {}
    report_path = ensure_dir(case_report_dir(case_id)) / "zscore_debug_report.md"
    top = df.iloc[0] if not df.empty else {}
    closest = df.iloc[(df["zscore"] - MANUAL_REFERENCE["ae_object_zscore"]).abs().argsort()[:1]].iloc[0] if not df.empty else {}
    public_safe = df[df["metric_classification"] == "public_safe"].head(5)
    exploratory = df[df["metric_classification"] == "exploratory"].head(5)
    unstable = df[df["metric_classification"] == "unstable"].head(5)
    content = f"""# Z-score Debug Report: {case_id}

## Objetivo

Esta depuracion investiga por que el analisis manual ad-hoc reporto un z-score aproximado de `~838`, mientras que la auditoria reproducible obtuvo `60.48` en `patch_extreme` y `27.19` en `roi_aggregate`. No se ajustan parametros para alcanzar el valor manual; se barre una grilla de ROI, patches y fondos para explicar sensibilidad.

## Fuente

- Tipo: `{source.get("source_type", config.get("source_type", "unknown"))}`
- SHA256: `{source.get("sha256", "pendiente")}`
- Frames/resolucion: `584`, `{source.get("resolution", {"width": 640, "height": 360})}`

## Modelo de error usado en esta depuracion

El CSV usa `background_mean_template`: cada estrategia de fondo crea un template medio de patches de fondo y mide MSE de patches de fondo y objeto contra ese template. Esto no sustituye al autoencoder auditado; sirve para aislar la aritmetica que puede inflar un z-score cuando el fondo tiene varianza muy pequena.

## Top configuraciones por z-score

{_markdown_table(df, ["roi_name", "patch_size", "object_patch_variant", "background_strategy", "background_patch_count", "background_mean_error", "background_std_error", "object_error", "zscore", "percentile", "metric_classification"], 12)}

## Configuracion mas cercana a `~838`

| campo | valor |
| --- | --- |
| roi | `{closest.get("roi_name", "")}` |
| patch_size | `{closest.get("patch_size", "")}` |
| object_patch_variant | `{closest.get("object_patch_variant", "")}` |
| background_strategy | `{closest.get("background_strategy", "")}` |
| zscore | `{closest.get("zscore", "")}` |
| background_mean_error | `{closest.get("background_mean_error", "")}` |
| background_std_error | `{closest.get("background_std_error", "")}` |
| object_error | `{closest.get("object_error", "")}` |
| classification | `{closest.get("metric_classification", "")}` |

## Por que puede aparecer `~838`

El z-score se calcula como `(object_error - background_mean) / background_std`. Con la referencia manual aproximada, `(0.147296 - 0.0001218) / 0.0001756 = 838.13`. Es decir, el valor extremo no requiere que el error del objeto cambie mucho: basta con que el fondo seleccionado sea muy homogeneo y produzca una desviacion estandar muy pequena. Este tipo de fondo puede existir si se usan patches oscuros estrictos, esquinas muy uniformes o filtros que excluyen casi toda variabilidad.

## Por que la auditoria bajo a `60.48`

La auditoria reproducible entreno y evaluo con fondos mas amplios y patches muestreados de forma sistematica. Eso aumenta la desviacion estandar del fondo y reduce el z-score. La senal sigue siendo anomala por percentil y error relativo, pero el z-score extremo se diluye al incluir compresion, gradientes, bordes, ruido y variacion normal del video.

## Clasificacion de uso

### public_safe

{_markdown_table(public_safe, ["roi_name", "patch_size", "background_strategy", "zscore", "percentile", "metric_classification"], 5) if not public_safe.empty else "_No hay metricas public_safe en el top ordenado por z-score._"}

### exploratory

{_markdown_table(exploratory, ["roi_name", "patch_size", "background_strategy", "zscore", "percentile", "metric_classification"], 5) if not exploratory.empty else "_No hay metricas exploratory destacadas._"}

### unstable

{_markdown_table(unstable, ["roi_name", "patch_size", "background_strategy", "zscore", "percentile", "metric_classification"], 5) if not unstable.empty else "_No hay metricas unstable destacadas._"}

## Recomendacion final

Para paper/reporte/Reddit/GitHub usar metricas `public_safe` y, como mucho, mencionar `exploratory` con cautela. No publicar el z-score `~838` como cifra principal salvo que se documente exactamente patch, fondo, escala, template/modelo de error y razon de la desviacion de fondo. El valor extremo debe quedar como analisis exploratorio o inestable, no como claim central.

## Artefactos

- CSV completo: `{(out_dir / "zscore_debug_grid.csv").as_posix()}`
- Panel visual: `{(out_dir / "zscore_debug_panel.png").as_posix()}`
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path
