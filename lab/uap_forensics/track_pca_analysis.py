from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


REQUIRED_MESSAGE = "PCA analysis requires a human-validated track, dynamic ROIs, and clean controls baseline."
CONTROL_CLASSES = ["near_background", "far_background", "compression_noise", "random_background", "hud_artifact"]
BACKGROUND_CLASSES = ["near_background", "far_background", "random_background"]
INPUT_SIZE = 64
MAX_OBJECT_SAMPLES = 320
MAX_CLASS_SAMPLES = 160


def track_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"


def validation_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "track_validation.json"


def dynamic_rois_csv(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv"


def object_crops_dir(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "track_based_analysis" / "crops"


def controls_dir(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "controls_analysis"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "pca_analysis")


def case_status_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "case_status.json"


def _log(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        return


def _require_inputs(case_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checks = {
        "track.json": track_path(case_id),
        "track_validation.json": validation_path(case_id),
        "dynamic_rois.csv": dynamic_rois_csv(case_id),
        "object crops": object_crops_dir(case_id),
        "controls_metrics.json": controls_dir(case_id) / "controls_metrics.json",
        "controls_timeseries.csv": controls_dir(case_id) / "controls_timeseries.csv",
    }
    for label, path in checks.items():
        found = path.exists()
        _log(f"PCA input {label}: {'found' if found else 'missing'} - {path}")
        if not found:
            raise RuntimeError(f"{REQUIRED_MESSAGE} Missing {label}: {path}")
    manifest = controls_dir(case_id) / "clean_controls_manifest.md"
    if manifest.exists():
        _log(f"PCA input clean_controls_manifest.md: found - {manifest}")
    else:
        _log(f"PCA input clean_controls_manifest.md: missing optional manifest - {manifest}")
    validation = read_json(validation_path(case_id))
    if not (validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target")):
        raise RuntimeError(f"{REQUIRED_MESSAGE} Track validation is not human-confirmed.")
    controls_metrics = read_json(controls_dir(case_id) / "controls_metrics.json")
    if controls_metrics.get("controls_version") != "Controls v0.2 clean masked":
        raise RuntimeError(f"{REQUIRED_MESSAGE} Expected Controls v0.2 clean masked, got {controls_metrics.get('controls_version')!r}.")
    if float(controls_metrics.get("control_validity_score") or 0) < 0.45:
        raise RuntimeError(f"{REQUIRED_MESSAGE} Control validity score is below 0.45.")
    return read_json(track_path(case_id)), validation, controls_metrics


def _frame_from_name(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)|crop_(\d+)", path.stem)
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def _paths_sampled(paths: list[Path], limit: int) -> list[Path]:
    paths = sorted(paths)
    if len(paths) <= limit:
        return paths
    idx = np.linspace(0, len(paths) - 1, limit).round().astype(int)
    return [paths[int(i)] for i in idx]


def _features(path: Path, label: str, control_status: str = "clean") -> dict[str, Any] | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(fft))
    yy, xx = np.indices(mag.shape)
    center = np.array([(mag.shape[0] - 1) / 2, (mag.shape[1] - 1) / 2])
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    high = radius > (0.35 * radius.max())
    return {
        "source_path": str(path),
        "frame": _frame_from_name(path),
        "class_label": label,
        "vector": arr.reshape(-1),
        "image": resized,
        "luminance_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "high_frequency_ratio": float(np.sum(mag[high]) / (np.sum(mag) + 1e-9)),
        "track_status": "TRACKING_ACTIVE" if label == "object" else "",
        "control_status": control_status if label != "object" else "",
    }


def _load_samples(case_id: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    object_paths = _paths_sampled(list(object_crops_dir(case_id).glob("*.png")), MAX_OBJECT_SAMPLES)
    for path in object_paths:
        item = _features(path, "object", "")
        if item:
            samples.append(item)
    root = controls_dir(case_id) / "controls"
    for class_label in CONTROL_CLASSES:
        folder = root / class_label
        if not folder.exists():
            continue
        paths = _paths_sampled(list(folder.glob("frame_*.png")), MAX_CLASS_SAMPLES)
        for path in paths:
            status = "artifact_isolated" if class_label == "hud_artifact" else "clean"
            item = _features(path, class_label, status)
            if item:
                samples.append(item)
    return samples


def _centroid(scores: np.ndarray, labels: list[str], wanted: list[str]) -> np.ndarray | None:
    idx = [i for i, label in enumerate(labels) if label in wanted]
    if not idx:
        return None
    return scores[idx, :2].mean(axis=0)


def _dist(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def _run_pca(samples: list[dict[str, Any]]) -> tuple[PCA, np.ndarray, np.ndarray]:
    x = np.vstack([sample["vector"] for sample in samples])
    x_scaled = StandardScaler(with_std=True).fit_transform(x)
    n = min(20, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(x_scaled)
    return pca, scores, x_scaled


def _metrics(case_id: str, samples: list[dict[str, Any]], pca: PCA, scores: np.ndarray, controls_metrics: dict[str, Any]) -> dict[str, Any]:
    labels = [sample["class_label"] for sample in samples]
    counts = {label: labels.count(label) for label in ["object", *CONTROL_CLASSES]}
    object_c = _centroid(scores, labels, ["object"])
    bg_c = _centroid(scores, labels, BACKGROUND_CLASSES)
    comp_c = _centroid(scores, labels, ["compression_noise"])
    hud_c = _centroid(scores, labels, ["hud_artifact"])
    distances = {
        "background": _dist(object_c, bg_c),
        "compression": _dist(object_c, comp_c),
        "hud": _dist(object_c, hud_c) if counts.get("hud_artifact", 0) else 0.0,
    }
    all_centroid_distances = [value for value in distances.values() if value > 0]
    score_scale = float(np.std(scores[:, :2]) + 1e-9)
    separation = float(np.mean(all_centroid_distances) / score_scale) if all_centroid_distances else 0.0
    try:
        sil = float(silhouette_score(scores[:, : min(5, scores.shape[1])], labels)) if len(set(labels)) > 1 and len(samples) > len(set(labels)) else 0.0
    except Exception:
        sil = 0.0
    nearest_similarity = float(1.0 / (1.0 + min(all_centroid_distances) / score_scale)) if all_centroid_distances else 0.0
    public_safe = float(max(0.0, min(1.0, controls_metrics.get("control_validity_score", 0) * (0.5 + min(0.5, max(0.0, sil + 0.15))))))
    ev = pca.explained_variance_ratio_
    return {
        "case_id": case_id,
        "total_object_samples": counts.get("object", 0),
        "total_control_samples": len(samples) - counts.get("object", 0),
        "samples_per_class": counts,
        "input_crop_size": [INPUT_SIZE, INPUT_SIZE],
        "preprocessing_used": "grayscale resize 64x64, flatten, StandardScaler(with_std=True), PCA randomized random_state=42",
        "pca_pc1_explained_variance": float(ev[0]) if len(ev) else 0.0,
        "pca_pc2_explained_variance": float(ev[1]) if len(ev) > 1 else 0.0,
        "pca_k5_explained_variance": float(np.sum(ev[: min(5, len(ev))])),
        "pca_k10_explained_variance": float(np.sum(ev[: min(10, len(ev))])),
        "object_centroid_pc1": float(object_c[0]) if object_c is not None else 0.0,
        "object_centroid_pc2": float(object_c[1]) if object_c is not None else 0.0,
        "background_centroid_pc1": float(bg_c[0]) if bg_c is not None else 0.0,
        "background_centroid_pc2": float(bg_c[1]) if bg_c is not None else 0.0,
        "object_vs_background_distance": distances["background"],
        "object_vs_compression_distance": distances["compression"],
        "object_vs_hud_distance": distances["hud"],
        "class_separation_score": separation,
        "silhouette_score": sil,
        "nearest_control_similarity_score": nearest_similarity,
        "pca_public_safe_score": public_safe,
        "controls_version": controls_metrics.get("controls_version"),
        "control_validity_score": controls_metrics.get("control_validity_score"),
        "notes": [
            "PCA compares human-validated object crops against Controls v0.2 clean masked baseline crops.",
            "Separation from controls indicates visual/statistical difference, not non-human origin.",
            "PCA is dimensionality reduction; it does not determine origin.",
        ],
    }


def _write_samples(path: Path, samples: list[dict[str, Any]], scores: np.ndarray) -> None:
    fields = ["sample_id", "frame", "class_label", "source_path", "pc1", "pc2", "pc3", "pc4", "pc5", "luminance_mean", "contrast_std", "high_frequency_ratio", "track_status", "control_status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            row = {
                "sample_id": idx,
                "frame": sample["frame"],
                "class_label": sample["class_label"],
                "source_path": sample["source_path"],
                "luminance_mean": sample["luminance_mean"],
                "contrast_std": sample["contrast_std"],
                "high_frequency_ratio": sample["high_frequency_ratio"],
                "track_status": sample["track_status"],
                "control_status": sample["control_status"],
            }
            for pc in range(5):
                row[f"pc{pc + 1}"] = float(scores[idx, pc]) if scores.shape[1] > pc else 0.0
            writer.writerow(row)


def _write_variance(path: Path, pca: PCA) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "explained_variance_ratio", "cumulative_explained_variance"])
        writer.writeheader()
        for idx, ratio in enumerate(pca.explained_variance_ratio_, start=1):
            writer.writerow({"component": idx, "explained_variance_ratio": float(ratio), "cumulative_explained_variance": float(cumulative[idx - 1])})


def _scatter_panel(path: Path, samples: list[dict[str, Any]], scores: np.ndarray) -> None:
    labels = [sample["class_label"] for sample in samples]
    colors = {
        "object": "#e15759",
        "near_background": "#4e79a7",
        "far_background": "#59a14f",
        "compression_noise": "#edc948",
        "random_background": "#76b7b2",
        "hud_artifact": "#af7aa1",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in ["object", *CONTROL_CLASSES]:
        idx = np.array([i for i, item in enumerate(labels) if item == label], dtype=int)
        if len(idx):
            ax.scatter(scores[idx, 0], scores[idx, 1], s=14, alpha=0.72, label=label, color=colors.get(label))
            c = scores[idx, :2].mean(axis=0)
            ax.scatter([c[0]], [c[1]], s=150, marker="x", color=colors.get(label), linewidths=3)
    ax.set_title("PCA baseline: object vs clean controls")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _variance_panel(path: Path, pca: PCA) -> None:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(1, len(ratios) + 1), ratios, label="component")
    ax.plot(np.arange(1, len(ratios) + 1), cumulative, marker="o", color="#e15759", label="cumulative")
    ax.axvline(5, color="#59a14f", linestyle="--", label=f"k=5 {cumulative[min(4, len(cumulative)-1)]:.3f}")
    ax.axvline(10, color="#af7aa1", linestyle="--", label=f"k=10 {cumulative[min(9, len(cumulative)-1)]:.3f}")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _distance_panel(path: Path, metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        ["background", "compression", "HUD"],
        [metrics["object_vs_background_distance"], metrics["object_vs_compression_distance"], metrics["object_vs_hud_distance"]],
        color=["#4e79a7", "#edc948", "#af7aa1"],
    )
    axes[0].set_title("Object centroid distance vs controls")
    axes[1].bar(
        ["separation", "silhouette", "nearest similarity", "public-safe"],
        [metrics["class_separation_score"], metrics["silhouette_score"], metrics["nearest_control_similarity_score"], metrics["pca_public_safe_score"]],
        color=["#e15759", "#59a14f", "#76b7b2", "#f28e2b"],
    )
    axes[1].set_title("Interpretive scores")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _contact_sheet(path: Path, samples: list[dict[str, Any]]) -> None:
    tiles: list[np.ndarray] = []
    for label in ["object", *CONTROL_CLASSES]:
        items = [sample for sample in samples if sample["class_label"] == label][:4]
        for sample in items:
            img = cv2.imread(sample["source_path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            tile = cv2.resize(img, (180, 140), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
    if not tiles:
        return
    cols = 4
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + h, c * w : c * w + w] = tile
    cv2.imwrite(str(path), sheet)


def _reconstruction_panel(path: Path, samples: list[dict[str, Any]], pca: PCA, scores: np.ndarray, x_scaled: np.ndarray) -> None:
    # Lightweight PCA reconstruction visualization only; not an autoencoder.
    vectors = np.vstack([sample["vector"] for sample in samples])
    mean = vectors.mean(axis=0)
    selected = []
    for label in ["object", "near_background", "compression_noise", "hud_artifact"]:
        idx = next((i for i, sample in enumerate(samples) if sample["class_label"] == label), None)
        if idx is not None:
            selected.append((label, idx))
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, max(3, len(selected) * 3)))
    if len(selected) == 1:
        axes = np.array([axes])
    for row, (label, idx) in enumerate(selected):
        original = samples[idx]["image"].astype(np.float32) / 255.0
        for col, k in enumerate([0, 5, 10]):
            ax = axes[row, col]
            if k == 0:
                img = original
                title = f"{label} original"
            else:
                k_eff = min(k, scores.shape[1])
                recon_scaled = np.dot(scores[idx, :k_eff], pca.components_[:k_eff]) + pca.mean_
                # Show relative structure after inverse scaling is unavailable; normalize for visual comparison.
                img = recon_scaled.reshape(INPUT_SIZE, INPUT_SIZE)
                img = (img - img.min()) / (img.max() - img.min() + 1e-9)
                title = f"{label} PCA k={k}"
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(case_id: str, validation: dict[str, Any], controls_metrics: dict[str, Any], metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report_dir = ensure_dir(case_report_dir(case_id))
    report = report_dir / "pca_analysis_report.md"
    lines = [
        f"# PCA Track-Based + Clean Controls Baseline Report: {case_id}",
        "",
        "## Scope",
        "",
        "This module compares the human-validated tracked object against Controls v0.2 clean masked baseline crops.",
        "",
        "PCA is a dimensionality reduction tool; it does not determine origin.",
        "",
        "Separation from controls indicates visual/statistical difference, not non-human origin.",
        "",
        "## Source",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- controls metrics: `{controls_dir(case_id) / 'controls_metrics.json'}`",
        f"- controls version: `{controls_metrics.get('controls_version')}`",
        f"- control validity score: `{controls_metrics.get('control_validity_score')}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        "",
        "## Samples",
        "",
    ]
    for label, count in metrics["samples_per_class"].items():
        lines.append(f"- {label}: `{count}`")
    lines += [
        "",
        "## PCA Metrics",
        "",
        f"- PC1 explained variance: `{metrics['pca_pc1_explained_variance']:.6f}`",
        f"- PC2 explained variance: `{metrics['pca_pc2_explained_variance']:.6f}`",
        f"- k=5 explained variance: `{metrics['pca_k5_explained_variance']:.6f}`",
        f"- k=10 explained variance: `{metrics['pca_k10_explained_variance']:.6f}`",
        f"- object vs background distance: `{metrics['object_vs_background_distance']:.6f}`",
        f"- object vs compression distance: `{metrics['object_vs_compression_distance']:.6f}`",
        f"- object vs HUD distance: `{metrics['object_vs_hud_distance']:.6f}`",
        f"- class separation score: `{metrics['class_separation_score']:.6f}`",
        f"- silhouette score: `{metrics['silhouette_score']:.6f}`",
        f"- nearest control similarity score: `{metrics['nearest_control_similarity_score']:.6f}`",
        f"- PCA public-safe score: `{metrics['pca_public_safe_score']:.6f}`",
        "",
        "## Interpretation",
        "",
    ]
    if metrics["class_separation_score"] > 1.0 and metrics["silhouette_score"] > 0:
        lines.append("The object occupies a distinguishable PCA region relative to at least some clean controls, within this crop/preprocessing setup.")
    else:
        lines.append("The object is not strongly separated from all controls in this PCA setup; interpret with caution.")
    lines += [
        "",
        "## Limitations",
        "",
        "- PCA depends on crop size, grayscale preprocessing, scaling and sample balance.",
        "- Controls are heuristic clean masked controls and still require visual review.",
        "- PCA does not identify material, distance, speed, origin or intent.",
        "- No ROI automation, object redetection, autoencoder or generative model was used.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "pca_manifest.md"
    classes = {
        "pca_metrics.json": "technical",
        "pca_samples.csv": "technical",
        "pca_explained_variance.csv": "technical",
        "pca_scatter_panel.png": "public_safe",
        "pca_explained_variance_panel.png": "technical",
        "pca_class_distance_panel.png": "interpretive",
        "pca_reconstruction_panel.png": "technical",
        "pca_contact_sheet_panel.png": "public_safe",
        "pca_analysis_report.md": "interpretive",
    }
    lines = ["# PCA Analysis Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in outputs.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | PCA separation is statistical/visual only; no origin claim |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, status: dict[str, Any]) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(status)
    write_json(path, existing)
    return path


def run_track_pca_analysis(case_id: str) -> dict[str, Any]:
    _log(f"Starting PCA analysis for case {case_id}")
    _track, validation, controls_metrics = _require_inputs(case_id)
    _log("Loading PCA samples from object crops and clean controls...")
    samples = _load_samples(case_id)
    if len(samples) < 12 or not any(sample["class_label"] == "object" for sample in samples):
        raise RuntimeError(f"{REQUIRED_MESSAGE} Not enough PCA samples after loading crops.")
    _log(f"Running PCA backend with {len(samples)} samples...")
    pca, scores, x_scaled = _run_pca(samples)
    metrics = _metrics(case_id, samples, pca, scores, controls_metrics)
    out = output_dir(case_id)
    _log(f"Writing PCA outputs to {out}")
    outputs: dict[str, Path] = {
        "pca_metrics.json": out / "pca_metrics.json",
        "pca_samples.csv": out / "pca_samples.csv",
        "pca_explained_variance.csv": out / "pca_explained_variance.csv",
        "pca_scatter_panel.png": out / "pca_scatter_panel.png",
        "pca_explained_variance_panel.png": out / "pca_explained_variance_panel.png",
        "pca_class_distance_panel.png": out / "pca_class_distance_panel.png",
        "pca_reconstruction_panel.png": out / "pca_reconstruction_panel.png",
        "pca_contact_sheet_panel.png": out / "pca_contact_sheet_panel.png",
    }
    _log("Writing pca_metrics.json")
    write_json(outputs["pca_metrics.json"], metrics)
    _log("Writing pca_samples.csv")
    _write_samples(outputs["pca_samples.csv"], samples, scores)
    _log("Writing pca_explained_variance.csv")
    _write_variance(outputs["pca_explained_variance.csv"], pca)
    _log("Writing PCA plots")
    _scatter_panel(outputs["pca_scatter_panel.png"], samples, scores)
    _variance_panel(outputs["pca_explained_variance_panel.png"], pca)
    _distance_panel(outputs["pca_class_distance_panel.png"], metrics)
    _reconstruction_panel(outputs["pca_reconstruction_panel.png"], samples, pca, scores, x_scaled)
    _contact_sheet(outputs["pca_contact_sheet_panel.png"], samples)
    _log("Writing PCA report")
    report = _write_report(case_id, validation, controls_metrics, metrics, outputs)
    outputs["pca_analysis_report.md"] = report
    manifest = _write_manifest(case_id, outputs)
    outputs["pca_manifest.md"] = manifest
    status_path = _update_case_status(
        case_id,
        {
            "pca_analysis_status": "complete",
            "pca_analysis_ready": True,
            "last_pca_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pca_class_separation_score": metrics["class_separation_score"],
            "pca_public_safe_score": metrics["pca_public_safe_score"],
            "pca_analysis_paths": {
                "output_dir": str(out),
                "metrics": str(outputs["pca_metrics.json"]),
                "samples": str(outputs["pca_samples.csv"]),
                "explained_variance": str(outputs["pca_explained_variance.csv"]),
                "report": str(report),
                "manifest": str(manifest),
                "panels": {
                    "scatter": str(outputs["pca_scatter_panel.png"]),
                    "explained_variance": str(outputs["pca_explained_variance_panel.png"]),
                    "class_distance": str(outputs["pca_class_distance_panel.png"]),
                    "reconstruction": str(outputs["pca_reconstruction_panel.png"]),
                    "contact_sheet": str(outputs["pca_contact_sheet_panel.png"]),
                },
            },
        },
    )
    _log("PCA analysis complete")
    return {
        "case_id": case_id,
        "pca_analysis_ready": True,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "case_status": str(status_path),
    }
