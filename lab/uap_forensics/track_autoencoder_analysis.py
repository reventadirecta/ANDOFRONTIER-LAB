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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except Exception as exc:  # pragma: no cover - reported at runtime with a clear message
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    random_split = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


REQUIRED_MESSAGE = "Autoencoder analysis requires a human-validated track, dynamic ROIs, and clean controls baseline."
CONTROLS_VERSION = "Controls v0.2 clean masked"
TRAIN_CONTROL_CLASSES = ["near_background", "far_background", "compression_noise", "random_background"]
EVAL_CONTROL_CLASSES = [*TRAIN_CONTROL_CLASSES, "hud_artifact"]
INPUT_SIZE = 64
SEED = 42


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


def pca_metrics_path(case_id: str) -> Path:
    return DATA_DIR / "outputs" / case_id / "pca_analysis" / "pca_metrics.json"


def output_dir(case_id: str) -> Path:
    return ensure_dir(DATA_DIR / "outputs" / case_id / "autoencoder_analysis")


def case_status_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "case_status.json"


def _log(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        return


class SmallConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _require_inputs(case_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for autoencoder analysis: {TORCH_IMPORT_ERROR}")
    checks = {
        "track.json": (track_path(case_id), "Autoencoder requires a human-validated track. Run tracking first."),
        "track_validation.json": (validation_path(case_id), "Autoencoder requires a human-validated track. Run tracking first."),
        "dynamic_rois.csv": (dynamic_rois_csv(case_id), "Autoencoder requires dynamic ROI output. Run Track-Based ROI first."),
        "object crops": (object_crops_dir(case_id), "Autoencoder requires dynamic ROI output. Run Track-Based ROI first."),
        "controls_metrics.json": (controls_dir(case_id) / "controls_metrics.json", "Autoencoder requires controls baseline output. Run Clean Controls first."),
        "controls_timeseries.csv": (controls_dir(case_id) / "controls_timeseries.csv", "Autoencoder requires controls baseline output. Run Clean Controls first."),
        "pca_metrics.json": (pca_metrics_path(case_id), "Autoencoder requires PCA baseline output. Run PCA analysis first."),
    }
    for label, (path, message) in checks.items():
        found = path.exists()
        _log(f"Autoencoder input {label}: {'found' if found else 'missing'} - {path}")
        if not found:
            raise RuntimeError(f"{message} Missing {label}: {path}")
    manifest = controls_dir(case_id) / "clean_controls_manifest.md"
    if manifest.exists():
        _log(f"Autoencoder input clean_controls_manifest.md: found - {manifest}")
    else:
        _log(f"Autoencoder input clean_controls_manifest.md: missing optional manifest - {manifest}")
    validation = read_json(validation_path(case_id))
    if not (validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target")):
        raise RuntimeError("Autoencoder requires a human-validated track. Run tracking first.")
    controls_metrics = read_json(controls_dir(case_id) / "controls_metrics.json")
    if controls_metrics.get("controls_version") != CONTROLS_VERSION:
        raise RuntimeError(f"Autoencoder requires controls baseline output. Expected {CONTROLS_VERSION}, got {controls_metrics.get('controls_version')!r}.")
    if float(controls_metrics.get("control_validity_score") or 0) < 0.45:
        raise RuntimeError("Autoencoder requires controls baseline output. Control validity score is below 0.45.")
    pca_metrics = read_json(pca_metrics_path(case_id))
    return read_json(track_path(case_id)), validation, controls_metrics, pca_metrics


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


def _read_dynamic_rois(case_id: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with dynamic_rois_csv(case_id).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row["frame"]))
            rows[frame] = row
    return rows


def _load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _load_samples(case_id: str, quick: bool) -> list[dict[str, Any]]:
    object_limit = 180 if quick else 320
    control_limit = 90 if quick else 160
    samples: list[dict[str, Any]] = []
    for path in _paths_sampled(list(object_crops_dir(case_id).glob("*.png")), object_limit):
        image = _load_image(path)
        if image is None:
            continue
        samples.append({"path": path, "frame": _frame_from_name(path), "class_label": "object", "image": image, "control_status": ""})
    root = controls_dir(case_id) / "controls"
    for label in EVAL_CONTROL_CLASSES:
        folder = root / label
        if not folder.exists():
            continue
        for path in _paths_sampled(list(folder.glob("frame_*.png")), control_limit):
            image = _load_image(path)
            if image is None:
                continue
            status = "artifact_isolated" if label == "hud_artifact" else "clean"
            samples.append({"path": path, "frame": _frame_from_name(path), "class_label": label, "image": image, "control_status": status})
    return samples


def _tensor(samples: list[dict[str, Any]]) -> torch.Tensor:
    arr = np.stack([sample["image"] for sample in samples]).astype(np.float32)
    return torch.from_numpy(arr[:, None, :, :])


def _train_model(samples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[SmallConvAutoencoder, list[dict[str, float]]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_samples = [s for s in samples if s["class_label"] in TRAIN_CONTROL_CLASSES and s["control_status"] == "clean"]
    if len(train_samples) < 24:
        raise RuntimeError("Autoencoder requires controls baseline output. Not enough clean control samples to train.")
    dataset = TensorDataset(_tensor(train_samples))
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    model = SmallConvAutoencoder(latent_dim=config["latent_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    curve: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None
    patience = config["early_stopping_patience"]
    stale = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(config["device"])
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(config["device"])
                val_losses.append(float(loss_fn(model(batch), batch).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if config["early_stopping_used"] and stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, curve


def _evaluate(model: SmallConvAutoencoder, samples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    x = _tensor(samples).to(config["device"])
    errors: list[dict[str, Any]] = []
    recons = []
    latents = []
    with torch.no_grad():
        for start in range(0, len(samples), config["batch_size"]):
            batch = x[start : start + config["batch_size"]]
            recon = model(batch)
            latent = model.encoder(batch)
            err = torch.mean((recon - batch) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            recons.append(recon.detach().cpu().numpy())
            latents.append(latent.detach().cpu().numpy())
            for idx, value in enumerate(err):
                errors.append({**samples[start + idx], "reconstruction_error": float(value)})
    return errors, x.detach().cpu().numpy(), np.concatenate(recons, axis=0), np.concatenate(latents, axis=0)


def _percentile(values: np.ndarray, value: float) -> float:
    return float(100.0 * np.mean(values <= value)) if len(values) else 0.0


def _metrics(
    case_id: str,
    errors: list[dict[str, Any]],
    curve: list[dict[str, float]],
    config: dict[str, Any],
    controls_metrics: dict[str, Any],
    pca_metrics: dict[str, Any],
) -> dict[str, Any]:
    labels = sorted(set(item["class_label"] for item in errors))
    by_class = {label: np.array([item["reconstruction_error"] for item in errors if item["class_label"] == label], dtype=np.float64) for label in labels}
    control_values = np.concatenate([by_class[label] for label in TRAIN_CONTROL_CLASSES if label in by_class])
    control_mean = float(control_values.mean()) if len(control_values) else 0.0
    control_std = float(control_values.std(ddof=1)) if len(control_values) > 1 else 1e-9
    object_values = by_class.get("object", np.array([], dtype=np.float64))
    object_mean = float(object_values.mean()) if len(object_values) else 0.0
    object_std = float(object_values.std(ddof=1)) if len(object_values) > 1 else 0.0
    object_z = (object_values - control_mean) / (control_std + 1e-9) if len(object_values) else np.array([])
    object_percentiles = np.array([_percentile(control_values, value) for value in object_values]) if len(object_values) else np.array([])
    hud_mean = float(by_class["hud_artifact"].mean()) if "hud_artifact" in by_class and len(by_class["hud_artifact"]) else None
    background_mean = float(np.mean([by_class[label].mean() for label in ["near_background", "far_background", "random_background"] if label in by_class and len(by_class[label])])) if any(label in by_class for label in ["near_background", "far_background", "random_background"]) else 0.0
    compression_mean = float(by_class["compression_noise"].mean()) if "compression_noise" in by_class and len(by_class["compression_noise"]) else 0.0
    public_z = float(np.mean(object_z)) if len(object_z) else 0.0
    exploratory_z = float(np.max(object_z)) if len(object_z) else 0.0
    ratio_bg = float(object_mean / (background_mean + 1e-9)) if background_mean else 0.0
    ratio_comp = float(object_mean / (compression_mean + 1e-9)) if compression_mean else 0.0
    ratio_hud = float(object_mean / (hud_mean + 1e-9)) if hud_mean else None
    anomaly_public = float(max(0.0, min(1.0, (np.mean(object_percentiles) / 100.0 if len(object_percentiles) else 0.0) * controls_metrics.get("control_validity_score", 0))))
    anomaly_exploratory = float(max(0.0, min(1.0, (_percentile(control_values, float(object_values.max())) / 100.0 if len(object_values) else 0.0))))
    return {
        "case_id": case_id,
        "mode": config["mode"],
        "training_strategy": config["training_strategy"],
        "model_type": "SmallConvAutoencoder",
        "input_crop_size": [INPUT_SIZE, INPUT_SIZE],
        "latent_dim": config["latent_dim"],
        "epochs": len(curve),
        "early_stopping_used": config["early_stopping_used"],
        "train_samples": int(sum(len(by_class.get(label, [])) for label in TRAIN_CONTROL_CLASSES)),
        "eval_samples_per_class": {label: int(len(values)) for label, values in by_class.items()},
        "train_loss_final": curve[-1]["train_loss"] if curve else None,
        "val_loss_final": curve[-1]["val_loss"] if curve else None,
        "reconstruction_error_mean_object": object_mean,
        "reconstruction_error_std_object": object_std,
        "reconstruction_error_mean_near_background": float(by_class["near_background"].mean()) if "near_background" in by_class else None,
        "reconstruction_error_mean_far_background": float(by_class["far_background"].mean()) if "far_background" in by_class else None,
        "reconstruction_error_mean_compression_noise": compression_mean,
        "reconstruction_error_mean_random_background": float(by_class["random_background"].mean()) if "random_background" in by_class else None,
        "reconstruction_error_mean_hud_artifact": hud_mean,
        "object_vs_background_error_ratio": ratio_bg,
        "object_vs_compression_error_ratio": ratio_comp,
        "object_vs_hud_error_ratio": ratio_hud,
        "object_error_percentile_vs_controls": float(np.mean(object_percentiles)) if len(object_percentiles) else 0.0,
        "mean_object_zscore_vs_controls": public_z,
        "max_object_zscore_vs_controls": exploratory_z,
        "public_safe_zscore": public_z,
        "exploratory_max_zscore": exploratory_z,
        "anomaly_score_public_safe": anomaly_public,
        "anomaly_score_exploratory": anomaly_exploratory,
        "controls_version": controls_metrics.get("controls_version"),
        "control_validity_score": controls_metrics.get("control_validity_score"),
        "pca_public_safe_score": pca_metrics.get("pca_public_safe_score"),
        "notes": [
            "Autoencoder was trained on Controls v0.2 clean masked baseline classes, not on automatically detected ROIs.",
            "HUD artifacts are evaluated separately and are not treated as normal clean background.",
            "Autoencoder reconstruction error indicates statistical/visual mismatch under this model; it does not determine origin.",
            "Extreme z-scores are exploratory and must not be used alone as public evidence.",
        ],
    }


def _write_timeseries(path: Path, errors: list[dict[str, Any]], dynamic: dict[int, dict[str, Any]], control_values: np.ndarray) -> None:
    mean = float(control_values.mean()) if len(control_values) else 0.0
    std = float(control_values.std(ddof=1)) if len(control_values) > 1 else 1e-9
    fields = ["frame", "timestamp", "class_label", "reconstruction_error", "reconstruction_error_zscore_vs_controls", "percentile_vs_controls", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "track_status", "control_status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in errors:
            frame = item.get("frame")
            roi = dynamic.get(frame or -1, {})
            err = float(item["reconstruction_error"])
            writer.writerow(
                {
                    "frame": frame,
                    "timestamp": "" if frame is None else frame / 30.0,
                    "class_label": item["class_label"],
                    "reconstruction_error": err,
                    "reconstruction_error_zscore_vs_controls": (err - mean) / (std + 1e-9),
                    "percentile_vs_controls": _percentile(control_values, err),
                    "bbox_x": roi.get("bbox_x", ""),
                    "bbox_y": roi.get("bbox_y", ""),
                    "bbox_w": roi.get("bbox_w", ""),
                    "bbox_h": roi.get("bbox_h", ""),
                    "track_status": roi.get("status", "TRACKING_ACTIVE") if item["class_label"] == "object" else "",
                    "control_status": item.get("control_status", ""),
                }
            )


def _write_curve(path: Path, curve: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(curve)


def _error_distribution_panel(path: Path, errors: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["object", *EVAL_CONTROL_CLASSES]
    for label in labels:
        values = [item["reconstruction_error"] for item in errors if item["class_label"] == label]
        if values:
            ax.hist(values, bins=24, alpha=0.45, label=label)
    ax.axvline(metrics["reconstruction_error_mean_object"], color="#e15759", linewidth=2, label="object mean")
    ax.set_title("Autoencoder reconstruction error distribution")
    ax.set_xlabel("MSE reconstruction error")
    ax.set_ylabel("samples")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _timeseries_panel(path: Path, errors: list[dict[str, Any]], control_values: np.ndarray) -> None:
    object_items = sorted([item for item in errors if item["class_label"] == "object" and item.get("frame") is not None], key=lambda item: item["frame"])
    if not object_items:
        return
    mean = float(control_values.mean()) if len(control_values) else 0.0
    std = float(control_values.std(ddof=1)) if len(control_values) > 1 else 1e-9
    frames = [item["frame"] for item in object_items]
    values = np.array([item["reconstruction_error"] for item in object_items])
    z = (values - mean) / (std + 1e-9)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, values, color="#4e79a7")
    axes[0].axhline(mean, color="#59a14f", linestyle="--", label="control mean")
    axes[0].set_title("Object reconstruction error by frame")
    axes[0].legend()
    axes[1].plot(frames, z, color="#e15759")
    axes[1].axhline(float(np.mean(z)), color="#f28e2b", linestyle="--", label="public-safe mean z")
    axes[1].scatter([frames[int(np.argmax(z))]], [float(np.max(z))], color="#af7aa1", label="exploratory peak")
    axes[1].set_title("Z-score vs clean controls")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reconstruction_examples_panel(path: Path, errors: list[dict[str, Any]], originals: np.ndarray, recons: np.ndarray) -> None:
    selected: list[tuple[str, int]] = []
    for label in ["object", "near_background", "compression_noise", "random_background", "hud_artifact"]:
        idx = next((i for i, item in enumerate(errors) if item["class_label"] == label), None)
        if idx is not None:
            selected.append((label, idx))
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, max(3, 2.3 * len(selected))))
    if len(selected) == 1:
        axes = np.array([axes])
    for row, (label, idx) in enumerate(selected):
        original = originals[idx, 0]
        recon = recons[idx, 0]
        err = np.abs(original - recon)
        for col, (img, title) in enumerate([(original, "original"), (recon, "reconstruction"), (err, "error map")]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{label} {title}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _latent_panel(path: Path, errors: list[dict[str, Any]], latents: np.ndarray) -> bool:
    if latents.shape[0] < 8 or latents.shape[1] < 2:
        return False
    scores = PCA(n_components=2, random_state=SEED).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in ["object", *EVAL_CONTROL_CLASSES]:
        idx = [i for i, item in enumerate(errors) if item["class_label"] == label]
        if idx:
            ax.scatter(scores[idx, 0], scores[idx, 1], s=20, alpha=0.72, label=label)
    ax.set_title("Autoencoder latent PCA")
    ax.set_xlabel("latent PC1")
    ax.set_ylabel("latent PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _summary_panel(path: Path, metrics: dict[str, Any]) -> None:
    classes = ["object", "near_background", "far_background", "compression_noise", "random_background", "hud_artifact"]
    values = [
        metrics.get("reconstruction_error_mean_object"),
        metrics.get("reconstruction_error_mean_near_background"),
        metrics.get("reconstruction_error_mean_far_background"),
        metrics.get("reconstruction_error_mean_compression_noise"),
        metrics.get("reconstruction_error_mean_random_background"),
        metrics.get("reconstruction_error_mean_hud_artifact"),
    ]
    valid = [(label, value) for label, value in zip(classes, values) if value is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([label for label, _ in valid], [value for _, value in valid], color="#4e79a7")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Mean reconstruction error by class")
    axes[1].bar(
        ["object percentile", "public-safe score", "exploratory score"],
        [metrics["object_error_percentile_vs_controls"], metrics["anomaly_score_public_safe"] * 100, metrics["anomaly_score_exploratory"] * 100],
        color=["#e15759", "#59a14f", "#af7aa1"],
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Public-safe vs exploratory summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(case_id: str, validation: dict[str, Any], metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    report_dir = ensure_dir(case_report_dir(case_id))
    report = report_dir / "autoencoder_analysis_report.md"
    lines = [
        f"# Autoencoder Track-Based + Clean Controls Baseline Report: {case_id}",
        "",
        "## Scope",
        "",
        "This module trains a small CPU-friendly autoencoder on Controls v0.2 clean masked baseline crops, then evaluates the human-validated tracked object against those controls.",
        "",
        "Autoencoder reconstruction error indicates statistical/visual mismatch under this model; it does not determine origin.",
        "",
        "Extreme z-scores are exploratory and must not be used alone as public evidence.",
        "",
        "## Traceability",
        "",
        f"- track: `{track_path(case_id)}`",
        f"- validation: `{validation_path(case_id)}`",
        f"- human validated: `{bool(validation.get('track_validated'))}`",
        f"- dynamic ROIs: `{dynamic_rois_csv(case_id)}`",
        f"- controls manifest: `{controls_dir(case_id) / 'clean_controls_manifest.md'}`",
        f"- PCA baseline: `{pca_metrics_path(case_id)}`",
        f"- controls version: `{metrics.get('controls_version')}`",
        f"- control validity score: `{metrics.get('control_validity_score')}`",
        "",
        "## Training Strategy",
        "",
        f"- strategy: `{metrics['training_strategy']}`",
        f"- model: `{metrics['model_type']}`",
        f"- input crop size: `{metrics['input_crop_size']}`",
        f"- latent dim: `{metrics['latent_dim']}`",
        f"- epochs: `{metrics['epochs']}`",
        f"- train samples: `{metrics['train_samples']}`",
        f"- final train loss: `{metrics['train_loss_final']}`",
        f"- final val loss: `{metrics['val_loss_final']}`",
        "",
        "## Public-Safe Metrics",
        "",
        f"- mean object z-score vs controls: `{metrics['public_safe_zscore']:.6f}`",
        f"- object error percentile vs controls: `{metrics['object_error_percentile_vs_controls']:.6f}`",
        f"- public-safe anomaly score: `{metrics['anomaly_score_public_safe']:.6f}`",
        f"- object/background error ratio: `{metrics['object_vs_background_error_ratio']:.6f}`",
        f"- object/compression error ratio: `{metrics['object_vs_compression_error_ratio']:.6f}`",
        "",
        "## Exploratory Metrics",
        "",
        f"- max object z-score vs controls: `{metrics['exploratory_max_zscore']:.6f}`",
        f"- exploratory anomaly score: `{metrics['anomaly_score_exploratory']:.6f}`",
        "",
        "## Eval Samples",
        "",
    ]
    for label, count in metrics["eval_samples_per_class"].items():
        lines.append(f"- {label}: `{count}`")
    lines += [
        "",
        "## Limitations",
        "",
        "- The result depends on crop size, sample balance, controls quality, architecture and training duration.",
        "- Controls v0.2 are clean masked controls, but still require visual review per case.",
        "- HUD artifacts are evaluated separately and are not normal clean background.",
        "- This does not estimate distance, speed, material, intent or origin.",
        "- No automatic ROI, object redetection, generative model or original-video modification was used.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_manifest(case_id: str, outputs: dict[str, Path]) -> Path:
    manifest = output_dir(case_id) / "autoencoder_manifest.md"
    classes = {
        "autoencoder_metrics.json": "technical",
        "autoencoder_timeseries.csv": "technical",
        "autoencoder_training_curve.csv": "debug",
        "autoencoder_config.json": "technical",
        "autoencoder_model.pt": "technical",
        "autoencoder_error_distribution_panel.png": "public_safe",
        "autoencoder_timeseries_panel.png": "technical",
        "autoencoder_reconstruction_examples_panel.png": "technical",
        "autoencoder_latent_panel.png": "exploratory",
        "autoencoder_summary_panel.png": "interpretive",
        "autoencoder_analysis_report.md": "interpretive",
    }
    lines = ["# Autoencoder Analysis Manifest", "", f"Case: `{case_id}`", "", "| output | path | classification | caution |", "| --- | --- | --- | --- |"]
    for name, path in outputs.items():
        lines.append(f"| `{name}` | `{path}` | `{classes.get(name, 'debug')}` | reconstruction error is model-dependent and not an origin claim |")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _update_case_status(case_id: str, metrics: dict[str, Any], outputs: dict[str, Path], report: Path, config_path: Path) -> Path:
    path = case_status_path(case_id)
    existing = read_json(path) if path.exists() else {"case_id": case_id}
    existing.update(
        {
            "autoencoder_analysis_status": "complete",
            "autoencoder_analysis_ready": True,
            "last_autoencoder_analysis_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "autoencoder_public_safe_score": metrics["anomaly_score_public_safe"],
            "autoencoder_exploratory_score": metrics["anomaly_score_exploratory"],
            "autoencoder_analysis_paths": {
                "output_dir": str(output_dir(case_id)),
                "metrics": str(outputs["autoencoder_metrics.json"]),
                "timeseries": str(outputs["autoencoder_timeseries.csv"]),
                "training_curve": str(outputs["autoencoder_training_curve.csv"]),
                "report": str(report),
                "config": str(config_path),
                "panels": {
                    "error_distribution": str(outputs["autoencoder_error_distribution_panel.png"]),
                    "timeseries": str(outputs["autoencoder_timeseries_panel.png"]),
                    "reconstruction_examples": str(outputs["autoencoder_reconstruction_examples_panel.png"]),
                    "latent": str(outputs.get("autoencoder_latent_panel.png", "")),
                    "summary": str(outputs["autoencoder_summary_panel.png"]),
                },
            },
        }
    )
    write_json(path, existing)
    return path


def run_track_autoencoder_analysis(case_id: str, quick: bool = False) -> dict[str, Any]:
    _log(f"Starting Autoencoder analysis for case {case_id}")
    _track, validation, controls_metrics, pca_metrics = _require_inputs(case_id)
    _log("Loading Autoencoder samples from object crops and clean controls...")
    samples = _load_samples(case_id, quick=quick)
    if not any(s["class_label"] == "object" for s in samples):
        raise RuntimeError("Autoencoder requires dynamic ROI output. No object crops were available.")
    config: dict[str, Any] = {
        "seed": SEED,
        "mode": "quick" if quick else "standard",
        "training_strategy": "controls_trained",
        "model": "SmallConvAutoencoder",
        "latent_dim": 16 if quick else 32,
        "crop_size": [INPUT_SIZE, INPUT_SIZE],
        "normalization": "grayscale float32 [0,1]",
        "train_classes": TRAIN_CONTROL_CLASSES,
        "eval_classes": ["object", *EVAL_CONTROL_CLASSES],
        "epochs": 8 if quick else 18,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stopping_used": True,
        "early_stopping_patience": 3 if quick else 5,
        "device": "cpu",
        "controls_version": controls_metrics.get("controls_version"),
    }
    out = output_dir(case_id)
    _log(f"Writing Autoencoder outputs to {out}")
    config_path = out / "autoencoder_config.json"
    _log("Writing autoencoder_config.json")
    write_json(config_path, config)
    _log("Training autoencoder backend...")
    model, curve = _train_model(samples, config)
    _log("Evaluating object and control reconstruction errors...")
    errors, originals, recons, latents = _evaluate(model, samples, config)
    metrics = _metrics(case_id, errors, curve, config, controls_metrics, pca_metrics)
    control_values = np.array([item["reconstruction_error"] for item in errors if item["class_label"] in TRAIN_CONTROL_CLASSES], dtype=np.float64)
    dynamic = _read_dynamic_rois(case_id)
    outputs: dict[str, Path] = {
        "autoencoder_metrics.json": out / "autoencoder_metrics.json",
        "autoencoder_timeseries.csv": out / "autoencoder_timeseries.csv",
        "autoencoder_training_curve.csv": out / "autoencoder_training_curve.csv",
        "autoencoder_config.json": config_path,
        "autoencoder_error_distribution_panel.png": out / "autoencoder_error_distribution_panel.png",
        "autoencoder_timeseries_panel.png": out / "autoencoder_timeseries_panel.png",
        "autoencoder_reconstruction_examples_panel.png": out / "autoencoder_reconstruction_examples_panel.png",
        "autoencoder_summary_panel.png": out / "autoencoder_summary_panel.png",
    }
    _log("Writing autoencoder_metrics.json")
    write_json(outputs["autoencoder_metrics.json"], metrics)
    _log("Writing autoencoder_timeseries.csv")
    _write_timeseries(outputs["autoencoder_timeseries.csv"], errors, dynamic, control_values)
    _log("Writing autoencoder_training_curve.csv")
    _write_curve(outputs["autoencoder_training_curve.csv"], curve)
    _log("Writing Autoencoder panels")
    _error_distribution_panel(outputs["autoencoder_error_distribution_panel.png"], errors, metrics)
    _timeseries_panel(outputs["autoencoder_timeseries_panel.png"], errors, control_values)
    _reconstruction_examples_panel(outputs["autoencoder_reconstruction_examples_panel.png"], errors, originals, recons)
    if _latent_panel(out / "autoencoder_latent_panel.png", errors, latents):
        outputs["autoencoder_latent_panel.png"] = out / "autoencoder_latent_panel.png"
    _summary_panel(outputs["autoencoder_summary_panel.png"], metrics)
    model_path = out / "autoencoder_model.pt"
    _log("Writing autoencoder_model.pt")
    torch.save({"state_dict": model.state_dict(), "config": config, "metrics": metrics}, model_path)
    outputs["autoencoder_model.pt"] = model_path
    _log("Writing Autoencoder report")
    report = _write_report(case_id, validation, metrics, outputs)
    outputs["autoencoder_analysis_report.md"] = report
    manifest = _write_manifest(case_id, outputs)
    outputs["autoencoder_manifest.md"] = manifest
    status_path = _update_case_status(case_id, metrics, outputs, report, config_path)
    _log("Autoencoder analysis complete")
    return {
        "case_id": case_id,
        "autoencoder_analysis_ready": True,
        "output_dir": str(out),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "case_status": str(status_path),
    }
