import numpy as np
import pandas as pd

from .paths import case_output_dir, ensure_dir
from .roi import load_roi_frames
from .visuals import save_image, save_panel


def _torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _extract_background_patches(frames: np.ndarray, patch_size: int = 16, stride: int = 16) -> np.ndarray:
    patches = []
    _, h, w = frames.shape
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    for frame in frames:
        for y in range(0, max(1, h - patch_size + 1), stride):
            for x in range(0, max(1, w - patch_size + 1), stride):
                overlaps_center = x < cx1 and x + patch_size > cx0 and y < cy1 and y + patch_size > cy0
                if not overlaps_center:
                    patch = frame[y : y + patch_size, x : x + patch_size]
                    if patch.shape == (patch_size, patch_size):
                        patches.append(patch)
    if not patches:
        raise RuntimeError("No background patches available. Use a larger ROI or smaller patch size.")
    return np.stack(patches).astype(np.float32) / 255.0


def run_autoencoder_analysis(config: dict, epochs: int = 15, patch_size: int = 16) -> dict:
    torch, nn, DataLoader, TensorDataset = _torch()
    case_id = config["case_id"]
    out = ensure_dir(case_output_dir(case_id) / "autoencoder")
    frames = np.stack(load_roi_frames(case_id, grayscale=True)).astype(np.float32)
    patches = _extract_background_patches(frames, patch_size=patch_size)
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
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / len(x_train))

    rows = []
    anomaly_maps = []
    bg_errors = []
    object_errors = []
    for idx, frame in enumerate(frames / 255.0):
        flat = torch.tensor(frame.reshape(1, -1), dtype=torch.float32)
        full_model = nn.Sequential(
            nn.Linear(flat.shape[1], flat.shape[1]),
        )
        del full_model
        # Patch-level reconstruction is used for the anomaly map to keep the model local and conservative.
        error_map = np.zeros_like(frame)
        counts = np.zeros_like(frame)
        for y in range(0, frame.shape[0] - patch_size + 1, max(1, patch_size // 2)):
            for x in range(0, frame.shape[1] - patch_size + 1, max(1, patch_size // 2)):
                patch = frame[y : y + patch_size, x : x + patch_size]
                vector = torch.tensor(patch.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    recon = model(vector).numpy().reshape(patch_size, patch_size)
                err = np.abs(patch - recon)
                error_map[y : y + patch_size, x : x + patch_size] += err
                counts[y : y + patch_size, x : x + patch_size] += 1
        error_map = error_map / np.maximum(counts, 1)
        anomaly_maps.append(error_map)
        h, w = frame.shape
        obj = error_map[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
        bg = np.concatenate(
            [
                error_map[: int(h * 0.25), :].reshape(-1),
                error_map[int(h * 0.75) :, :].reshape(-1),
                error_map[:, : int(w * 0.25)].reshape(-1),
                error_map[:, int(w * 0.75) :].reshape(-1),
            ]
        )
        bg_mean = float(bg.mean())
        bg_std = float(bg.std() + 1e-9)
        obj_mean = float(obj.mean())
        bg_errors.append(bg_mean)
        object_errors.append(obj_mean)
        rows.append(
            {
                "frame_index": idx,
                "background_error_mean": bg_mean,
                "object_error_mean": obj_mean,
                "object_error_zscore": (obj_mean - bg_mean) / bg_std,
                "object_error_percentile_vs_background": float((bg < obj_mean).mean() * 100),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "autoencoder_metrics.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(out / "training_curve.csv", index=False)
    mean_map = np.mean(anomaly_maps, axis=0)
    save_image(out / "anomaly_map_mean.png", 255 * mean_map / (mean_map.max() + 1e-9))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)

    save_panel(
        out / "autoencoder_panel.png",
        {
            "mean frame": frames.mean(axis=0),
            "mean anomaly": 255 * mean_map / (mean_map.max() + 1e-9),
            "max anomaly": 255 * np.max(anomaly_maps, axis=0) / (np.max(anomaly_maps) + 1e-9),
        },
    )
    return {"case_id": case_id, "output_dir": str(out), "epochs": epochs, "patches": int(len(patches))}
