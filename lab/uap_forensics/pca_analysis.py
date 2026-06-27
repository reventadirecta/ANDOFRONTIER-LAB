import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .paths import case_output_dir, ensure_dir
from .roi import load_roi_frames
from .visuals import save_image, save_panel


def run_pca_analysis(config: dict, max_components: int = 20) -> dict:
    case_id = config["case_id"]
    out = ensure_dir(case_output_dir(case_id) / "pca")
    frames = load_roi_frames(case_id, grayscale=True)
    h, w = frames[0].shape
    matrix = np.stack([frame.reshape(-1) for frame in frames]).astype(np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    normalized = scaler.fit_transform(matrix)
    n_components = min(max_components, normalized.shape[0], normalized.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(normalized)
    reconstructed_norm = pca.inverse_transform(scores)
    reconstructed = scaler.inverse_transform(reconstructed_norm)
    residual = np.abs(matrix - reconstructed)

    metrics = pd.DataFrame(
        {
            "frame_index": np.arange(matrix.shape[0]),
            "roi_mean": matrix.mean(axis=1),
            "roi_std": matrix.std(axis=1),
            "pca_residual_mean": residual.mean(axis=1),
            "pca_residual_max": residual.max(axis=1),
        }
    )
    mean_signature = matrix.mean(axis=0)
    metrics["corr_roi_vs_mean_signature"] = [
        np.corrcoef(row, mean_signature)[0, 1] if np.std(row) and np.std(mean_signature) else np.nan
        for row in matrix
    ]
    metrics["corr_frame_to_previous"] = np.nan
    for i in range(1, len(matrix)):
        metrics.loc[i, "corr_frame_to_previous"] = np.corrcoef(matrix[i - 1], matrix[i])[0, 1]
    metrics.to_csv(out / "pca_metrics.csv", index=False)

    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(out / "pca_explained_variance.csv", index=False)

    for k in (5, 10):
        k_eff = min(k, n_components)
        pca_k = PCA(n_components=k_eff, svd_solver="full")
        scores_k = pca_k.fit_transform(normalized)
        recon_k = scaler.inverse_transform(pca_k.inverse_transform(scores_k))
        mean_recon = recon_k.mean(axis=0).reshape(h, w)
        mean_residual = np.abs(matrix - recon_k).mean(axis=0).reshape(h, w)
        save_image(out / f"pca_reconstruction_k{k}.png", mean_recon)
        save_image(out / f"pca_residual_k{k}.png", mean_residual * 4)

    components = {}
    for idx in range(min(3, n_components)):
        comp = pca.components_[idx].reshape(h, w)
        comp = 255 * (comp - comp.min()) / (np.ptp(comp) + 1e-6)
        components[f"PC{idx + 1}"] = comp
    save_panel(
        out / "pca_panel.png",
        {
            "mean roi": matrix.mean(axis=0).reshape(h, w),
            "mean residual": residual.mean(axis=0).reshape(h, w) * 4,
            **components,
        },
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(explained["component"], explained["cumulative_variance_ratio"], marker="o")
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative variance")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out / "pca_explained_variance.png", dpi=150)
    plt.close(fig)
    return {"case_id": case_id, "output_dir": str(out), "components": int(n_components)}
