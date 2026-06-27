from pathlib import Path

import pandas as pd

from .paths import DATA_DIR, case_output_dir, ensure_dir


def collect_case_summary(case_id: str) -> dict:
    base = case_output_dir(case_id)
    row = {"case_id": case_id}
    pca_metrics = base / "pca" / "pca_metrics.csv"
    pca_var = base / "pca" / "pca_explained_variance.csv"
    ae_metrics = base / "autoencoder" / "autoencoder_metrics.csv"
    lum = base / "base" / "luminance_profiles.csv"
    if pca_metrics.exists():
        df = pd.read_csv(pca_metrics)
        row["pca_residual_mean"] = df["pca_residual_mean"].mean()
        row["pca_corr_mean_signature_mean"] = df["corr_roi_vs_mean_signature"].mean()
    if pca_var.exists():
        df = pd.read_csv(pca_var)
        row["pca_components_for_90pct"] = int((df["cumulative_variance_ratio"] < 0.9).sum() + 1)
    if ae_metrics.exists():
        df = pd.read_csv(ae_metrics)
        row["autoencoder_object_zscore_mean"] = df["object_error_zscore"].mean()
        row["autoencoder_object_percentile_mean"] = df["object_error_percentile_vs_background"].mean()
    if lum.exists():
        df = pd.read_csv(lum)
        row["luminance_std_mean"] = df["std_luminance"].mean()
    return row


def compare_against_controls(case_id: str, control_case_ids: list[str]) -> Path:
    rows = [collect_case_summary(case_id)]
    for control_id in control_case_ids:
        row = collect_case_summary(control_id)
        row["is_control"] = True
        rows.append(row)
    rows[0]["is_control"] = False
    out = ensure_dir(DATA_DIR / "outputs" / "comparisons")
    path = out / f"{case_id}_control_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
