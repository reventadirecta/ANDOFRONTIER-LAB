from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


MODULE_ORDER = [
    "Tracking",
    "Controls",
    "Motion",
    "Spectral",
    "Thermal/IR",
    "SRV",
    "PCA",
    "Autoencoder",
]

STATUS_COLORS = {
    "strong": "#2ca02c",
    "moderate": "#4e79a7",
    "weak": "#f28e2b",
    "exploratory": "#af7aa1",
    "missing": "#9d9d9d",
    "failed": "#e15759",
    "ready": "#59a14f",
}


def output_dir(case_id: str) -> Path:
    return ensure_dir(case_report_dir(case_id) / "unified_report")


def case_status_path(case_id: str) -> Path:
    return DATA_DIR / "cases" / case_id / "case_status.json"


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path) if path.exists() else {}
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available in source metadata"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _public_text(value: Any, fallback: str = "not available in source metadata") -> str:
    if value is None or value == "" or value == {}:
        return fallback
    if isinstance(value, str) and value.lower() in {"unknown", "not available", "<source_video_path>"}:
        return fallback
    return str(value)


def _duration(value: Any) -> str:
    if value is None or value == "" or value == "unknown":
        return "not available in source metadata"
    try:
        return f"{float(value):.3f} s"
    except Exception:
        return str(value)


def _fps(value: Any) -> str:
    if value is None or value == "" or value == "unknown":
        return "not available in source metadata"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _resolution(value: Any) -> str:
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if width and height:
            return f"{width}x{height}"
    if isinstance(value, str) and value and value.lower() != "unknown":
        return value
    return "not available in source metadata"


def _tracking_quality(data: dict[str, Any]) -> dict[str, Any]:
    track = data.get("track", {})
    quality = data.get("tracking_quality", {})
    if quality:
        return quality
    if isinstance(track.get("tracking_quality"), dict):
        return track["tracking_quality"]
    if isinstance(track.get("summary"), dict):
        return track["summary"]
    return {}


def _strength_from_score(score: float | None, strong: float = 0.75, moderate: float = 0.45) -> str:
    if score is None:
        return "missing"
    if score >= strong:
        return "strong"
    if score >= moderate:
        return "moderate"
    if score > 0:
        return "weak"
    return "missing"


def _load_case(case_id: str) -> dict[str, Any]:
    base = DATA_DIR
    source = {
        **_safe_json(base / "cases" / case_id / "batch_source.json"),
        **_safe_json(base / "cases" / case_id / "case_metadata.json"),
    }
    return {
        "case_status": _safe_json(base / "cases" / case_id / "case_status.json"),
        "batch_source": source,
        "track": _safe_json(base / "outputs" / case_id / "interactive_tracking" / "track.json"),
        "tracking_quality": _safe_json(base / "outputs" / case_id / "interactive_tracking" / "tracking_quality.json"),
        "validation": _safe_json(base / "cases" / case_id / "track_validation.json"),
        "motion": _safe_json(base / "outputs" / case_id / "motion_analysis" / "motion_metrics.json"),
        "spectral": _safe_json(base / "outputs" / case_id / "spectral_analysis" / "spectral_metrics.json"),
        "thermal": _safe_json(base / "outputs" / case_id / "thermal_analysis" / "thermal_metrics.json"),
        "srv": _safe_json(base / "outputs" / case_id / "srv_analysis" / "srv_metrics.json"),
        "srv_core": _safe_json(base / "outputs" / case_id / "srv_analysis" / "object_core" / "srv_core_metrics.json"),
        "controls": _safe_json(base / "outputs" / case_id / "controls_analysis" / "controls_metrics.json"),
        "pca": _safe_json(base / "outputs" / case_id / "pca_analysis" / "pca_metrics.json"),
        "autoencoder": _safe_json(base / "outputs" / case_id / "autoencoder_analysis" / "autoencoder_metrics.json"),
    }


def _paths(case_id: str) -> dict[str, Path]:
    base = DATA_DIR
    return {
        "tracking_report": base / "outputs" / case_id / "interactive_tracking" / "tracking_report.md",
        "dynamic_rois": base / "outputs" / case_id / "track_based_analysis" / "dynamic_rois.csv",
        "track_based_report": base / "reports" / case_id / "track_based_analysis_report.md",
        "motion_report": base / "reports" / case_id / "motion_analysis_report.md",
        "spectral_report": base / "reports" / case_id / "spectral_analysis_report.md",
        "thermal_report": base / "reports" / case_id / "thermal_analysis_report.md",
        "srv_report": base / "reports" / case_id / "srv_analysis_report.md",
        "controls_report": base / "reports" / case_id / "controls_report.md",
        "pca_report": base / "reports" / case_id / "pca_analysis_report.md",
        "autoencoder_report": base / "reports" / case_id / "autoencoder_analysis_report.md",
    }


def _module_status(data: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    validation = data["validation"]
    motion = data["motion"]
    spectral = data["spectral"]
    thermal = data["thermal"]
    srv = data["srv"]
    srv_core = data["srv_core"]
    controls = data["controls"]
    pca = data["pca"]
    ae = data["autoencoder"]
    tracking = "strong" if validation.get("track_validated") and validation.get("track_is_correct") and validation.get("object_is_real_target") else "missing"
    roi = "ready" if _exists(paths["dynamic_rois"]) else "missing"
    motion_strength = _strength_from_score(motion.get("motion_continuity_score"))
    if motion_strength == "strong" and float(motion.get("track_stability_score") or 0) < 0.5:
        motion_strength = "moderate"
    spectral_strength = _strength_from_score(spectral.get("high_frequency_energy_ratio"), strong=0.55, moderate=0.35)
    thermal_strength = _strength_from_score(thermal.get("thermal_contrast_index"), strong=0.12, moderate=0.04)
    srv_strength = "missing"
    if srv:
        srv_strength = "moderate"
        if srv_core and int(srv_core.get("low_confidence_frames") or 0) >= int(srv_core.get("valid_core_frames") or 1):
            srv_strength = "weak"
    controls_strength = _strength_from_score(controls.get("control_validity_score"), strong=0.8, moderate=0.55)
    pca_strength = _strength_from_score(pca.get("pca_public_safe_score"), strong=0.75, moderate=0.45)
    ae_strength = _strength_from_score(ae.get("anomaly_score_public_safe"), strong=0.75, moderate=0.6)
    if ae and ae_strength == "weak" and float(ae.get("exploratory_max_zscore") or 0) > 3:
        ae_strength = "exploratory"
    return {
        "tracking": tracking,
        "human_validation": tracking,
        "dynamic_roi": roi,
        "motion": motion_strength,
        "spectral": spectral_strength,
        "thermal_ir": thermal_strength,
        "srv": srv_strength,
        "controls": controls_strength,
        "pca": pca_strength,
        "autoencoder": ae_strength,
    }


def _overall(status: dict[str, str]) -> str:
    strong = sum(1 for value in status.values() if value == "strong")
    moderate = sum(1 for value in status.values() if value == "moderate")
    exploratory = sum(1 for value in status.values() if value == "exploratory")
    missing = sum(1 for value in status.values() if value == "missing")
    if strong >= 3 and moderate >= 2:
        return "Public-safe technical anomaly assessment is usable with limitations; no origin claim."
    if strong + moderate >= 4:
        return "Public-safe technical assessment is usable but mixed; emphasize limitations and controls."
    if exploratory and strong + moderate >= 2:
        return "Assessment is mostly exploratory; use only with careful caveats."
    if missing > 4:
        return "Assessment incomplete; more modules or review required."
    return "Assessment is limited; do not present as strong evidence."


def _scorecard(path: Path, status: dict[str, str]) -> None:
    labels = ["Tracking", "Controls", "Motion", "Spectral", "Thermal/IR", "SRV", "PCA", "Autoencoder"]
    keys = ["tracking", "controls", "motion", "spectral", "thermal_ir", "srv", "pca", "autoencoder"]
    colors = [STATUS_COLORS.get(status.get(key, "missing"), "#9d9d9d") for key in keys]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(labels, [1] * len(labels), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Unified Case Scorecard - no origin claim")
    for idx, key in enumerate(keys):
        ax.text(0.5, idx, status.get(key, "missing"), va="center", ha="center", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    out = ["| item | status | notes |", "| --- | --- | --- |"]
    out.extend(f"| {a} | `{b}` | {c} |" for a, b, c in rows)
    return out


def _write_markdown(case_id: str, data: dict[str, Any], status: dict[str, str], outputs: dict[str, Path], overall: str) -> Path:
    source = data["batch_source"]
    track = data["track"]
    tracking_quality = _tracking_quality(data)
    validation = data["validation"]
    motion = data["motion"]
    spectral = data["spectral"]
    thermal = data["thermal"]
    srv = data["srv"]
    srv_core = data["srv_core"]
    controls = data["controls"]
    pca = data["pca"]
    ae = data["autoencoder"]
    track_summary = track.get("summary", {})
    modules_completed = [name for name, state in status.items() if state not in {"missing", "failed"}]
    lines = [
        "# AndoFrontier Unified UAP Forensic Report",
        "",
        "## Case",
        "",
        f"- case_id: `{case_id}`",
        f"- original filename: `{_public_text(source.get('original_filename'))}`",
        f"- duration: `{_duration(source.get('duration_seconds'))}`",
        f"- fps: `{_fps(source.get('fps'))}`",
        f"- resolution: `{_resolution(source.get('resolution'))}`",
        f"- codec: `{_public_text(source.get('codec'))}`",
        f"- source quality: `{_public_text(source.get('source_type') or source.get('source_quality'), 'not independently verified')}`",
        f"- source_ir_mode: `{_public_text(thermal.get('source_ir_mode') or data['case_status'].get('source_ir_mode'))}`",
        "",
        "## Executive Summary",
        "",
        f"The case has tracking validation status `{status['tracking']}` and `{len(modules_completed)}` workflow gates or modules with available outputs.",
        f"Overall public-safe assessment: **{overall}**",
        "",
        "- Strongest support comes from validated tracking and clean controls when those modules are marked strong.",
        "- Moderate modules provide auxiliary measurements but should not be read as standalone proof.",
        "- Exploratory metrics, especially peak autoencoder z-scores, are retained for internal review only.",
        "- The analysis does not determine origin, nature, intent, distance or material.",
        "",
        "## Workflow Gate Status",
        "",
        "*_Status values: ready, missing, weak, exploratory, failed, moderate, strong._",
        "",
        "*_Dynamic ROI is a gate; module strengths are interpretive labels._",
        "",
    ]
    lines += _table(
        [
            ("tracking", status["tracking"], "manual object selection and tracker output"),
            ("human validation", status["human_validation"], "track_validation.json"),
            ("dynamic ROI", status["dynamic_roi"], "dynamic_rois.csv"),
            ("motion", status["motion"], "Motion / Optical Flow metrics"),
            ("spectral", status["spectral"], "luminance and frequency metrics"),
            ("thermal/IR", status["thermal_ir"], "relative IR intensity only"),
            ("SRV", status["srv"], "visual reconstruction, non-generative"),
            ("controls", status["controls"], "Controls v0.2 clean masked"),
            ("PCA", status["pca"], "baseline PCA vs controls"),
            ("autoencoder", status["autoencoder"], "baseline autoencoder vs controls"),
        ]
    )
    lines += [
        "",
        "## Tracking Summary",
        "",
        f"- tracker used: `{motion.get('tracker_used') or track.get('backend') or track_summary.get('backend') or 'unknown'}`",
        f"- tracking_valid_frames: `{tracking_quality.get('tracked_frames') or motion.get('valid_tracked_frames') or track_summary.get('tracked_frames') or 'unknown'}`",
        f"- tracking_lost_frames: `{tracking_quality.get('lost_frames') if tracking_quality.get('lost_frames') is not None else track_summary.get('lost_frames', 0)}`",
        f"- tracking_low_confidence_frames: `{tracking_quality.get('low_confidence_frames') if tracking_quality.get('low_confidence_frames') is not None else track_summary.get('low_confidence_frames', 0)}`",
        f"- tracking_predicted_only_frames: `{tracking_quality.get('predicted_only_frames') if tracking_quality.get('predicted_only_frames') is not None else track_summary.get('predicted_only_frames', 0)}`",
        f"- continuity: `{_fmt(motion.get('motion_continuity_score'))}`",
        f"- validation status: `{validation}`",
        "- bbox source: human-marked object prompt followed by tracker; no automatic ROI substitution in downstream modules.",
        "- limitation: all downstream measurements depend on the correctness of this track.",
        "",
        "## Motion / Optical Flow Summary",
        "",
        f"- mean velocity px/frame: `{_fmt(motion.get('mean_velocity_px_frame'))}`",
        f"- max velocity px/frame: `{_fmt(motion.get('max_velocity_px_frame'))}`",
        f"- mean acceleration px/frame^2: `{_fmt(motion.get('mean_acceleration_px_frame2'))}`",
        f"- mean optical flow: `{_fmt(motion.get('mean_optical_flow_magnitude_inside_roi'))}`",
        f"- max optical flow: `{_fmt(motion.get('max_optical_flow_magnitude_inside_roi'))}`",
        f"- continuity score: `{_fmt(motion.get('motion_continuity_score'))}`",
        f"- stability score: `{_fmt(motion.get('track_stability_score'))}`",
        f"- motion_lost_frames: `{motion.get('lost_frames', 0)}`",
        f"- interpretation: `{status['motion']}` motion signal in pixel-space only; not real-world speed.",
        "",
        "## Spectral Summary",
        "",
        f"- mean luminance: `{_fmt(spectral.get('mean_luminance'))}`",
        f"- luminance std: `{_fmt(spectral.get('luminance_std'))}`",
        f"- flicker index: `{_fmt(spectral.get('luminance_flicker_index'))}`",
        f"- dominant temporal frequency Hz: `{_fmt(spectral.get('dominant_temporal_frequency_hz'))}`",
        f"- spectral entropy: `{_fmt(spectral.get('spectral_entropy_temporal'))}`",
        f"- high-frequency ratio: `{_fmt(spectral.get('high_frequency_energy_ratio'))}`",
        f"- noise/compression proxy: `{_fmt(spectral.get('compression_noise_proxy'))}`",
        f"- interpretation: `{status['spectral']}` spectral signal; depends on compression and sensor processing.",
        "",
        "## Thermal / IR Summary",
        "",
        f"- source_ir_mode: `{thermal.get('source_ir_mode', 'unknown')}`",
        f"- calibration_available: `{thermal.get('calibration_available', False)}`",
        f"- temperature_units_available: `{thermal.get('temperature_units_available', False)}`",
        f"- mean ROI intensity: `{_fmt(thermal.get('mean_roi_intensity'))}`",
        f"- mean background intensity: `{_fmt(thermal.get('mean_background_intensity'))}`",
        f"- delta ROI-background: `{_fmt(thermal.get('roi_background_delta_mean'))}`",
        f"- hot/cold pixel ratio: `{_fmt(thermal.get('hot_pixel_ratio'))}` / `{_fmt(thermal.get('cold_pixel_ratio'))}`",
        f"- thermal contrast index: `{_fmt(thermal.get('thermal_contrast_index'))}`",
        f"- intensity stability score: `{_fmt(thermal.get('intensity_stability_score'))}`",
        "- Relative IR intensity only; no radiometric temperature units available.",
        "",
        "## SRV / Visual Reconstruction Summary",
        "",
        f"- crop count: `{srv.get('crop_count', 'unknown')}`",
        f"- normalized size: `{srv.get('normalized_crop_size', 'unknown')}`",
        f"- sharpness: `{_fmt(srv.get('mean_crop_sharpness'))}`",
        f"- contrast: `{_fmt(srv.get('mean_contrast'))}`",
        f"- super_resolution_used: `{srv.get('super_resolution_used', False)}`",
        f"- object-core confidence: `{status['srv']}`",
        f"- low confidence frames: `{srv_core.get('low_confidence_frames', 'unknown')}`",
        f"- artifact contamination score: `{_fmt(srv_core.get('artifact_contamination_score'))}`",
        "- interpretation: bbox context is usable; object-core reconstruction may be low confidence depending on metrics. No generative model used.",
        "",
        "## Controls Summary",
        "",
        f"- controls version: `{controls.get('controls_version', 'missing')}`",
        f"- control_validity_score: `{_fmt(controls.get('control_validity_score'))}`",
        f"- artifact_contamination_rate: `{_fmt(controls.get('artifact_contamination_rate'))}`",
        f"- hud_leakage_rate: `{_fmt(controls.get('hud_leakage_rate'))}`",
        f"- fallback_control_rate: `{_fmt(controls.get('fallback_control_rate'))}`",
        f"- accepted/rejected candidates: `{controls.get('accepted_control_candidates', 'unknown')}` / `{controls.get('rejected_control_candidates', 'unknown')}`",
        f"- object vs background delta IR intensity: `{_fmt(controls.get('object_vs_background_delta_ir_intensity'))}`",
        f"- HUD similarity: `{_fmt(controls.get('hud_similarity_score'))}`",
        f"- compression similarity: `{_fmt(controls.get('compression_similarity_score'))}`",
        f"- interpretation: `{status['controls']}` controls baseline quality.",
        "",
        "## PCA Summary",
        "",
        f"- object samples: `{pca.get('total_object_samples', 'missing')}`",
        f"- control samples: `{pca.get('total_control_samples', 'missing')}`",
        f"- PC1: `{_fmt(pca.get('pca_pc1_explained_variance'))}`",
        f"- PC2: `{_fmt(pca.get('pca_pc2_explained_variance'))}`",
        f"- k5: `{_fmt(pca.get('pca_k5_explained_variance'))}`",
        f"- k10: `{_fmt(pca.get('pca_k10_explained_variance'))}`",
        f"- object_vs_background_distance: `{_fmt(pca.get('object_vs_background_distance'))}`",
        f"- object_vs_compression_distance: `{_fmt(pca.get('object_vs_compression_distance'))}`",
        f"- object_vs_hud_distance: `{_fmt(pca.get('object_vs_hud_distance'))}`",
        f"- silhouette score: `{_fmt(pca.get('silhouette_score'))}`",
        f"- pca_public_safe_score: `{_fmt(pca.get('pca_public_safe_score'))}`",
        f"- interpretation: PCA `{status['pca']}` visual-statistical separation from clean controls.",
        "",
        "## Autoencoder Summary",
        "",
        f"- training strategy: `{ae.get('training_strategy', 'missing')}`",
        f"- train samples: `{ae.get('train_samples', 'missing')}`",
        f"- eval samples: `{ae.get('eval_samples_per_class', {})}`",
        f"- epochs: `{ae.get('epochs', 'missing')}`",
        f"- object percentile vs controls: `{_fmt(ae.get('object_error_percentile_vs_controls'))}`",
        f"- public_safe_zscore: `{_fmt(ae.get('public_safe_zscore'))}`",
        f"- exploratory max zscore: `{_fmt(ae.get('exploratory_max_zscore'))}`",
        f"- anomaly_score_public_safe: `{_fmt(ae.get('anomaly_score_public_safe'))}`",
        f"- anomaly_score_exploratory: `{_fmt(ae.get('anomaly_score_exploratory'))}`",
        f"- interpretation: Autoencoder `{status['autoencoder']}` public-safe anomaly signal; peaks are exploratory.",
        "",
        "## Cross-Module Assessment",
        "",
    ]
    lines += _table(
        [
            ("tracking confidence", status["tracking"], "human validation and continuity"),
            ("control quality", status["controls"], "clean masked controls"),
            ("motion signal strength", status["motion"], "pixel-space motion only"),
            ("spectral signal strength", status["spectral"], "tracked ROI spectral metrics"),
            ("IR contrast strength", status["thermal_ir"], "relative intensity only"),
            ("PCA separation strength", status["pca"], "dimensionality reduction baseline"),
            ("autoencoder anomaly strength", status["autoencoder"], "model-dependent reconstruction error"),
            ("SRV reconstruction confidence", status["srv"], "interpretive visual reconstruction"),
        ]
    )
    lines += [
        "",
        "## Public-Safe Statement",
        "",
        "The object was tracked with human validation, dynamic ROIs were reconstructed from that track, and the resulting object crops were compared against clean masked controls. Motion, spectral, relative IR, PCA and autoencoder modules provide traceable measurements, with some modules showing visual/statistical differences under their respective methods. This analysis does not determine origin, physical nature, material, distance, intent or non-human provenance.",
        "",
        "## Limitations",
        "",
        "- FLIR/IR sensor stream has no radiometric calibration available.",
        "- Compression, codec processing and possible auto-gain can affect all visual metrics.",
        "- Source chain is not independently verified unless source metadata says otherwise.",
        "- Results depend on the human-validated tracking path.",
        "- ROI and crop choices constrain all module outputs.",
        "- SRV is interpretive and non-generative; it does not prove physical structure.",
        "- PCA is dimensionality reduction, not classification of origin.",
        "- Autoencoder metrics are model-dependent and trained against available controls.",
        "- No origin claim is made.",
        "",
        "## Recommended Next Steps",
        "",
        "- Review the full track visually end to end.",
        "- Run the same pipeline on more cases.",
        "- Compare against mundane controls with similar sensor/compression conditions.",
        "- Improve controls per sensor type and HUD layout.",
        "- Export content pack only with public-safe metrics.",
        "- Seek external human review if publishing technical conclusions.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path.name}`")
    report = outputs["unified_case_report.md"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _summary_json(case_id: str, data: dict[str, Any], status: dict[str, str], overall: str) -> dict[str, Any]:
    tracking_quality = _tracking_quality(data)
    return {
        "case_id": case_id,
        "module_status": status,
        "key_metrics": {
            "valid_frames": data["motion"].get("valid_tracked_frames"),
            "tracking_lost_frames": tracking_quality.get("lost_frames"),
            "motion_lost_frames": data["motion"].get("lost_frames"),
            "low_confidence_frames": tracking_quality.get("low_confidence_frames"),
            "predicted_only_frames": tracking_quality.get("predicted_only_frames"),
            "motion_continuity_score": data["motion"].get("motion_continuity_score"),
            "thermal_contrast_index": data["thermal"].get("thermal_contrast_index"),
            "control_validity_score": data["controls"].get("control_validity_score"),
            "pca_public_safe_score": data["pca"].get("pca_public_safe_score"),
            "autoencoder_public_safe_score": data["autoencoder"].get("anomaly_score_public_safe"),
        },
        "public_safe_metrics": {
            "tracking": status["tracking"],
            "controls": status["controls"],
            "motion": status["motion"],
            "spectral": status["spectral"],
            "thermal_ir": status["thermal_ir"],
            "pca": status["pca"],
            "autoencoder_public_safe_zscore": data["autoencoder"].get("public_safe_zscore"),
        },
        "exploratory_metrics": {
            "autoencoder_exploratory_max_zscore": data["autoencoder"].get("exploratory_max_zscore"),
            "autoencoder_exploratory_score": data["autoencoder"].get("anomaly_score_exploratory"),
        },
        "limitations": [
            "No origin claim.",
            "Relative IR intensity only; no radiometric temperature units available.",
            "Results depend on human-validated tracking.",
            "Compression, auto-gain, ROI/crop choices and source chain limit interpretation.",
            "PCA and autoencoder metrics are model/preprocessing dependent.",
        ],
        "recommended_next_steps": [
            "Review full track visually.",
            "Run same pipeline on more cases and mundane controls.",
            "Improve controls per sensor/HUD layout.",
            "Use public-safe metrics only for content.",
            "Seek external review before publication.",
        ],
        "overall_public_safe_assessment": overall,
    }


def _metrics_card(summary: dict[str, Any]) -> dict[str, Any]:
    status = summary["module_status"]
    strongest = [key for key, value in status.items() if value == "strong"]
    weakest = [key for key, value in status.items() if value in {"weak", "missing"}]
    return {
        "headline_metrics": summary["key_metrics"],
        "public_safe_score_summary": summary["overall_public_safe_assessment"],
        "strongest_modules": strongest,
        "weakest_modules": weakest,
        "caution_text": "Technical visual/statistical assessment only. No origin or non-human provenance claim.",
    }


def _manifest(path: Path, outputs: dict[str, Path]) -> None:
    classes = {
        "unified_case_report.md": "public_safe/technical",
        "unified_case_summary.json": "technical",
        "unified_metrics_card.json": "content_ready",
        "unified_case_scorecard.png": "content_ready/technical",
        "unified_report_manifest.md": "technical",
    }
    lines = ["# Unified Report Manifest", "", "| output | file | classification | caution |", "| --- | --- | --- | --- |"]
    for name, out_path in outputs.items():
        lines.append(f"| `{name}` | `{out_path.name}` | `{classes.get(name, 'technical')}` | no origin claim |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_status(case_id: str, outputs: dict[str, Path], overall: str) -> Path:
    path = case_status_path(case_id)
    status = _safe_json(path) or {"case_id": case_id}
    status.update(
        {
            "unified_report_status": "complete",
            "unified_report_ready": True,
            "last_unified_report_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "overall_public_safe_assessment": overall,
            "unified_report_paths": {
                "report": str(outputs["unified_case_report.md"]),
                "summary": str(outputs["unified_case_summary.json"]),
                "metrics_card": str(outputs["unified_metrics_card.json"]),
                "scorecard": str(outputs["unified_case_scorecard.png"]),
                "manifest": str(outputs["unified_report_manifest.md"]),
            },
        }
    )
    write_json(path, status)
    return path


def generate_unified_case_report(case_id: str) -> dict[str, Any]:
    data = _load_case(case_id)
    paths = _paths(case_id)
    status = _module_status(data, paths)
    overall = _overall(status)
    out = output_dir(case_id)
    outputs = {
        "unified_case_report.md": out / "unified_case_report.md",
        "unified_case_summary.json": out / "unified_case_summary.json",
        "unified_metrics_card.json": out / "unified_metrics_card.json",
        "unified_case_scorecard.png": out / "unified_case_scorecard.png",
        "unified_report_manifest.md": out / "unified_report_manifest.md",
    }
    _write_markdown(case_id, data, status, outputs, overall)
    summary = _summary_json(case_id, data, status, overall)
    write_json(outputs["unified_case_summary.json"], summary)
    write_json(outputs["unified_metrics_card.json"], _metrics_card(summary))
    _scorecard(outputs["unified_case_scorecard.png"], status)
    _manifest(outputs["unified_report_manifest.md"], outputs)
    status_path = _update_status(case_id, outputs, overall)
    return {
        "case_id": case_id,
        "unified_report_ready": True,
        "overall_public_safe_assessment": overall,
        "module_status": status,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "case_status": str(status_path),
    }
