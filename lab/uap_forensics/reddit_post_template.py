from __future__ import annotations

import time
import re
from pathlib import Path, PureWindowsPath
from typing import Any

from .io import read_json, write_json
from .paths import DATA_DIR, case_report_dir, ensure_dir


REQUIRED_MESSAGE = "Reddit Post Template requires a unified report first."
PUBLIC_SOURCE_URL = "not provided"
SOURCE_CHAIN = "not independently verified yet"
ORIGINAL_RELEASE_CONTEXT = "not provided"
PLACEHOLDER_PATTERNS = [
    r"\[EDIT",
    r"\[EDITAR",
    r"<SOURCE",
    r"C:\\Users\\",
    r"\\runtime\\",
    r"C:\\Workspaces",
    r"resolution:\s*`\{\}`",
    r"filename:\s*`unknown`",
    r"duration:\s*`not available",
    r"FPS:\s*`not available",
    r"source path",
]


def output_dir(case_id: str) -> Path:
    return ensure_dir(case_report_dir(case_id) / "reddit_template")


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path) if path.exists() else {}
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available in source metadata"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _public_text(value: Any, fallback: str = "not provided") -> str:
    if value is None or value == "" or value == {}:
        return fallback
    if isinstance(value, str) and value.lower() in {"unknown", "not available", "<source_video_path>"}:
        return fallback
    return str(value)


def _duration(value: Any) -> str:
    if value is None or value == "" or value == "unknown":
        return "not available in source metadata"
    try:
        return f"{float(value):.3f} seconds"
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
    if isinstance(value, str) and value and value != "unknown":
        return value
    return "not available in source metadata"


def _case_slug(case_id: str, src: dict[str, Any]) -> str:
    raw = " ".join([case_id, str(src.get("original_filename", ""))])
    match = re.search(r"(?:DOD|dod)[_\- ]?(\d+)", raw)
    if match:
        return f"DOD_{match.group(1)}"
    return case_id


def _title(case_id: str, src: dict[str, Any]) -> str:
    return f"Human-validated UAP/OVNI video analysis for {_case_slug(case_id, src)}"


def _safe_case_info(case_id: str, src: dict[str, Any], thermal: dict[str, Any]) -> str:
    source_quality = src.get("source_type") or src.get("source_quality") or "user_imported_unverified"
    return "\n".join(
        [
            f"- case_id: `{case_id}`",
            f"- filename: `{_public_text(src.get('original_filename'), 'not available in source metadata')}`",
            f"- duration: `{_duration(src.get('duration_seconds'))}`",
            f"- FPS: `{_fps(src.get('fps'))}`",
            f"- resolution: `{_resolution(src.get('resolution'))}`",
            f"- codec: `{_public_text(src.get('codec'), 'not available in source metadata')}`",
            f"- source quality: `{_public_text(source_quality, 'not independently verified')}`",
            f"- IR mode: `{_public_text(thermal.get('source_ir_mode'), 'not available in source metadata')}`",
            f"- public source URL: `{PUBLIC_SOURCE_URL}`",
            f"- source chain / provenance notes: `{SOURCE_CHAIN}`",
            f"- original release context: `{ORIGINAL_RELEASE_CONTEXT}`",
        ]
    )


def _validate_public_text(label: str, text: str) -> None:
    failures = [pattern for pattern in PLACEHOLDER_PATTERNS if re.search(pattern, text, re.IGNORECASE)]
    if failures:
        raise RuntimeError(f"Public report validation failed for {label}: forbidden placeholder/private pattern(s): {', '.join(failures)}")


def _sanitize_public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_public_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_public_payload(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("/", "\\")
        if "C:\\Users\\" in normalized or "C:\\Workspaces\\" in normalized or "\\runtime\\" in normalized:
            name = PureWindowsPath(value).name
            return name or "redacted for public report"
    return value


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


def _public_case_info_payload(source: dict[str, Any]) -> dict[str, Any]:
    blocked = {"video_path", "source_video", "tracking_status", "track_based_analysis_ready"}
    return {key: value for key, value in source.items() if key not in blocked}


def _public_tracking_quality(data: dict[str, Any]) -> dict[str, Any]:
    quality = dict(_tracking_quality(data))
    if "lost_frames" in quality:
        quality["tracking_lost_frames"] = quality.pop("lost_frames")
    if "low_confidence_frames" in quality:
        quality["tracking_low_confidence_frames"] = quality.pop("low_confidence_frames")
    if "predicted_only_frames" in quality:
        quality["tracking_predicted_only_frames"] = quality.pop("predicted_only_frames")
    return quality


def _public_module_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics = {key: dict(data[key]) for key in ["motion", "spectral", "thermal", "srv", "srv_core", "controls", "pca", "autoencoder"]}
    if "lost_frames" in metrics["motion"]:
        metrics["motion"]["motion_lost_frames"] = metrics["motion"].pop("lost_frames")
    metrics["tracking_quality"] = _public_tracking_quality(data)
    return metrics


def _validate_consistency(case_id: str, data: dict[str, Any], payload: dict[str, Any], post_texts: dict[str, str]) -> None:
    workflow = data["summary"].get("module_status", {})
    key_metrics = data["summary"].get("key_metrics", {})
    case_info = payload.get("case_info", {})
    if case_info.get("tracking_status") == "not_started" and workflow.get("tracking") == "strong":
        raise RuntimeError(f"Public report consistency failed for {case_id}: import tracking_status contradicts workflow_status.tracking.")
    if case_info.get("track_based_analysis_ready") is False and workflow.get("dynamic_roi") == "ready":
        raise RuntimeError(f"Public report consistency failed for {case_id}: import track_based_analysis_ready contradicts workflow_status.dynamic_roi.")
    tracking_lost = _tracking_quality(data).get("lost_frames")
    if tracking_lost is not None:
        expected = f"tracking_lost_frames: `{tracking_lost}`"
        for label, text in post_texts.items():
            if expected not in text:
                raise RuntimeError(f"Public report consistency failed for {label}: missing {expected}.")
        unified_report = (case_report_dir(case_id) / "unified_report" / "unified_case_report.md").read_text(encoding="utf-8")
        if expected not in unified_report:
            raise RuntimeError(f"Public report consistency failed for {case_id}: unified report missing {expected}.")
    if "lost_frames" in key_metrics:
        raise RuntimeError(f"Public report consistency failed for {case_id}: ambiguous key_metrics.lost_frames is not allowed; use tracking_lost_frames or motion_lost_frames.")
    if key_metrics.get("tracking_lost_frames") != tracking_lost:
        raise RuntimeError(f"Public report consistency failed for {case_id}: summary tracking_lost_frames differs from tracking_quality.")
    if "lost_frames" in payload.get("module_metrics", {}).get("motion", {}):
        raise RuntimeError(f"Public report consistency failed for {case_id}: ambiguous module_metrics.motion.lost_frames is not allowed.")
    if "lost_frames" in payload.get("module_metrics", {}).get("tracking_quality", {}):
        raise RuntimeError(f"Public report consistency failed for {case_id}: ambiguous module_metrics.tracking_quality.lost_frames is not allowed.")
    payload_tracking_lost = payload.get("module_metrics", {}).get("tracking_quality", {}).get("tracking_lost_frames")
    if tracking_lost != payload_tracking_lost:
        raise RuntimeError(f"Public report consistency failed for {case_id}: payload tracking_lost_frames differs from tracking_quality.")


def _load(case_id: str) -> dict[str, Any]:
    unified = case_report_dir(case_id) / "unified_report"
    required = [
        unified / "unified_case_report.md",
        unified / "unified_case_summary.json",
        unified / "unified_metrics_card.json",
        unified / "unified_case_scorecard.png",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError(REQUIRED_MESSAGE)
    return {
        "summary": _safe_json(unified / "unified_case_summary.json"),
        "metrics_card": _safe_json(unified / "unified_metrics_card.json"),
        "source": {
            **_safe_json(DATA_DIR / "cases" / case_id / "batch_source.json"),
            **_safe_json(DATA_DIR / "cases" / case_id / "case_metadata.json"),
        },
        "track": _safe_json(DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track.json"),
        "tracking_quality": _safe_json(DATA_DIR / "outputs" / case_id / "interactive_tracking" / "tracking_quality.json"),
        "validation": _safe_json(DATA_DIR / "cases" / case_id / "track_validation.json"),
        "motion": _safe_json(DATA_DIR / "outputs" / case_id / "motion_analysis" / "motion_metrics.json"),
        "spectral": _safe_json(DATA_DIR / "outputs" / case_id / "spectral_analysis" / "spectral_metrics.json"),
        "thermal": _safe_json(DATA_DIR / "outputs" / case_id / "thermal_analysis" / "thermal_metrics.json"),
        "srv": _safe_json(DATA_DIR / "outputs" / case_id / "srv_analysis" / "srv_metrics.json"),
        "srv_core": _safe_json(DATA_DIR / "outputs" / case_id / "srv_analysis" / "object_core" / "srv_core_metrics.json"),
        "controls": _safe_json(DATA_DIR / "outputs" / case_id / "controls_analysis" / "controls_metrics.json"),
        "pca": _safe_json(DATA_DIR / "outputs" / case_id / "pca_analysis" / "pca_metrics.json"),
        "autoencoder": _safe_json(DATA_DIR / "outputs" / case_id / "autoencoder_analysis" / "autoencoder_metrics.json"),
    }


def _workflow_rows(status: dict[str, str]) -> list[tuple[str, str, str]]:
    return [
        ("Human-marked object tracking", status.get("tracking", "missing"), "Object selected by human prompt; downstream modules use this target."),
        ("Track validation", status.get("human_validation", "missing"), "Human validation file confirms whether track is usable."),
        ("Dynamic ROI reconstruction", status.get("dynamic_roi", "missing"), "Frame-by-frame ROIs reconstructed from the validated track."),
        ("Motion / Optical Flow", status.get("motion", "missing"), "Pixel-space motion and optical flow inside dynamic ROI."),
        ("Spectral Analysis", status.get("spectral", "missing"), "Luminance, temporal frequency and spatial-frequency metrics."),
        ("Thermal / IR relative intensity", status.get("thermal_ir", "missing"), "Relative FLIR/IR intensity only; no radiometric temperature."),
        ("SRV / Visual Reconstruction", status.get("srv", "missing"), "Non-generative visual reconstruction; interpretive only."),
        ("Controls v0.2 clean masked", status.get("controls", "missing"), "Clean background/compression/random controls; HUD isolated."),
        ("PCA baseline", status.get("pca", "missing"), "Dimensionality-reduction baseline vs clean controls."),
        ("Autoencoder baseline", status.get("autoencoder", "missing"), "Controls-trained reconstruction-error baseline."),
        ("Unified Case Report", "ready", "Aggregated technical report; no origin claim."),
    ]


def _table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Step | Status | Note |", "| --- | --- | --- |"]
    lines.extend(f"| {a} | `{b}` | {c} |" for a, b, c in rows)
    return "\n".join(lines)


def _metrics_md(data: dict[str, Any]) -> str:
    m = data["motion"]; s = data["spectral"]; t = data["thermal"]; c = data["controls"]; p = data["pca"]; a = data["autoencoder"]; srv = data["srv"]; core = data["srv_core"]
    tq = _tracking_quality(data)
    return f"""
### Tracking

- tracking_valid_frames: `{tq.get('tracked_frames') or m.get('valid_tracked_frames', 'not available')}`
- tracking_lost_frames: `{tq.get('lost_frames', 'not available')}`
- tracking_low_confidence_frames: `{tq.get('low_confidence_frames', 'not available')}`
- tracking_predicted_only_frames: `{tq.get('predicted_only_frames', 'not available')}`
- continuity / validation: `{_fmt(m.get('motion_continuity_score'))}` / `{data['validation'].get('track_validated', False)}`

### Motion

- motion_lost_frames: `{m.get('lost_frames', 0)}`
- mean velocity: `{_fmt(m.get('mean_velocity_px_frame'))}` px/frame
- max velocity: `{_fmt(m.get('max_velocity_px_frame'))}` px/frame
- mean acceleration: `{_fmt(m.get('mean_acceleration_px_frame2'))}` px/frame^2
- optical flow mean/max: `{_fmt(m.get('mean_optical_flow_magnitude_inside_roi'))}` / `{_fmt(m.get('max_optical_flow_magnitude_inside_roi'))}`
- stability: `{_fmt(m.get('track_stability_score'))}`

### Spectral

- mean luminance: `{_fmt(s.get('mean_luminance'))}`
- flicker index: `{_fmt(s.get('luminance_flicker_index'))}`
- dominant temporal frequency: `{_fmt(s.get('dominant_temporal_frequency_hz'))}` Hz
- high-frequency ratio: `{_fmt(s.get('high_frequency_energy_ratio'))}`
- noise/compression proxy: `{_fmt(s.get('compression_noise_proxy'))}`

### Thermal / IR

- source_ir_mode: `{t.get('source_ir_mode', 'unknown')}`
- calibration_available: `{t.get('calibration_available', False)}`
- temperature units: `not available`
- mean ROI intensity: `{_fmt(t.get('mean_roi_intensity'))}`
- mean background intensity: `{_fmt(t.get('mean_background_intensity'))}`
- ROI-background delta: `{_fmt(t.get('roi_background_delta_mean'))}`
- thermal contrast index: `{_fmt(t.get('thermal_contrast_index'))}`

### Controls

- controls version: `{c.get('controls_version', 'not available')}`
- control_validity_score: `{_fmt(c.get('control_validity_score'))}`
- artifact contamination: `{_fmt(c.get('artifact_contamination_rate'))}`
- HUD leakage: `{_fmt(c.get('hud_leakage_rate'))}`
- accepted/rejected candidates: `{c.get('accepted_control_candidates', 'not available')}` / `{c.get('rejected_control_candidates', 'not available')}`
- object vs background delta: `{_fmt(c.get('object_vs_background_delta_ir_intensity'))}`

### PCA

- object/control samples: `{p.get('total_object_samples', 'not available')}` / `{p.get('total_control_samples', 'not available')}`
- PC1 / PC2: `{_fmt(p.get('pca_pc1_explained_variance'))}` / `{_fmt(p.get('pca_pc2_explained_variance'))}`
- k5 / k10: `{_fmt(p.get('pca_k5_explained_variance'))}` / `{_fmt(p.get('pca_k10_explained_variance'))}`
- object vs background distance: `{_fmt(p.get('object_vs_background_distance'))}`
- object vs HUD distance: `{_fmt(p.get('object_vs_hud_distance'))}`
- public-safe score: `{_fmt(p.get('pca_public_safe_score'))}`
- caution: silhouette is `{_fmt(p.get('silhouette_score'))}`, so PCA should be read as auxiliary.

### Autoencoder

- training strategy: `{a.get('training_strategy', 'not available')}`
- public-safe z-score: `{_fmt(a.get('public_safe_zscore'))}`
- object percentile vs controls: `{_fmt(a.get('object_error_percentile_vs_controls'))}`
- anomaly public-safe score: `{_fmt(a.get('anomaly_score_public_safe'))}`
- exploratory max z-score: `{_fmt(a.get('exploratory_max_zscore'))}` **exploratory only, not a headline**

### SRV

- bbox context crop count: `{srv.get('crop_count', 'not available')}`
- object-core confidence: `{data['summary'].get('module_status', {}).get('srv', 'not available')}`
- low confidence frames: `{core.get('low_confidence_frames', 'not available')}`
- no generative model used: `true`
""".strip()


def _interpretation(data: dict[str, Any], lang: str = "en") -> str:
    st = data["summary"].get("module_status", {})
    if lang == "es":
        labels = {
            "Tracking": "Mide si el objeto marcado manualmente se sigue de forma consistente.",
            "Motion": "Mide movimiento en pixeles/frame y optical flow dentro de la ROI.",
            "Spectral": "Mide luminancia, flicker y frecuencias; depende de compresión/sensor.",
            "Thermal / IR": "Mide intensidad IR relativa; no hay temperatura radiométrica.",
            "Controls": "Compara contra controles limpios v0.2 y artefactos aislados.",
            "PCA": "Reduce dimensionalidad para comparar objeto frente a controles.",
            "Autoencoder": "Mide error de reconstrucción bajo un modelo entrenado con controles.",
            "SRV": "Reconstrucción visual no generativa e interpretativa.",
        }
        return "\n\n".join(f"### {k}\n\n- Qué mide: {v}\n- Resultado: `{st.get(k.lower().replace(' / ', '_').replace(' ', '_'), st.get('thermal_ir' if k=='Thermal / IR' else 'srv' if k=='SRV' else k.lower(), 'missing'))}`\n- Cautela: no determina origen ni naturaleza física." for k, v in labels.items())
    labels = {
        "Tracking": ("tracking", "Tracks the human-marked object through the video."),
        "Motion": ("motion", "Measures pixel-space velocity, acceleration and optical flow."),
        "Spectral": ("spectral", "Measures luminance, flicker and frequency content."),
        "Thermal / IR": ("thermal_ir", "Measures relative IR intensity only."),
        "Controls": ("controls", "Compares the object against clean masked controls."),
        "PCA": ("pca", "Compares object/control structure in reduced dimensions."),
        "Autoencoder": ("autoencoder", "Measures reconstruction mismatch under a controls-trained model."),
        "SRV": ("srv", "Non-generative visual reconstruction for review."),
    }
    return "\n\n".join(f"### {title}\n\n- What was measured: {desc}\n- Result: `{st.get(key, 'missing')}`\n- Caution: this does not determine origin, identity, or physical nature." for title, (key, desc) in labels.items())


def _attachments(case_id: str) -> list[str]:
    candidates = [
        (DATA_DIR / "outputs" / case_id / "interactive_tracking" / "track_overlay_preview_web.mp4", "track_overlay_preview_web.mp4"),
        (DATA_DIR / "reports" / case_id / "unified_report" / "unified_case_scorecard.png", "unified_case_scorecard.png"),
        (DATA_DIR / "outputs" / case_id / "motion_analysis" / "motion_trajectory_panel.png", "motion_trajectory_panel.png"),
        (DATA_DIR / "outputs" / case_id / "spectral_analysis" / "spectral_fft_panel.png", "spectral_fft_panel.png"),
        (DATA_DIR / "outputs" / case_id / "thermal_analysis" / "thermal_intensity_panel.png", "thermal_intensity_panel.png"),
        (DATA_DIR / "outputs" / case_id / "controls_analysis" / "controls_summary_panel.png", "controls_summary_panel.png"),
        (DATA_DIR / "outputs" / case_id / "pca_analysis" / "pca_scatter_panel.png", "pca_scatter_panel.png"),
        (DATA_DIR / "outputs" / case_id / "autoencoder_analysis" / "autoencoder_summary_panel.png", "autoencoder_summary_panel.png"),
    ]
    return [name for path, name in candidates if path.exists()]


def _post_en(case_id: str, data: dict[str, Any]) -> str:
    src = data["source"]; summary = data["summary"]; status = summary.get("module_status", {})
    workflow = _table(_workflow_rows(status))
    title = _title(case_id, src)
    return f"""# {title}

## Public summary

This is a human-validated technical analysis of a UAP/OVNI video using tracking, dynamic ROI reconstruction, clean controls, PCA and autoencoder baselines.

It is **technically anomalous under the tested workflow; no origin claim.**

## Short version

- This is not an origin claim.
- The object was tracked using human-validated tracking.
- Dynamic ROIs were reconstructed from that track.
- The case was compared against Controls v0.2 clean masked baselines.
- Modules executed include motion/optical flow, spectral, relative FLIR/IR, SRV, PCA and autoencoder baselines.
- Overall assessment: `{summary.get('overall_public_safe_assessment')}`
- Best public-safe phrasing: `Technically anomalous under the tested workflow; no origin claim.`

## What this is

This is a reproducible technical workflow, not a claim of origin.

## Source / case info

{_safe_case_info(case_id, src, data['thermal'])}

## Workflow

{workflow}

## Key metrics

{_metrics_md(data)}

## Module-by-module interpretation

{_interpretation(data, 'en')}

## Current assessment

`{summary.get('overall_public_safe_assessment')}`

This means the case is technically analyzable and some modules show visual/statistical differences from controls, but the workflow does not determine origin, identity, or physical nature.

## Limitations

- FLIR/IR is relative only; no radiometric temperature is available.
- Compression, codec processing and auto-gain may affect the imagery.
- Source chain limitations remain unless provenance is independently verified.
- All metrics depend on tracking validity.
- Controls affect the baseline comparison.
- PCA is dimensionality reduction.
- Autoencoder is model-dependent.
- SRV is interpretive and non-generative.
- No origin claim.

## What I am looking for from the community

- Better source/provenance information.
- Mundane control comparisons.
- Sensor/context information.
- Critique of tracking validity.
- Critique of controls.
- Suggestions for vector/pulsation correlation.
- Phase segmentation ideas.
- Reflectance/illumination relative analysis.

## Suggested attachments

{chr(10).join(f'- `{item}`' for item in _attachments(case_id)) or '- Generated panels are available in the case report outputs.'}

## Disclosure

Analysis generated with AndoFrontier Lab, a source-available UAP forensic workflow under a non-commercial/source-available license.
"""


def _post_es(case_id: str, data: dict[str, Any]) -> str:
    src = data["source"]; summary = data["summary"]
    return f"""# {_title(case_id, src)}

## Resumen público

Este es un análisis técnico validado por humano de un video UAP/OVNI usando tracking, ROI dinámica, controles limpios, PCA y autoencoder.

Es **técnicamente anómalo bajo el workflow probado; sin afirmación de origen.**

## Versión corta

- Esto no es una afirmación de origen.
- El objeto fue seguido mediante tracking validado por humano.
- Se reconstruyó una ROI dinámica frame a frame.
- Se comparó contra controles limpios v0.2.
- Se ejecutaron módulos de movimiento, espectral, FLIR/IR relativo, SRV, PCA y autoencoder.
- Assessment general: `{summary.get('overall_public_safe_assessment')}`
- Frase pública recomendada: `Técnicamente anómalo bajo el workflow probado; sin afirmación de origen.`

## Qué es esto

Es un workflow técnico reproducible, no una afirmación sobre el origen del objeto.

## Fuente / caso

{_safe_case_info(case_id, src, data['thermal'])}

## Workflow

{_table(_workflow_rows(summary.get('module_status', {})))}

## Métricas clave

{_metrics_md(data)}

## Interpretación por módulo

{_interpretation(data, 'es')}

## Assessment actual

`{summary.get('overall_public_safe_assessment')}`

Esto significa que el caso es técnicamente analizable y algunos módulos muestran diferencias visuales/estadísticas frente a controles, pero el workflow no determina origen, identidad ni naturaleza física.

## Limitaciones

- FLIR/IR es relativo; no hay temperatura radiométrica.
- Compresión, codec y auto-gain pueden afectar la imagen.
- La cadena de fuente sigue limitada si no se verifica externamente.
- Todas las métricas dependen del tracking.
- Los controles condicionan la comparación.
- PCA es reducción dimensional.
- Autoencoder depende del modelo.
- SRV es interpretativo y no generativo.
- No hay claim de origen.

## Qué busco de la comunidad

- Mejor fuente/procedencia.
- Controles mundanos comparables.
- Información de sensor/contexto.
- Crítica del tracking.
- Crítica de controles.
- Ideas para correlación vector/pulsación.
- Segmentación por fases.
- Análisis relativo de reflectancia/iluminación.

## Adjuntos sugeridos

{chr(10).join(f'- `{item}`' for item in _attachments(case_id)) or '- Los paneles generados están disponibles en los outputs del reporte del caso.'}

## Disclosure

Análisis generado con AndoFrontier Lab, un workflow forense UAP source-available bajo licencia no comercial/source-available.
"""


def generate_reddit_post_template(case_id: str) -> dict[str, Any]:
    data = _load(case_id)
    out = output_dir(case_id)
    post_en = out / "reddit_post_template_en.md"
    post_es = out / "reddit_post_template_es.md"
    short_en = out / "reddit_post_short_en.md"
    titles = out / "reddit_title_options_en.txt"
    data_json = out / "reddit_post_data.json"
    manifest = out / "reddit_template_manifest.md"
    post_en_text = _post_en(case_id, data)
    post_es_text = _post_es(case_id, data)
    short_en_text = f"""# {_title(case_id, data['source'])}

This is not an origin claim. I am sharing a reproducible technical workflow: human-validated tracking, dynamic ROI reconstruction, clean controls v0.2, motion/spectral/relative IR/PCA/autoencoder baselines, and a unified case report.

Technically anomalous under the tested workflow; no origin claim.

Overall assessment: `{data['summary'].get('overall_public_safe_assessment')}`

Key public-safe points: tracking `{data['summary'].get('module_status', {}).get('tracking')}`, controls `{data['summary'].get('module_status', {}).get('controls')}`, motion `{data['summary'].get('module_status', {}).get('motion')}`, PCA `{data['summary'].get('module_status', {}).get('pca')}`, autoencoder `{data['summary'].get('module_status', {}).get('autoencoder')}`.

Tracking quality: tracking_lost_frames: `{_tracking_quality(data).get('lost_frames', 'not available')}`, tracking_low_confidence_frames: `{_tracking_quality(data).get('low_confidence_frames', 'not available')}`, tracking_predicted_only_frames: `{_tracking_quality(data).get('predicted_only_frames', 'not available')}`.

Limitations: relative FLIR/IR only, no radiometric temperature, compression/auto-gain may affect imagery, all metrics depend on tracking and controls, PCA is dimensionality reduction, autoencoder is model-dependent, and SRV is interpretive. I am looking for technical critique of the workflow, controls, tracking validity, and possible mundane comparisons.
"""
    for label, text in {"reddit_post_template_en.md": post_en_text, "reddit_post_template_es.md": post_es_text, "reddit_post_short_en.md": short_en_text}.items():
        _validate_public_text(label, text)
    post_en.write_text(post_en_text, encoding="utf-8")
    post_es.write_text(post_es_text, encoding="utf-8")
    short_en.write_text(short_en_text, encoding="utf-8")
    titles.write_text(
        "\n".join(
            [
                _title(case_id, data["source"]),
                f"Human-validated UAP tracking and dynamic ROI analysis: {_case_slug(case_id, data['source'])}",
                "Frame-by-frame UAP tracking with clean controls and PCA baseline",
                "A reproducible UAP forensic workflow: tracking, IR, controls, PCA, autoencoder",
                "Technical UAP case analysis with no origin claim",
                "Dynamic ROI UAP analysis using clean controls v0.2",
                "FLIR UAP technical workflow: motion, spectral, IR, PCA and autoencoder",
                "Human-validated tracking and public-safe UAP metrics",
                "Traceable UAP video analysis with controls and unified report",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _validate_public_text("reddit_title_options_en.txt", titles.read_text(encoding="utf-8"))
    safe_source = _public_case_info_payload(data["source"])
    payload = {
        "case_info": safe_source,
        "workflow_status": data["summary"].get("module_status", {}),
        "workflow_state": {
            "tracking_status": data["summary"].get("module_status", {}).get("tracking"),
            "track_based_analysis_ready": data["summary"].get("module_status", {}).get("dynamic_roi") == "ready",
        },
        "module_metrics": _public_module_metrics(data),
        "limitations": data["summary"].get("limitations", []),
        "recommended_attachments": _attachments(case_id),
        "public_source_url": PUBLIC_SOURCE_URL,
        "source_chain": SOURCE_CHAIN,
        "original_release_context": ORIGINAL_RELEASE_CONTEXT,
        "no_origin_claim": True,
    }
    payload = _sanitize_public_payload(payload)
    _validate_consistency(case_id, data, payload, {"reddit_post_template_en.md": post_en_text, "reddit_post_template_es.md": post_es_text, "reddit_post_short_en.md": short_en_text})
    _validate_public_text("reddit_post_data.json", str(payload))
    write_json(data_json, payload)
    manifest.write_text(
        "\n".join(
            [
                "# Reddit Template Manifest",
                "",
                "| file | classification | caution |",
                "| --- | --- | --- |",
                f"| `{post_en.name}` | public_template | ready for review; no origin claim |",
                f"| `{post_es.name}` | public_template | ready for review; no origin claim |",
                f"| `{short_en.name}` | public_template | short version; no origin claim |",
                f"| `{titles.name}` | title_options | non-clickbait only |",
                f"| `{data_json.name}` | technical | structured public data |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status_path = DATA_DIR / "cases" / case_id / "case_status.json"
    status = read_json(status_path) if status_path.exists() else {"case_id": case_id}
    status.update(
        {
            "reddit_template_status": "complete",
            "reddit_template_ready": True,
            "last_reddit_template_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reddit_template_paths": {
                "folder": str(out),
                "template_en": str(post_en),
                "template_es": str(post_es),
                "short_en": str(short_en),
                "title_options_en": str(titles),
                "data": str(data_json),
                "manifest": str(manifest),
            },
        }
    )
    write_json(status_path, status)
    return {"case_id": case_id, "reddit_template_ready": True, "output_dir": str(out), "outputs": status["reddit_template_paths"], "case_status": str(status_path)}
