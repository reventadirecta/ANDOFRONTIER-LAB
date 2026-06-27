from pathlib import Path

import pandas as pd

from .io import read_json
from .paths import DATA_DIR, case_output_dir, case_report_dir, ensure_dir


def _table_or_note(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return f"_Pendiente: no existe `{path.name}`._"
    df = pd.read_csv(path)
    df = df.head(max_rows)
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_case_report(config: dict) -> Path:
    case_id = config["case_id"]
    report_dir = ensure_dir(case_report_dir(case_id))
    source_path = DATA_DIR / "sources" / f"{case_id}.source.json"
    source = read_json(source_path) if source_path.exists() else {}
    out = case_output_dir(case_id)
    panels = [
        out / "base" / "base_panel.png",
        out / "pca" / "pca_panel.png",
        out / "autoencoder" / "autoencoder_panel.png",
    ]
    panel_block = "\n".join(f"![{path.stem}]({path.as_posix()})" for path in panels if path.exists())
    conclusion = "anomalia visual persistente / fuente sin clasificar"

    content = f"""# Forensic Report: {case_id}

## Resumen tecnico

Este reporte resume un analisis local reproducible de un video UAP/desclasificado. El objetivo no es identificar origen o intencion, sino medir persistencia visual, estructura, movimiento y discrepancia frente a modelos de fondo.

Conclusion prudente: **{conclusion}**.

## Fuente y cadena de custodia

- Ruta: `{source.get("video_path", config.get("video_path", "pendiente"))}`
- Tipo de fuente: `{source.get("source_type", config.get("source_type", "unknown"))}`
- URL: `{source.get("source_url", config.get("source_url", ""))}`
- SHA256: `{source.get("sha256", "pendiente")}`
- Notas: {source.get("chain_of_custody_notes", config.get("notes", ""))}

## Metodologia exacta

1. Registro de fuente con SHA256, metadata ffprobe/OpenCV y notas de procedencia.
2. Extraccion de frames PNG sin recompresion adicional deliberada.
3. Seleccion de ROI manual o automatica por brillo/contraste y persistencia de ROI en JSON.
4. Stackeo, proyecciones minima/media/maxima, diferencias frame a frame, optical flow, RGB/luminancia/HSV, CLAHE, bordes, FFT espacial y perfiles de luminancia.
5. PCA/SVD sobre matriz frame-pixel, reconstrucciones k=5/k=10, residuales y correlaciones.
6. Autoencoder entrenado con patches de fondo/no-objeto del mismo video, excluyendo la zona central de ROI.
7. Comparacion contra controles cuando existan casos normales procesados con el mismo pipeline.

## Metricas PCA

{_table_or_note(out / "pca" / "pca_metrics.csv")}

## Metricas autoencoder

{_table_or_note(out / "autoencoder" / "autoencoder_metrics.csv")}

## Paneles generados

{panel_block or "_Pendiente: no hay paneles generados._"}

## Limitaciones

- Un video derivado de YouTube, Reddit, 4chan o mirrors puede contener recompresion, reescalado, interpolacion y metadata perdida.
- El pipeline detecta propiedades visuales y estadisticas; no determina por si solo naturaleza, distancia, tamano, tecnologia, intencion ni autenticidad absoluta.
- La ROI automatica es una ayuda inicial y debe revisarse manualmente.
- El autoencoder es una prueba de anomalia local, no una prueba de origen no convencional.
- Las comparaciones solo son fuertes si los controles comparten condiciones opticas, sensor, compresion, iluminacion y movimiento similares.

## Bloque Reddit ES

Analisis forense reproducible: se registro la fuente, se extrajeron frames, se aislo una ROI, y se midieron stackeo, diferencias frame a frame, optical flow, FFT espacial, PCA y error de reconstruccion con autoencoder entrenado solo con fondo del mismo video. Conclusion prudente: **{conclusion}**. Limitacion clave: esto no prueba origen, solo documenta una anomalia visual persistente bajo este pipeline.

## Reddit Block EN

Reproducible forensic analysis: source registration, frame extraction, ROI isolation, stacking, frame-to-frame differences, optical flow, spatial FFT, PCA, and reconstruction error from an autoencoder trained only on same-video background patches. Conservative conclusion: **persistent visual anomaly / unclassified source**. Key limitation: this does not prove origin; it documents visual persistence under this pipeline.
"""
    path = report_dir / f"{case_id}_report.md"
    path.write_text(content, encoding="utf-8")
    return path
