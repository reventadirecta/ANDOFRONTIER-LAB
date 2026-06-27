# Methodology Notes

## Principio

El laboratorio mide senales visuales reproducibles sin convertirlas en afirmaciones fuertes de origen. Cada salida debe poder regenerarse desde una fuente local, una config de caso y una version del pipeline.

## Flujo minimo replicable

1. Registrar fuente y SHA256.
2. Extraer frames sin recomprimir a formatos con perdida.
3. Definir ROI manual o automatica y persistirla en JSON.
4. Ejecutar analisis base y guardar metricas/paneles.
5. Ejecutar PCA sobre matriz frame-pixel.
6. Entrenar autoencoder con fondo/no-objeto del mismo video.
7. Repetir en controles comparables.
8. Reportar con lenguaje prudente.

## Controles recomendados

- CCTV nocturno.
- Insectos cerca de lente.
- Polvo IR.
- Luces desenfocadas.
- Reflejos de lente.
- Compresion fuerte.
- Videos low-light.

## Lenguaje de conclusion

Usar como conclusion por defecto: "anomalia visual persistente / fuente sin clasificar".
