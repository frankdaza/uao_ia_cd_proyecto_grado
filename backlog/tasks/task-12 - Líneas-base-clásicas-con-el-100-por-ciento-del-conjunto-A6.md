---
id: TASK-12
title: Líneas base clásicas con el 100 por ciento del conjunto (A6)
status: Done
assignee:
  - Frank Daza
created_date: '2026-08-17 01:10'
updated_date: '2026-08-18 01:56'
labels:
  - baseline
  - bitacora
milestone: m-2
dependencies:
  - TASK-8
references:
  - 'http://proceedings.mlr.press/v97/tan19a.html'
  - 'https://arxiv.org/abs/1512.03385'
  - 'https://arxiv.org/abs/2506.21937'
  - 'https://arxiv.org/abs/2401.15804'
documentation:
  - docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex
  - docs/trabajo_de_grado/Bitacora Metodologica de Hallazgos.tex
  - .cursor/skills/ejecutar-experimento/SKILL.md
priority: high
type: task
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Actividad del anteproyecto:** A6 — Entrenamiento de las líneas base clásicas.

**Qué.** Entrenar EfficientNet-B0 y ResNet-50 con cabeza clásica usando el `Trainer` de task-8 y **las mismas particiones** de `results/splits.json`, con el 100 % de los datos de entrenamiento de cada fold. Establece el límite superior de exactitud contra el cual se interpretará todo lo demás.

**Por qué.** Sin una línea base creíble, cualquier número del HQCNN es incomparable. Y "creíble" tiene un requisito preciso: **mismas particiones, mismo bucle, mismo presupuesto de épocas**. Si la línea base se entrenara con otro código o otros folds, la comparación mediría la diferencia entre dos implementaciones y no entre dos arquitecturas. Estas 10 corridas (2 modelos × 5 folds) son además las celdas al 100 % que la campaña de task-13 **reutiliza**, no repite.

**Entregable.** 10 corridas en `results/experiments.csv`, pesos en `models/`, historial por época en `results/history/` y la tabla de media ± desviación estándar por métrica.

**Posición en el diseño experimental.**

```mermaid
flowchart LR
  Splits["results/splits.json<br/>(task-6)"] --> B["Trainer (task-8)"]
  BB["Fabrica de backbones<br/>(task-7)"] --> B
  B --> E["EfficientNet-B0<br/>5 folds al 100%"]
  B --> R["ResNet-50<br/>5 folds al 100%"]
  E --> CSV["results/experiments.csv"]
  R --> CSV
  CSV --> Reuso["task-13 reutiliza<br/>estas 10 celdas"]
  CSV --> Comp["Comparacion con la literatura:<br/>haddou2025hqcm, khan2024brain"]
```

**Contraste con la literatura.** Los resultados deben compararse con los reportados en trabajos híbridos sobre el mismo dataset (`haddou2025hqcm`, `khan2024brain`) y con las arquitecturas originales (`tan2019efficientnet`, `he2016deep`). Si la línea base queda muy por debajo de lo publicado, hay un problema de implementación que debe resolverse **antes** de la campaña, no interpretarse como un hallazgo.

**Claves BibTeX.** `tan2019efficientnet`, `he2016deep`, `haddou2025hqcm`, `khan2024brain`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EfficientNet-B0 y ResNet-50 se entrenan con el Trainer de task-8 y las particiones de results/splits.json, sin código de entrenamiento propio
- [x] #2 Se completan las 10 corridas (2 arquitecturas x 5 folds) al 100 por ciento del entrenamiento y quedan escritas en results/experiments.csv
- [x] #3 Se reporta media y desviación estándar por métrica sobre los 5 folds para cada arquitectura
- [x] #4 Los índices efectivamente cargados se verifican contra results/splits.json para garantizar que el HQCNN verá los mismos folds
- [x] #5 Los resultados se contrastan con la literatura sobre el mismo dataset y se explica por qué los números no son directamente comparables cuando corresponda
- [x] #6 Las 10 celdas quedan reutilizables por task-13 con la clave (modelo, fracción 1.00, fold) correctamente registrada
- [x] #7 El tiempo de inferencia se mide con el mismo protocolo que usará el HQCNN, para que la comparación de costo sea válida
- [x] #8 Hallazgo registrado en hallazgos/h2_experimentacion.tex con \label{hallazgo:task-12}, tabla de resultados y comparación con la literatura
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Pesos persistidos con state_dict() en models/ e historial por época en results/history/
- [x] #2 Semilla, dispositivo y versión de pesos registrados en cada fila del CSV
- [x] #3 La limitación del extractor congelado queda declarada para que la tabla no se lea como el máximo alcanzable por la arquitectura
- [x] #4 Sin rutas absolutas ni dependencias de Google Drive en el script de experimentos
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Crear ClassicalBaseline (src/models/baseline.py) y build_model (src/models/factory.py) como prerequisito compartido con TASK-13.
2. Implementar src/experiments/baselines.py: bucle sobre (efficientnet_b0, resnet50) x 5 folds al 100%, delegando a Trainer(..., fold=k), construir_loaders_para_fold, escribir_corrida_csv y escribir_historial_json.
3. Reanudabilidad via corrida_completada(); version de pesos en logs y hallazgo (VERSIONES_PESOS), no en COLUMNAS_CSV.
4. Funciones auxiliares: verificar_indices_fold, consolidar_metricas, guardar_resumen_csv.
5. Pruebas en tests/test_baseline.py; sonda --sonda (1 epoca, 1 fold) y campana completa.
6. Hallazgo en hallazgos/h2_experimentacion.tex con label hallazgo:task-12.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Trampas conocidas.**

- La comparación con la literatura tiene una asimetría que hay que declarar: la mayoría de los trabajos publicados sobre este dataset entrena **todo** el backbone o usa el *split* original `Training`/`Testing`, no validación cruzada con extractor congelado. Reportar una exactitud algo menor no es un fallo, pero **sí** hay que explicar por qué los números no son directamente comparables.
- Una desviación estándar muy alta entre folds es señal de que algo depende del muestreo más de lo esperado (tasa de aprendizaje, presupuesto de épocas, o un fold con distribución atípica). Investigar antes de la campaña: con 60 celdas el problema se multiplica.
- Con backbone congelado, la línea base tiene muy pocos parámetros entrenables y puede quedar por debajo de su potencial. Es la condición del experimento (misma restricción para todos), no un error, y debe quedar escrito para que nadie lea la tabla como el máximo alcanzable por EfficientNet-B0.
- No reentrenar las 10 celdas en task-13: la reanudabilidad del `Trainer` debe reconocerlas por su clave. Duplicarlas rompería el balance del diseño factorial y metería dos observaciones de la misma celda en la ANOVA.
- ResNet-50 con `IMAGENET1K_V2` no es intercambiable con `V1` a efectos de comparación bibliográfica; registrar la versión usada.
- El tiempo de inferencia de las líneas base es la referencia contra la cual se leerá el costo del HQCNN. Medirlo con el mismo protocolo (calentamiento y sincronización) o la comparación de costo será engañosa.

Validacion: 19 passed (test_baseline + test_backbones). 10 filas unicas en experiments.csv (deduplicadas). Sonda MPS ~45s/ep EfficientNet; campana completa ~1.7h en MPS. Acc val media: EfficientNet 93.69%, ResNet 92.03%.

**Addenda (TASK-20 / decisión D3, campaña en Colab Pro+).** Las 10 corridas al 100 % producidas por esta tarea se ejecutaron en `mps`. Como la campaña de task-13 pasa íntegramente a CUDA, esas filas se archivan en `results/historico_mps.csv` (historiales en `results/history_mps/`) y las celdas se reejecutan en GPU para que las 60 del diseño factorial compartan hardware. Sin esa homogeneidad, el tiempo de entrenamiento y la latencia de inferencia que exige A9 quedarían confundidos con el dispositivo justo en la fracción del 100 %.

El protocolo y los criterios de A6 no cambian y la tarea sigue cerrada: lo que se repite es la medición, no el diseño. La evidencia MPS no se elimina y queda referenciada en la bitácora.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementadas ClassicalBaseline, build_model y src/experiments/baselines.py. Diez corridas al 100% (5 folds x 2 modelos) en experiments.csv con pesos e historiales. Hallazgo hallazgo:task-12 en h2_experimentacion.tex. Verificado con pytest (19 passed) y conteo de 10 celdas unicas en CSV.
<!-- SECTION:FINAL_SUMMARY:END -->
