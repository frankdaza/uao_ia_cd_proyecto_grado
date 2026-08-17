---
id: TASK-8
title: Bucle unificado de entrenamiento y evaluación (Trainer)
status: Done
assignee:
  - Frank Daza
created_date: '2026-08-17 01:06'
updated_date: '2026-08-17 02:36'
labels:
  - infra
  - bitacora
milestone: m-1
dependencies:
  - TASK-4
  - TASK-6
  - TASK-7
references:
  - 'https://www.nature.com/articles/s41591-018-0316-z'
  - 'https://doi.org/10.1016/j.media.2017.07.005'
documentation:
  - docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex
  - docs/trabajo_de_grado/Bitacora Metodologica de Hallazgos.tex
  - AGENTS.md
priority: high
type: feature
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Qué.** Un **único** bucle de entrenamiento y evaluación que sirve a las líneas base, a la ablación de profundidad y a la campaña experimental completa. Recibe un `nn.Module` ya construido y un `ExperimentConfig`, lee los índices de `results/splits.json`, emite el contrato de task-4 e instrumenta los tiempos con calentamiento y sincronización de dispositivo.

**Por qué.** Es la pieza que evita el fracaso más común de un benchmark: tres bucles distintos (uno por familia de experimento) producen resultados no comparables sin que nadie lo note. Basta que uno use `Adam` y otro `AdamW`, o que midan el tiempo de forma distinta, para que la diferencia entre HQCNN y línea base deje de ser atribuible a la arquitectura. Con un solo bucle, la comparación es una sustitución de modelo y nada más: es exactamente el principio de sustitución de Liskov aplicado al diseño experimental.

**Entregable.** `src/train/trainer.py` con la clase `Trainer`, reanudabilidad frente al CSV existente y persistencia de pesos con `state_dict()`.

**Flujo del entrenamiento.**

```mermaid
flowchart TB
  Cfg["ExperimentConfig<br/>(task-1)"] --> T["Trainer"]
  Splits["results/splits.json<br/>(task-6)"] --> T
  M["nn.Module — clasico o hibrido<br/>(task-7, task-10)"] --> T
  T --> Ep["Bucle de epocas<br/>presupuesto FIJO e identico"]
  Ep --> Hist["EpochRecord →<br/>results/history/*.json"]
  Ep --> Fin["Evaluacion final del fold"]
  Fin --> Rec["RunRecord →<br/>experiments.csv + wandb"]
  Fin --> SD["models/{modelo}_{frac}_f{fold}.pt<br/>state_dict unicamente"]
```

**Principios SOLID que esta tarea materializa.**

- **SRP:** el `Trainer` entrena y evalúa; no construye modelos, no arma particiones, no formatea resultados.
- **OCP:** un modelo nuevo entra por la fábrica; el bucle no se toca.
- **LSP:** el HQCNN es sustituible por una línea base clásica sin ninguna rama condicional.
- **DIP:** el `Trainer` depende de `nn.Module` y `ExperimentConfig`; **no importa** `pennylane` ni `torchvision`.
- **DRY:** un solo bucle de épocas, un solo criterio de selección de época, un solo protocolo de medición de tiempos.

**Riesgo metodológico que esta tarea cierra.** Elegir la época a reportar mirando el fold de validación y luego reportar ese mismo fold produce una estimación optimista (fuga por selección de época). El criterio debe ser un presupuesto de épocas fijo o una partición interna extraída del entrenamiento, aplicado **igual** a todos los modelos.

**Claves BibTeX.** `esteva2019guide`, `litjens2017survey`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 El mismo Trainer entrena el modelo clásico y el híbrido sin ninguna rama condicional por tipo de modelo
- [x] #2 El presupuesto de épocas, el optimizador, la función de pérdida y la tasa de aprendizaje son idénticos para todos los modelos y provienen de la configuración
- [x] #3 No se selecciona la época a reportar mirando el fold de validación que luego se reporta: el criterio elegido está documentado y se aplica igual a todos los modelos
- [x] #4 El Trainer no importa pennylane ni torchvision: depende solo de nn.Module y ExperimentConfig (DIP)
- [x] #5 Los índices de entrenamiento y validación se leen de results/splits.json y no se remuestrean
- [x] #6 Los tiempos de entrenamiento e inferencia se miden con calentamiento y sincronización de dispositivo según el protocolo de task-4
- [x] #7 Emite exactamente el contrato de task-4 (RunRecord y EpochRecord) sin campos ad hoc
- [x] #8 Los pesos se persisten con state_dict() exclusivamente y la recarga reproduce las métricas reportadas
- [x] #9 Las corridas son reanudables: una celda del diseño factorial ya presente en el CSV se omite en lugar de duplicarse
- [x] #10 Hallazgo breve en hallazgos/h1_arquitectura.tex con \label{hallazgo:task-8} que documenta el criterio de épocas y los hiperparámetros comunes
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tipado de Python 3.12 y docstrings NumPy en español latinoamericano
- [x] #2 Prueba de sustituibilidad: un modelo clásico y uno con capa cuántica entrenan con el mismo Trainer
- [x] #3 Sin torch.save del módulo completo en ninguna ruta del código
- [x] #4 Presupuesto de épocas y optimizador congelados y documentados antes de iniciar cualquier campaña
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Crear src/train/dataloading.py con construir_loaders_para_fold(): lee splits.json vía obtener_indices(), drop_last=True en train, sin remuestreo.
2. Crear src/train/trainer.py: Trainer(modelo, cfg, dispositivo, fold); Adam+CrossEntropyLoss solo sobre requires_grad; bucle fijo cfg.epocas; emite RunRecord/EpochRecord; reutiliza sincronizar_dispositivo y medir_inferencia_ms_por_lote; n_capas_vqc vía getattr(modelo, n_capas_vqc, None).
3. Reanudabilidad: corrida_completada() delega a corrida_existe con tupla (modelo, data_fraction, fold, semilla); pesos en models/{modelo}_{frac}_f{fold}_s{semilla}.pt con state_dict exclusivamente.
4. Crear tests/test_trainer.py: sustituibilidad clásico/cuántico simulado, reload state_dict, imports DIP, corrida_existe 4-tupla.
5. Documentar hallazgo en h1_arquitectura.tex: presupuesto fijo 15 épocas, Adam lr=1e-3, criterio última época.
6. Verificar con uv run pytest tests/test_trainer.py -q y regresión completa.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Trampas conocidas.**

- `self._modelo.train()` reactiva las capas de normalización por lotes del backbone congelado. El `Trainer` **no** debe conocer esa particularidad: la solución limpia es que el módulo híbrido sobrescriba `train()` para reimponer `eval()` en su backbone (task-10). Si se parchea desde el `Trainer`, se rompe DIP y aparece la rama condicional que se quería evitar.
- Solo los parámetros con `requires_grad=True` deben entrar al optimizador; con un backbone congelado, pasarle `modelo.parameters()` completo desperdicia memoria en momentos de Adam que nunca se usan.
- `CrossEntropyLoss` espera **logits**, no probabilidades. El VQC devuelve valores esperados en el intervalo [-1, 1]: son logits válidos pero de escala pequeña, lo que achata el softmax y limita la confianza alcanzable. El efecto es real y debe documentarse (se retoma en task-10); lo que **no** se debe hacer es aplicar softmax antes de la pérdida.
- Medir el tiempo sin `torch.cuda.synchronize()` o `torch.mps.synchronize()` mide el encolado asíncrono de kernels. En el HQCNN el costo dominante está en la simulación del circuito y el sesgo sería grande.
- No usar detención temprana sobre el fold de validación que después se reporta: es precisamente la fuga que la AC prohíbe. Si se quiere detención temprana, la señal debe venir de una partición interna extraída del entrenamiento.
- Guardar el módulo completo con `torch.save(modelo)` falla o no es portable con `TorchLayer`. Solo `state_dict()`; y verificar que la recarga reproduce las métricas antes de confiar en los artefactos.
- La reanudabilidad debe comparar la **celda completa** `(modelo, fracción, fold, semilla)`. Comparar solo por modelo omitiría folds pendientes y dejaría el diseño factorial incompleto sin aviso.
- Si se paraleliza la campaña, dos procesos escribiendo el mismo CSV lo corrompen: escribir un archivo por corrida y consolidar en task-14.

**Decisión a congelar aquí.** Presupuesto de épocas, optimizador, tasa de aprendizaje y criterio de época reportada. Cualquier cambio posterior invalida las comparaciones ya ejecutadas y obliga a repetir la campaña completa, no solo la celda afectada.

Plan revisado: clave reanudabilidad 4-tupla, helper dataloading separado del Trainer (SRP), duck typing n_capas_vqc, convención de pesos con semilla, drop_last en train loader.

Validación: uv run pytest tests/test_trainer.py -q (8 passed); uv run pytest tests/ -q (57 passed). Hallazgo en h1_arquitectura.tex con label hallazgo:task-8.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementado Trainer unificado (src/train/trainer.py) con helper dataloading, reanudabilidad 4-tupla, state_dict, presupuesto fijo de 15 épocas y hiperparámetros congelados. Verificado con pytest: 8 tests en test_trainer.py, 57 en suite completa.
<!-- SECTION:FINAL_SUMMARY:END -->
