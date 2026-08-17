---
id: TASK-13
title: Campaña experimental k-fold en escenarios de escasez (A8)
status: To Do
assignee:
  - Frank Daza
created_date: '2026-08-17 01:12'
updated_date: '2026-08-17 01:12'
labels:
  - escasez
  - bitacora
milestone: m-2
dependencies:
  - TASK-11
  - TASK-12
references:
  - 'https://www.nature.com/articles/s41467-022-32550-3'
  - 'https://www.nature.com/articles/s41746-025-01597-z'
  - 'https://www.nature.com/articles/s41598-021-93651-5'
documentation:
  - docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex
  - docs/trabajo_de_grado/Bitacora Metodologica de Hallazgos.tex
  - .cursor/skills/ejecutar-experimento/SKILL.md
priority: high
type: task
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Actividad del anteproyecto:** A8 — Campaña experimental con validación cruzada en escenarios de escasez.

**Qué.** Ejecutar el diseño factorial completo `{hqcnn, efficientnet_b0, resnet50} × {10 %, 25 %, 50 %, 100 %} × 5 folds` = **60 celdas**, de las cuales 10 se **reutilizan** de task-12, así que son 50 corridas nuevas.

**Por qué.** Es el experimento que responde la pregunta de investigación. El diseño es factorial y **balanceado** (decisión D2: el HQCNN también se entrena al 100 %) porque el análisis de task-15 necesita estimar el término de interacción `modelo × fracción`; un diseño incompleto dejaría la hipótesis central sin prueba estadística. La comparación es **pareada**: todas las celdas comparten los folds y las submuestras de `results/splits.json`, de modo que la diferencia entre modelos no arrastra ruido de partición.

**Entregable.** `results/experiments.csv` con 60 filas, `results/history/` con el historial por época de cada celda, pesos en `models/` y un informe de estado de ejecución.

**Diseño de la campaña.**

```mermaid
flowchart TB
  Sel["selected_hparams.json<br/>L congelado (task-11)"] --> C["Campana experimental"]
  Base["10 celdas al 100%<br/>reutilizadas de task-12"] --> C
  Splits["results/splits.json<br/>folds y fracciones fijas"] --> C
  C --> G["3 modelos x 4 fracciones x 5 folds<br/>= 60 celdas (50 nuevas)"]
  G --> CSV["results/experiments.csv"]
  G --> H["results/history/*.json"]
  CSV --> T14["task-14 — A9"]
  CSV --> T15["task-15 — A10"]
  H --> T16["task-16 — A11"]
```

**Orden de ejecución recomendado.** De menor a mayor costo: primero las fracciones pequeñas, y las celdas del HQCNN al 100 % **al final**. Son las más caras de todo el proyecto y así, si el presupuesto se agota, lo que falta es la celda más costosa y no la mitad del diseño.

**Claves BibTeX.** `caro2022generalization`, `gupta2025systematic`. Excepción justificada: `li2021cvstability` (estabilidad de la validación cruzada en imagen médica).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Se completan las 60 celdas del diseño factorial (3 modelos x 4 fracciones x 5 folds) reutilizando las 10 celdas al 100 por ciento de task-12 sin repetirlas
- [ ] #2 El HQCNN usa exclusivamente la profundidad L congelada en results/selected_hparams.json
- [ ] #3 Todas las celdas comparten results/splits.json, de modo que la comparación entre modelos es pareada por fold y fracción
- [ ] #4 El CSV consolidado tiene exactamente una fila por celda, sin duplicados, verificado programáticamente antes de pasar al análisis
- [ ] #5 Existe historial por época para todas las celdas, suficiente para reconstruir las curvas de A11 sin reentrenar
- [ ] #6 La campaña es reanudable y se documenta el estado de ejecución: celdas completadas, pendientes y fallidas con su motivo
- [ ] #7 Ninguna celda fallida desaparece del registro: se anota explícitamente con la causa
- [ ] #8 Se compara el costo real total de cómputo con la estimación de task-11 y se explica la desviación
- [ ] #9 Hallazgo registrado en hallazgos/h2_experimentacion.tex con \label{hallazgo:task-13} con el resumen del diseño y el estado de ejecución
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Ningún hiperparámetro cambia a mitad de campaña; si un cambio resulta imprescindible, se documenta y se repiten todas las celdas comparables
- [ ] #2 Las pruebas informales quedan fuera del CSV oficial
- [ ] #3 Pesos e historiales persistidos con la convención de nombres del contrato de task-4
- [ ] #4 Semilla, dispositivo y commit registrados en cada fila
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verificar las precondiciones antes de lanzar nada: `L` congelado en `results/selected_hparams.json`, decisión *go* registrada en task-11, `results/splits.json` con el hash del manifiesto válido y las 10 celdas de task-12 presentes en el CSV.
2. Recorrer el diseño factorial delegando toda la lógica al `Trainer`, que ya sabe omitir celdas hechas:

```python
MODELOS = ("efficientnet_b0", "resnet50", "hqcnn")
FRACCIONES = (0.10, 0.25, 0.50, 1.00)

for fraccion in FRACCIONES:
    for nombre in MODELOS:
        for fold in range(cfg_base.n_folds):
            if celda_completada(nombre, fraccion, fold):
                continue
            cfg = replace(
                cfg_base,
                modelo=nombre,
                data_fraction=fraccion,
                n_capas=hparams_seleccionados["n_capas"],
            )
            set_seed(cfg.semilla)
            modelo = build_model(nombre, cfg)
            registro, historial = Trainer(modelo, cfg, get_device()).ajustar(
                *construir_loaders(cfg, particiones, fold=fold)
            )
            escribir_registro(registro)
            escribir_historial(historial, cfg, fold)
```

3. Ejecutar por bloques (una fracción a la vez) para que una sesión interrumpida de Colab no pierda más que el bloque en curso, con wandb en modo offline y sincronización posterior.
4. Registrar las celdas fallidas de forma explícita, con el motivo (memoria agotada, sesión cortada, error numérico), en lugar de dejarlas ausentes del CSV sin rastro.
5. Verificar la integridad del diseño al terminar:

```python
esperadas = len(MODELOS) * len(FRACCIONES) * cfg_base.n_folds  # 60
assert len(df) == esperadas, f"Faltan celdas: {esperadas - len(df)}"
assert not df.duplicated(subset=["modelo", "data_fraction", "fold"]).any()
```

6. Comparar el costo real total con la estimación de task-11 y explicar la desviación: es información valiosa para el capítulo de método y para cualquier réplica.
7. Registrar el hallazgo en `hallazgos/h2_experimentacion.tex`: resumen del diseño factorial, tabla de estado de ejecución (completadas, pendientes, fallidas) y costo real frente al estimado.

Ejecución (ver el skill `ejecutar-experimento`):

```bash
uv run python -m src.experiments.campana --fraccion 0.10
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Trampas conocidas.**

- **El riesgo dominante es la tentación de "arreglar" algo a mitad de campaña.** Cualquier cambio de hiperparámetro, de semilla, de preprocesamiento o de `L` invalida las celdas ya ejecutadas. Si un cambio resulta imprescindible, hay que repetir **todo** el bloque comparable, no solo añadir la celda nueva; mezclar celdas de dos configuraciones produce una tabla que parece completa y no lo es.
- Un fallo silencioso (memoria agotada, sesión cortada) deja el CSV con menos de 60 filas y desbalancea la ANOVA de dos vías. La verificación de integridad del punto 5 no es opcional: es la condición para pasar a task-14.
- No promediar folds al escribir el CSV. La unidad de observación es la corrida; promediar aquí destruye la información que task-15 necesita para estimar la varianza dentro de cada celda.
- Las celdas del HQCNN al 100 % son las más costosas del proyecto por el costo de `parameter-shift`. Dejarlas al final es una decisión de gestión de riesgo, no una preferencia estética.
- Si la campaña se paraleliza, dos procesos escribiendo el mismo CSV lo corrompen: un archivo por corrida y consolidación en task-14.
- Reentrenar por accidente las 10 celdas de task-12 metería dos observaciones de la misma celda en el análisis. La reanudabilidad debe comparar la clave completa `(modelo, fracción, fold)`.
- Las corridas al 10 % son rápidas y tentadoras para "probar cosas". Todo lo que se pruebe fuera del protocolo debe quedar fuera del CSV oficial, en un archivo aparte, o el registro experimental pierde credibilidad.

**Nota sobre la validación cruzada.** Los 5 folds comparten datos de entrenamiento entre sí; la varianza entre folds mide estabilidad de la partición, no error de muestreo independiente. Esta limitación, declarada desde task-6, es la que condiciona la lectura de los valores p en task-15.
<!-- SECTION:NOTES:END -->
