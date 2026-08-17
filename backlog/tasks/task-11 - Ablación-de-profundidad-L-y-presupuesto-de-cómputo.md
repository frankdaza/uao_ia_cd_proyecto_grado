---
id: TASK-11
title: Ablación de profundidad L y presupuesto de cómputo
status: To Do
assignee:
  - Frank Daza
created_date: '2026-08-17 01:10'
updated_date: '2026-08-17 01:10'
labels:
  - qml
  - bitacora
milestone: m-1
dependencies:
  - TASK-8
  - TASK-10
references:
  - 'https://arxiv.org/abs/2101.02138'
  - 'https://www.nature.com/articles/s41467-018-07090-4'
  - 'https://www.nature.com/articles/s41467-022-32550-3'
documentation:
  - docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex
  - docs/trabajo_de_grado/Bitacora Metodologica de Hallazgos.tex
  - .cursor/skills/ejecutar-experimento/SKILL.md
priority: high
type: spike
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Actividad del anteproyecto:** resto de A2 — selección de la profundidad del ansatz. Además, **compuerta de viabilidad** para toda la Fase 2.

**Qué.** Un protocolo económico para elegir `L ∈ {2, 4, 6}` y, en la misma corrida, **medir** el costo real de la campaña de task-13. Cierra con `results/selected_hparams.json` y una decisión explícita de *go/no-go*.

**Por qué.** Esta tarea cierra dos riesgos a la vez, y los cierra juntos porque comparten la misma medición.

1. **Riesgo de entrenabilidad.** `L` gobierna el equilibrio entre expresividad y entrenabilidad: más capas representan más funciones, pero achatan los gradientes hacia la meseta árida (`holmes2022connecting`, `McClean2018B`). No hay un valor "correcto" a priori; hay que medirlo en este dataset y con esta codificación.
2. **Riesgo de cómputo.** Con `parameter-shift` cada parámetro cuesta 2 evaluaciones de circuito. Con `L = 6` son `6 × 4 × 3 = 72` parámetros, es decir **144 evaluaciones por muestra y por paso**, multiplicadas por el tamaño del lote, por las épocas, y luego por las 60 celdas del diseño factorial. Descubrir a mitad de la campaña que no cabe en el tiempo disponible cuesta semanas; medirlo antes cuesta horas.

**Protocolo económico y declarado.** Un solo fold, la fracción del 25 %, presupuesto de épocas reducido y explícito. No es un experimento de A8 y **no debe reportarse como si tuviera significancia estadística**: es una decisión de diseño instrumentada.

**Entregable.** `results/ablacion_L.csv`, `results/selected_hparams.json` con `L` congelado y el registro del presupuesto medido con su decisión.

**Flujo de la ablación y la compuerta.**

```mermaid
flowchart TB
  L2["L = 2"] --> Corr["Trainer (task-8)<br/>1 fold, fraccion 25%,<br/>presupuesto reducido"]
  L4["L = 4"] --> Corr
  L6["L = 6"] --> Corr
  Corr --> Tab["Tabla comparativa:<br/>F1 macro, tiempo por epoca,<br/>norma de gradiente, n de parametros"]
  Tab --> Sel["L congelado →<br/>results/selected_hparams.json"]
  Tab --> Est["Extrapolacion medida:<br/>costo de las 60 celdas"]
  Est --> Gate{"go / no-go"}
  Gate -->|go| T13["task-13 — campana completa"]
  Gate -->|no-go| Mit["Mitigacion documentada:<br/>cache de caracteristicas,<br/>menos epocas, k reducido"]
```

**Decisión D2 en juego.** El diseño factorial balanceado incluye al HQCNN también al 100 % de los datos (3 modelos × 4 fracciones × 5 folds = 60 corridas), lo que es necesario para la ANOVA de dos vías de task-15. Esa decisión queda **sujeta al presupuesto medido aquí**: si el costo la hace inviable, la alternativa y su impacto sobre el análisis estadístico deben documentarse antes de continuar.

**Claves BibTeX.** `holmes2022connecting`, `McClean2018B`, `caro2022generalization`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existe una tabla comparativa para L en 2, 4 y 6 con F1 macro de validación, tiempo por época, norma media del gradiente y número de parámetros cuánticos
- [ ] #2 El protocolo de la ablación es económico y está declarado por escrito (un fold, una fracción, presupuesto de épocas reducido) y no se presenta como un resultado de A8
- [ ] #3 Se reportan las curvas de pérdida por profundidad, de modo que no se confunda una arquitectura peor con una que no convergió en el presupuesto reducido
- [ ] #4 El criterio de selección de L se escribe antes de observar los resultados y se aplica tal cual
- [ ] #5 L queda congelado en results/selected_hparams.json antes de iniciar task-13, con criterio, fecha y commit
- [ ] #6 Se reporta el costo medido por corrida y la extrapolación a las 60 celdas del diseño factorial en horas de cómputo, indicando el dispositivo
- [ ] #7 Existe una decisión explícita go/no-go sobre la viabilidad de la campaña y, si es no-go, las mitigaciones evaluadas y la adoptada quedan documentadas con su impacto metodológico
- [ ] #8 La decisión D2 de entrenar el HQCNN también al 100 por ciento se confirma o se ajusta con base en el presupuesto medido
- [ ] #9 Hallazgo registrado en hallazgos/h1_arquitectura.tex con \label{hallazgo:task-11}, tabla comparativa y decisión
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 La ablación reutiliza el Trainer de task-8 sin escribir un bucle de entrenamiento nuevo
- [ ] #2 Si se adopta el precálculo de características como mitigación, su incompatibilidad con el aumento de datos queda declarada y reflejada en el método
- [ ] #3 El simulador cuántico usado es el mismo que usará la campaña, para que los tiempos sean extrapolables
- [ ] #4 Tipado de Python 3.12 y docstrings NumPy en español latinoamericano
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Declarar el protocolo **antes** de correr, para que no se confunda con la campaña de A8: fold 0, fracción 0.25, presupuesto de épocas reducido (por ejemplo un tercio del de la campaña), semilla 42, mismo `Trainer` y mismo optimizador.
2. Ejecutar las tres profundidades reutilizando la infraestructura, sin escribir un bucle nuevo:

```python
FILAS: list[dict[str, float | int]] = []
for n_capas in (2, 4, 6):
    cfg = replace(cfg_base, n_capas=n_capas, epocas=EPOCAS_ABLACION, data_fraction=0.25)
    set_seed(cfg.semilla)
    modelo = build_model("hqcnn", cfg)
    registro, historial = Trainer(modelo, cfg, get_device()).ajustar(
        cargador_train, cargador_val
    )
    FILAS.append(
        {
            "n_capas": n_capas,
            "f1_val_macro": registro.f1_val_macro,
            "accuracy_val": registro.accuracy_val,
            "segundos_por_epoca": registro.train_time_s / cfg.epocas,
            "n_params_cuanticos": n_capas * cfg.n_qubits * 3,
            "norma_gradiente_inicial": norma_gradiente_inicial(n_capas),
        }
    )
```

3. Guardar también la curva de pérdida de cada `L`: sin ella no se puede distinguir "peor arquitectura" de "no convergió en el presupuesto reducido", que es el error de interpretación más probable aquí.
4. Extrapolar el costo de la campaña con la medición real, no con una estimación teórica:

```python
def estimar_campana(segundos_por_epoca: float, epocas: int, n_celdas: int = 60) -> float:
    """Estima las horas de cómputo de la campaña completa.

    Notes
    -----
    Se parte del tiempo por época medido en la ablación y se escala por el
    número de celdas del diseño factorial. Es una cota inferior: las corridas
    al 100 % procesan más muestras por época que la fracción del 25 % usada
    en la ablación.
    """
    return segundos_por_epoca * epocas * n_celdas / 3600
```

5. Escribir el criterio de selección **antes** de ver los resultados (por ejemplo: mayor F1 macro; ante diferencias dentro del ruido, la `L` más pequeña por costo y entrenabilidad) y aplicarlo tal cual.
6. Congelar la decisión en `results/selected_hparams.json` con `L`, el criterio, la fecha y el commit.
7. Tomar la decisión *go/no-go*. Si es *no-go*, evaluar y documentar las mitigaciones: precalcular las características del backbone congelado, reducir el presupuesto de épocas, reducir `k`, o restringir el diseño factorial. Registrar cuál se adopta y su impacto en el análisis estadístico.
8. Confirmar o ajustar D2 (HQCNN también al 100 %) con base en el presupuesto medido.
9. Registrar el hallazgo en `hallazgos/h1_arquitectura.tex`: tabla comparativa, curvas de pérdida, presupuesto estimado y decisión con su justificación.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Trampas conocidas.**

- **La mitigación más tentadora tiene un costo metodológico oculto.** Precalcular las características del backbone congelado elimina el paso convolucional en cada época y puede acelerar la campaña de forma drástica; pero es **incompatible con el aumento de datos**, porque el aumento cambia la imagen en cada época y por tanto sus características. Si se adopta, hay que declararlo: el experimento pasa a ser "sin aumento" y task-5 debe reflejarlo, o hay que cachear varias variantes aumentadas por imagen. Adoptarla en silencio invalidaría la comparación con la literatura y la propia descripción del método.
- Medir la norma del gradiente sobre **todos** los parámetros no sirve: la señal de meseta árida está en el bloque cuántico. Promediar sobre los pesos del VQC exclusivamente.
- Una ablación con un solo fold **no** tiene significancia estadística. Reportarla como resultado sería un error de interpretación; es una decisión de diseño instrumentada y así debe presentarse en la bitácora y en el documento final.
- Concluir "L = 6 es peor" cuando lo que ocurrió es que no convergió en el presupuesto reducido es el error más probable de esta tarea. De ahí la exigencia de reportar la curva de pérdida junto con el F1.
- Si la ablación usa `lightning.qubit` y la campaña usa `default.qubit`, los tiempos medidos **no** son extrapolables: cambia el camino de diferenciación. Elegir un simulador, declararlo y usar el mismo en ambas.
- La extrapolación desde la fracción del 25 % es una **cota inferior**: las celdas al 100 % procesan cuatro veces más muestras por época. Aplicar el factor correspondiente por fracción en lugar de multiplicar por 60 sin más.
- El tiempo en Colab no es estable entre sesiones (GPU asignada, contención). Registrar el dispositivo con cada medición y, si es posible, medir en el mismo entorno donde correrá la campaña.

**Regla de cierre.** `L` queda congelado antes de iniciar task-13. Cambiar `L` a mitad de la campaña invalidaría las celdas ya ejecutadas, no solo la afectada.
<!-- SECTION:NOTES:END -->
