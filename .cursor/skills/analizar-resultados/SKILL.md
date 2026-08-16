---
name: analizar-resultados
description: Ejecuta análisis estadístico (ANOVA, Tukey HSD, pruebas de normalidad) y genera figuras comparativas desde CSV en results/. Usar al analizar métricas experimentales, comparar HQCNN vs baselines, o documentar significancia estadística en la tesis.
---

# Análisis Estadístico de Resultados Experimentales

Procedimiento para actividades **A10** (ANOVA) y **A11** (brecha de generalización) del anteproyecto. Salida en **Español Latinoamericano**.

## Entrada esperada

CSV en `results/` con columnas mínimas:

| Columna | Descripción |
| :--- | :--- |
| `modelo` | `hqcnn`, `efficientnet_b0`, `resnet50` |
| `data_fraction` | 0.10, 0.25, 0.50, 1.00 |
| `fold` | 1–5 |
| `accuracy_train` / `accuracy_val` | Exactitud train y validación |
| `f1_val` | F1-Score ponderado en validación |
| `train_time_s` | Tiempo de entrenamiento (segundos) |

## Procedimiento

### Paso 1: Cargar y agregar

```bash
uv run python -c "import pandas as pd; print(pd.read_csv('results/metrics_*.csv').head())"
```

Calcular por `(modelo, data_fraction)`:

- Media y desviación estándar de `accuracy_val`, `f1_val`.
- Brecha de generalización: `G = |accuracy_train - accuracy_val|`.

### Paso 2: Pruebas de supuestos

Con `scipy.stats`:

- **Shapiro-Wilk** por grupo (normalidad de métricas).
- **Levene** (homogeneidad de varianzas entre modelos).

Documentar si se cumplen supuestos para ANOVA paramétrica.

### Paso 3: ANOVA y post-hoc

Con `statsmodels`:

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("results/metrics_hqcnn_20260816.csv")
model = ols("accuracy_val ~ C(modelo)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
tukey = pairwise_tukeyhsd(df["accuracy_val"], df["modelo"])
```

Repetir por fracción de datos si el diseño lo requiere.

### Paso 4: Visualización

Guardar en `results/figures/`:

- Barras con error bars (media ± std) por modelo y fracción.
- Curvas de aprendizaje (loss/accuracy vs época) si hay series temporales.
- Matrices de confusión por fold (si están disponibles).
- Boxplot de brecha de generalización `G`.

Usar `matplotlib` + `seaborn`; etiquetas en español.

### Paso 5: Reporte para la tesis

Redactar un párrafo con:

- Estadístico F y p-valor del ANOVA.
- Comparaciones significativas del Tukey HSD.
- Interpretación de la brecha de generalización (sobreajuste).

Sugerir `\cite{}` para la metodología estadística si aún no está citada (skill `agregar-cita`).

## Salida

| Artefacto | Ubicación |
| :--- | :--- |
| Tabla ANOVA | `results/anova_{fecha}.csv` |
| Tukey HSD | `results/tukey_{fecha}.csv` |
| Figuras | `results/figures/` |
| Resumen narrativo | comentario al usuario o sección `.tex` solicitada |

## Invocación

> "Analiza los resultados del CSV en results/ y genera ANOVA comparando HQCNN vs EfficientNet al 25 %."
