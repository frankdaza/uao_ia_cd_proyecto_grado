---
name: ejecutar-experimento
description: Automatiza entrenamiento y evaluación de HQCNN o baselines clásicos (EfficientNet-B0, ResNet-50) con UV, k-fold estratificado, wandb y persistencia en results/. Usar al entrenar, ejecutar experimentos, validar modelos híbridos o líneas base en escenarios de escasez de datos.
---

# Ejecución de Experimentos (PyTorch 2.9 + PennyLane 0.45)

Asegura reproducibilidad y control de calidad en cada iteración experimental. Todo log y documentación en **Español Latinoamericano**.

## Precondiciones

```bash
uv sync                    # o uv sync --frozen si existe uv.lock
uv run python -c "import torch, pennylane; print(torch.__version__, pennylane.__version__)"
```

En Colab: instalar con pins del `pyproject.toml` (ver `cuadernos-jupyter.mdc`).

## Procedimiento

### Paso 1: Verificar dataset

- Confirmar ruta del **Brain Tumor MRI Dataset** (7023 imágenes, 4 clases).
- Pipeline: resize 224×224, normalización ImageNet, augmentation (rotaciones leves, flips).
- Auditar balance de clases (glioma, meningioma, pituitario, no tumor).

### Paso 2: Fijar semillas

Inyectar o verificar `set_seed(42)` (ver `python-y-ml.mdc`) antes de cualquier split o entrenamiento.

### Paso 3: Inicializar modelo

**Baselines clásicos:**

- EfficientNet-B0 y ResNet-50 con `weights=` API (no `pretrained=True`).
- Congelar extractor; entrenar cabeza de clasificación.

**HQCNN:**

- Extractor: EfficientNet-B0 congelado → capa densa → `qml.qnn.TorchLayer`.
- VQC: `AngleEmbedding` + `StronglyEntanglingLayers`, `L ∈ {2, 4, 6}`, 4 qubits.
- QNode: `interface="torch"`, `diff_method="parameter-shift"`, dispositivo `default.qubit`.

Documentar:

- Parámetros entrenables vs congelados.
- `n_qubits`, `n_layers`, `diff_method`, dispositivo PyTorch y simulador cuántico.

### Paso 4: Escenarios de escasez

Crear subconjuntos estratificados al **10 %, 25 %, 50 % y 100 %** con `StratifiedShuffleSplit` (`random_state=42`).

### Paso 5: Validación cruzada

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Modelo fresco por fold.
- Logs por época (no por batch); usar `tqdm`.
- Registrar tiempo de entrenamiento e inferencia por fold.

### Paso 6: Métricas

Por fold y escenario, calcular y guardar:

- Accuracy, F1-Score ponderado.
- Sensibilidad y especificidad por clase.
- Loss (train/val).
- Brecha de generalización: `G = |Acc_train - Acc_val|`.
- Latencia de inferencia (ms/batch).

### Paso 7: Persistencia

| Artefacto | Ubicación |
| :--- | :--- |
| Pesos (`state_dict`) | `models/{modelo}_{fraccion}_fold{k}_acc{xx}.pth` |
| Métricas tabulares | `results/metrics_{modelo}_{fecha}.csv` |
| Curvas y confusion matrices | `results/figures/` |
| Monitoreo en tiempo real | wandb (proyecto configurado por el usuario) |

Registrar en wandb/CSV: hiperparámetros, semilla, fracción de datos, fold, tiempos y métricas.

### Paso 8: Ejecución local

```bash
uv run python src/train.py --model hqcnn --data-fraction 0.25 --folds 5
```

Adaptar según el script existente; nunca ejecutar con `python` del sistema.

**Local queda reservado para análisis y sondas.** El entrenamiento de la campaña factorial (A8) corre en Colab Pro+ (decisión D3).

### Paso 9: Ejecución en Google Colab Pro+ (campaña A8)

Cuaderno canónico: [`notebooks/colab_campana.ipynb`](../../../notebooks/colab_campana.ipynb). Es delgado por diseño: importa `src.experiments.campana` y **no** reimplementa modelos, particiones ni bucles de entrenamiento.

| Aspecto | Regla |
| :--- | :--- |
| GPU | **T4**. El cuello de botella es la simulación del VQC con `parameter-shift`, que se evalúa en CPU; una GPU de gama alta acelera el backbone congelado, no el término dominante |
| Dependencias | Pines del `README` (PyTorch 2.9.1 + cu128, PennyLane 0.45.1). Única excepción a UV |
| Rutas de Drive | Solo en el cuaderno. `src/` deriva todo de `ExperimentConfig` con `pathlib` |
| Dataset | Zip de Drive copiado al disco local antes de descomprimir; leer desde Drive domina el tiempo por época |
| Persistencia | `results/` y `models/` enlazados a Drive. Sembrar desde el repositorio **solo** si Drive está vacío |
| wandb | `WANDB_MODE=offline` y `wandb sync` al cerrar la sesión |
| Homogeneidad | Las 60 celdas se ejecutan en `cuda`. Toda corrida con otro dispositivo o con presupuesto de épocas distinto se archiva antes de entrenar |

Secuencia de ejecución:

```bash
python -m src.experiments.campana --archivar-no-cuda
python -m src.experiments.campana --modelo hqcnn --fraccion 0.10 --fold 0 \
    --max-epocas 1 --sin-baselines-previas   # sonda: cierra la compuerta de TASK-11
python -m src.experiments.campana --archivar-no-cuda

python -m src.experiments.campana --fraccion 1.00 --modelo efficientnet_b0 --sin-baselines-previas
python -m src.experiments.campana --fraccion 1.00 --modelo resnet50 --sin-baselines-previas
python -m src.experiments.campana --fraccion 0.10
python -m src.experiments.campana --fraccion 0.25
python -m src.experiments.campana --fraccion 0.50
python -m src.experiments.campana --fraccion 1.00   # HQCNN al 100 %, el bloque más caro
python -m src.experiments.campana --verificar
```

Tras cada bloque, actualizar la tabla de estado de ejecución del hallazgo `hallazgo:task-13` en la bitácora con celdas completadas, pendientes, fallidas con motivo y costo acumulado. El registro es incremental, no se difiere al cierre de la campaña.

## Invocación

> "Ejecutar experimento: entrena HQCNN al 25 % con 4 qubits, L=2, k-fold=5."
