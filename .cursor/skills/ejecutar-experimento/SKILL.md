---
description: Procedimiento estructurado para automatizar, monitorear y documentar la ejecución de modelos de Machine Learning (Híbridos Cuánticos o Clásicos).
---

# Ejecución de Experimentos de Machine Learning en PyTorch & PennyLane

**Objetivo:** Asegurar que cada iteración de la experimentación se realice bajo estrictas normas de control de calidad y reproducibilidad. Permite automatizar la creación del pipeline PyTorch.

> [!IMPORTANTE]
> Todo registro, log o documentación generada mediante esta habilidad debe producirse en Español Latinoamericano.

## Pasos del Procedimiento

Para ejecutar un experimento usando la Arquitectura Híbrida CNN-VQC (HQCNN) o los "Classical Baselines" (EfficientNet-B0), procede estrictamente en este orden:

# Paso 1: Verificación del Dataset
Si el objetivo es utilizar un dataset específico, el Agente debe confirmar la ruta del `Brain Tumor MRI Dataset` y revisar que el pipeline de *data augmentation* contenga los pasos estandarizados (Redimensionado, Normalización a tensores, y Data Augmentation si corresponde).

# Paso 2: Fijación de Semillas
El agente debe asegurarse o inyectar código que inicialice la semilla (e.g., `set_seed(42)`) para Pytorch, Random y NumPy. Si el código actual no lo tiene, debe proporcionarlo.

# Paso 3: Inicialización del Modelo
Instancia el modelo propuesto (HQCNN con `TorchLayer` de PennyLane, o Backbone Clásico). El Agente debe:
- Identificar y documentar el número de parámetros entrenables.
- Validar el Ansatz Cuántico utilizado (e.g., *Strongly Entangling Layers*) y la profundidad de sus operaciones.

# Paso 4: Entrenamiento con Validación Cruzada
Si la tarea exige evaluación rigurosa:
* Construye iteraciones sobre un mecanismo de *Stratified k-fold Cross-Validation* (por defecto $k=5$).
* Evita imprimir el "loss" de cada lote de datos individual (*batch*) de las capas de entrenamiento (para evitar ensuciar los logs de Colab). Solo imprime resúmenes por época.
* Mide y guarda los tiempos computacionales de ejecución.

# Paso 5: Persistencia (Guardado)
Finalizado el entrenamiento interactivo o el script de experimentación local:
* Exportar el modelo/los pesos en formato `.pth` en el directorio `models/`. Nombrar el archivo descriptivamente (ej. `hqcnn_kfold1_epoch50_acc89.pth`).
* Imprimir, visualizar o exportar as métricas calculadas como Accuracy, F1-Score y Loss.

## Cómo Invocar Esta Habilidad
**Tú**: "Ejecutar experimento: Configura un entrenamiento para la red híbrida usando el dataset al 25% y 4 qubits."
