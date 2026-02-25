# Filosofía y Estrategia de Agentes de IA

Este repositorio contiene la Tesis de Maestría de Frank Daza. Dado el rigor académico, metodológico y científico requerido para este proyecto (el cual involucra Aprendizaje Automático Cuántico Híbrido con PyTorch y PennyLane, así como documentación en LaTeX), el uso de Agentes de Inteligencia Artificial (IA) en Cursor debe estar estrictamente gobernado por las reglas y habilidades definidas aquí.

## 📌 Objetivo Principal

El uso de la IA en este proyecto no es solo para "escribir código más rápido", sino para:
1.  **Garantizar la reproducibilidad** de los experimentos científicos.
2.  **Mantener la rigurosidad académica** en la redacción del documento de tesis en LaTeX.
3.  **Estandarizar procesos repetitivos** (como la ejecución de pruebas y la adición de referencias bibliográficas).
4.  **Minimizar la brecha metodológica** garantizando que cualquier bloque de código interactúe de forma segura y transparente.

Todas las interacciones con la IA deben realizarse exclusivamente en **Español Latinoamericano**.

---

## 🛠️ Reglas del Proyecto (`.cursor/rules`)

Las reglas son restricciones globales que el Agente debe respetar de forma incondicional, dependiendo del contexto del archivo en el que esté trabajando.

| Regla | Descripción |
| :--- | :--- |
| `python-y-ml.mdc` | Define las directrices estrictas para escribir código de Machine Learning con **PyTorch** y **PennyLane**. Incluye fijación de semillas (seeds), tipado estricto, docstrings científicos (estilo NumPy), y manejo de data. |
| `escritura-latex.mdc` | Reglas sobre cómo se debe editar el anteproyecto y la tesis en LaTeX. Enfatiza el tono académico, el uso correcto de citas con BibTeX y la prohibición de inventar referencias o sobreescribir estilos base. |
| `cuadernos-jupyter.mdc` | Instrucciones de calidad para la carpeta `notebooks/`. Obliga a que los notebooks sean auto-contenidos, contengan celdas Markdown explicando la teoría y mantengan limpios los outputs antes de los commits. |

---

## 🚀 Habilidades del Agente (`.cursor/skills`)

Las habilidades (Skills) son procedimientos paso-a-paso estructurados que la IA tiene autoridad de ejecutar a solicitud. Si se requiere ejecutar un Skill, la IA **debe** leer su archivo correspondiente y seguir la secuencia exacta.

| Habilidad (Skill) | Directorio | Descripción |
| :--- | :--- | :--- |
| **Ejecutar Experimento** | `ejecutar-experimento` | Flujo automatizado para preparar el dataset, instanciar la arquitectura híbrida (HQCNN), iniciar el entrenamiento PyTorch y almacenar las métricas y pesos (weights) en el directorio `results/`. |
| **Agregar Citación** | `agregar-cita` | Procedimiento estándar para insertar nuevas referencias bibliográficas. Incluye el formateo en BibTeX, la inserción alfabética en `Referencias.bib`, y la citación en el documento `.tex`. |

---

## 📖 Instrucciones de Uso para el Agente

Cuando se asigne una nueva tarea u objetivo al Agente (por ejemplo, implementar una nueva capa cuántica o escribir un capítulo metodológico):

1.  **Analizar el contexto**: El Agente debe deducir si está tocando archivos Python, LaTeX o Notebooks y cargar en memoria la regla aplicable `.mdc`.
2.  **Aplicar el idioma**: Asegurarse de que cualquier comentario, explicación teórica, log, docstring o interacción en el chat se redacte en **Español Latinoamericano**.
3.  **Prohibición de "Cajas Negras"**: Cualquier código propuesto por la IA para la arquitectura híbrida (CNN + VQC) debe ser explícitamente rastreable. La IA explicará *por qué* definió un *Ansatz* o un *embedding* específico basándose en la literatura.
4.  **Citar Siempre**: Si la IA introduce un concepto de QML, debe sugerir o requerir una citación formal para el `.tex`.

---
*Este documento establece el contrato marco para la interacción humano-IA en el proyecto de Tesis de Maestría de Frank Daza (2025).*
