# Filosofía y Estrategia de Agentes de IA

Este repositorio contiene la Tesis de Maestría de Frank Daza. Dado el rigor académico, metodológico y científico requerido para este proyecto (Aprendizaje Automático Cuántico Híbrido con PyTorch y PennyLane, documentación en LaTeX y experimentación reproducible), el uso de Agentes de Inteligencia Artificial (IA) en Cursor debe estar estrictamente gobernado por las reglas y habilidades definidas aquí.

## Objetivo Principal

El uso de la IA en este proyecto no es solo para "escribir código más rápido", sino para:

1. **Garantizar la reproducibilidad** de los experimentos científicos.
2. **Mantener la rigurosidad académica** en la redacción del documento de tesis en LaTeX.
3. **Estandarizar procesos repetitivos** (ejecución de experimentos, análisis estadístico y adición de referencias bibliográficas).
4. **Minimizar la brecha metodológica** garantizando que cualquier bloque de código interactúe de forma segura y transparente.

Todas las interacciones con la IA deben realizarse exclusivamente en **Español Latinoamericano**.

---

## Gestor Canónico: UV

**UV** (Astral) es el único gestor permitido de intérprete Python y de paquetes en entornos locales. Quedan prohibidos en recetas del agente, salvo la excepción de Google Colab:

- `pip`, `pipenv`, `poetry`, `conda`, `mamba`, `pyenv`, `virtualenv` y `python -m venv` como flujo principal.

| Acción | Comando canónico |
| :--- | :--- |
| Fijar Python | `uv python pin 3.12` → [`.python-version`](.python-version) |
| Añadir dependencia | `uv add <paquete>` |
| Sincronizar entorno | `uv sync` o `uv sync --frozen` (con `uv.lock`) |
| Ejecutar script | `uv run python src/train.py` |
| Jupyter local | `uv run jupyter lab` |

**Excepción Colab:** el runtime de Google no usa UV. Allí se documenta `pip install` con las mismas versiones del [`pyproject.toml`](pyproject.toml).

**PyTorch:** los índices oficiales (CPU/CUDA) se configuran en `tool.uv.index` y `tool.uv.sources` del `pyproject.toml`, no con `pip install --index-url` suelto.

---

## Stack Congelado (agosto 2026)

No instalar versiones "latest" sin verificar compatibilidad con **PennyLane 0.45**. El combo PyTorch 2.9 + PennyLane 0.45 está validado; versiones más nuevas de PyTorch (2.12+) pueden romper `TorchLayer` y la regla de cambio de parámetros.

| Componente | Versión | Rol |
| :--- | :--- | :--- |
| Python | 3.12 | Intérprete fijado por UV |
| PennyLane + Lightning | 0.45.1 / 0.45.0 | VQC, `TorchLayer`, simulación |
| PyTorch | 2.9.1 | Entrenamiento híbrido end-to-end |
| torchvision | 0.24.1 | EfficientNet-B0, ResNet-50 |
| NumPy | `>=2.0,<2.3` | Requisito de PennyLane |
| scikit-learn | 1.9.0 | k-fold estratificado, métricas |
| SciPy | 1.16.x | Shapiro-Wilk, Levene |
| statsmodels | 0.14.6 | ANOVA, Tukey HSD |
| pandas / matplotlib / seaborn | 2.3 / 3.10 / 0.13 | Tablas y figuras |
| wandb | 0.28.x | Monitoreo de experimentos |
| tqdm / Pillow | vigentes | Logs y carga de MRI |

**Dispositivo PyTorch:** `cuda` (Colab) > `mps` (macOS) > `cpu`.

**Simulador cuántico por defecto:** `default.qubit` con `interface="torch"` y `diff_method="parameter-shift"`. `lightning.qubit` solo como acelerador de prototipo (no default con `parameter-shift` + batching en `TorchLayer`).

---

## APIs Vigentes vs Deprecadas

El agente **no debe** copiar recetas obsoletas del README ni de notebooks antiguos.

| Deprecado | Usar en su lugar |
| :--- | :--- |
| `pretrained=True` en torchvision | `weights=EfficientNet_B0_Weights.IMAGENET1K_V1` |
| QNode sin `interface` | `@qml.qnode(dev, interface="torch", diff_method="parameter-shift")` |
| `qml.expval(qml.PauliZ(i))` | `qml.expval(qml.Z(i))` |
| `pip install torch` (local) | `uv add torch` / `uv sync` |
| `python train.py` (sistema) | `uv run python train.py` |
| Guardar módulo completo con `TorchLayer` | Guardar `model.state_dict()` |
| Skill `add-citation` | Skill `agregar-cita` |

---

## Reglas del Proyecto (`.cursor/rules`)

| Regla | Descripción |
| :--- | :--- |
| `python-y-ml.mdc` | Código Python/ML: UV, semillas, tipado 3.12, PyTorch 2.9, PennyLane 0.45, transfer learning con Weights API. |
| `escritura-latex.mdc` | LaTeX académico: `natbib` + `apalike`, BibTeX en `Referencias.bib`, sin alucinar DOIs. |
| `cuadernos-jupyter.mdc` | Notebooks: UV en local, pins en Colab, outputs limpios, modularidad en `src/` o `models/`. |
| `engram.mdc` | Memoria persistente Engram: cuándo guardar, buscar y recuperar contexto de la tesis. |

---

## Habilidades del Agente (`.cursor/skills`)

Si se requiere ejecutar un Skill, la IA **debe** leer su `SKILL.md` y seguir la secuencia exacta.

| Habilidad | Directorio | Descripción |
| :--- | :--- | :--- |
| **Ejecutar Experimento** | `ejecutar-experimento` | Pipeline HQCNN o baselines clásicos: dataset, semillas, k-fold, métricas, `models/` + `results/` + wandb. |
| **Agregar Citación** | `agregar-cita` | BibTeX validado (Crossref/DOI) → `Referencias.bib` → `~\cite{}` en el `.tex`. |
| **Analizar Resultados** | `analizar-resultados` | ANOVA, Tukey HSD, brecha de generalización y figuras desde CSV en `results/`. |

---

## Instrucciones de Uso para el Agente

1. **Analizar el contexto:** deducir si se tocan archivos Python, LaTeX o Notebooks y cargar la regla `.mdc` aplicable.
2. **Aplicar el idioma:** comentarios, docstrings, logs y chat en **Español Latinoamericano**.
3. **Prohibición de cajas negras:** toda arquitectura híbrida (CNN + VQC) debe ser rastreable; explicar *por qué* se eligió un *Ansatz* o *embedding* con base en la literatura.
4. **Citar siempre:** si se introduce un concepto de QML, sugerir o requerir una citación formal en el `.tex` (skill `agregar-cita`).
5. **Reproducibilidad:** antes de entrenar localmente, verificar `uv sync`; fijar semillas; registrar hiperparámetros en wandb o CSV.

---

## CodeGraph (inteligencia estructural del código)

<!-- CODEGRAPH START -->

Este proyecto tiene CodeGraph inicializado (`.codegraph/`). La herramienta MCP `codegraph_explore` es el índice preconstruido del código — úsala **antes** de bucles grep/read para preguntas estructurales.

- **Cuándo usarla:** cómo funciona X, flujo de X a Y, qué llama qué, dónde está definido X, o explorar un área del repositorio.
- **Una llamada suele bastar:** devuelve fuente verbatim con números de línea, rutas de llamada (incluye dispatch dinámico) y radio de impacto.
- **Confiar en los resultados** del grafo AST; no re-verificar con grep. Si la respuesta incluye banner `⚠️` de archivos pendientes de sync, usar Read solo en esos archivos.
- **CLI equivalente** (subagentes sin MCP): `codegraph explore "<consulta>"` en la raíz del repo.
- **Si falta `.codegraph/`:** ejecutar `codegraph init` y reintentar.

<!-- CODEGRAPH END -->

*Este documento establece el contrato marco para la interacción humano-IA en el proyecto de Tesis de Maestría de Frank Daza (2026).*

<!-- BACKLOG.MD GUIDELINES START -->
<!-- backlog.md-instructions-version: 1.50.1 -->
<CRITICAL_INSTRUCTION>

## Backlog.md Workflow

This project uses Backlog.md for task and project management.

**For every user request in this project, run `backlog instructions overview` before answering or taking action.**

Use the overview to decide whether to search, read, create, or update Backlog tasks.

Before task lifecycle actions, read the matching detailed guide:
- `backlog instructions task-creation` before creating or splitting tasks
- `backlog instructions task-execution` before planning, changing status or assignee, adding a plan or implementation notes, or implementing task work
- `backlog instructions task-finalization` before checking acceptance criteria, writing final summaries, or moving tasks to terminal statuses

Use `backlog <command> --help` before running unfamiliar commands. Help shows options, fields, and examples.

Do not edit Backlog task, draft, document, decision, or milestone markdown files directly. Use the `backlog` CLI so metadata, relationships, and history stay consistent.

</CRITICAL_INSTRUCTION>
<!-- BACKLOG.MD GUIDELINES END -->
