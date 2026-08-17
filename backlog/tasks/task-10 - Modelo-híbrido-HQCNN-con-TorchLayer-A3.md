---
id: TASK-10
title: Modelo híbrido HQCNN con TorchLayer (A3)
status: Done
assignee:
  - Frank Daza
created_date: '2026-08-17 01:08'
updated_date: '2026-08-17 03:00'
labels:
  - arquitectura
  - qml
  - bitacora
milestone: m-1
dependencies:
  - TASK-7
  - TASK-9
references:
  - 'https://arxiv.org/abs/1811.04968'
  - 'https://arxiv.org/abs/1803.00745'
  - 'https://quantum-journal.org/papers/q-2020-10-09-340/'
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
**Actividad del anteproyecto:** A3 — Integración del modelo híbrido HQCNN.

**Qué.** Un `nn.Module` de extremo a extremo que va del tensor de imagen a 4 logits: backbone congelado (task-7) → capa densa de reducción → acotación de ángulos → `qml.qnn.TorchLayer` sobre el QNode de task-9 → `CrossEntropyLoss`.

**Por qué.** Es el punto donde el diseño deja de ser dos piezas y pasa a ser un modelo entrenable de extremo a extremo con un solo grafo de diferenciación. `TorchLayer` hace que el circuito variacional se comporte como cualquier capa de PyTorch, lo que permite que el `Trainer` de task-8 lo trate exactamente igual que a una línea base clásica (LSP) y que el gradiente fluya del logit hasta los pesos cuánticos por la regla de cambio de parámetros (`bergholm2018pennylane`, `mitarai2018quantum`). La transferencia clásico-cuántica con extractor congelado es el patrón establecido por `mari2020transfer`.

**Entregable.** `src/models/hqcnn.py` con la clase `HQCNN`, la verificación numérica del gradiente y el conteo de parámetros por bloque.

**Arquitectura de extremo a extremo.**

```mermaid
flowchart LR
  MRI["MRI 224x224x3"] --> BB["Backbone congelado<br/>EfficientNet-B0 (eval)"]
  BB --> Lat["Vector latente 1280"]
  Lat --> Dense["Linear(1280 → 4)<br/>UNICO bloque clasico entrenable"]
  Dense --> Acot["tanh(x) * pi<br/>acotacion de angulos"]
  Acot --> AE["AngleEmbedding 4 qubits"]
  AE --> SEL["StronglyEntanglingLayers L"]
  SEL --> Z["expval Z local por qubit"]
  Z --> Logits["4 logits → CrossEntropyLoss"]
```

**Detalle crítico de la integración.** La acotación entre la capa densa y el `AngleEmbedding` no es cosmética: sin ella, los ángulos se envuelven módulo 2π y dos características muy distintas se codifican en el mismo estado, destruyendo información **sin ningún error visible**. `tanh` escalado a [-π, π] resuelve el problema y la elección debe quedar justificada.

**Segundo detalle crítico.** El valor esperado de Z vive en [-1, 1], así que los logits tienen escala pequeña y el softmax queda plano: la confianza máxima alcanzable está acotada por construcción. Es una propiedad del diseño, no un error; si se decide añadir un factor de escala aprendible, debe aplicarse **igual en todas las corridas** y quedar registrado.

**Claves BibTeX.** `bergholm2018pennylane`, `mitarai2018quantum`, `mari2020transfer`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 El modelo es un único nn.Module que va del tensor de imagen a 4 logits, sin pasos manuales intermedios
- [x] #2 El QNode se integra con qml.qnn.TorchLayer declarando interface=torch, diff_method=parameter-shift y default.qubit
- [x] #3 La salida de la capa densa se acota antes del AngleEmbedding y la elección está justificada por la periodicidad de los ángulos
- [x] #4 El gradiente que llega a los pesos cuánticos se verifica numéricamente contra diferencias finitas con tolerancia declarada
- [x] #5 El módulo mantiene el backbone en eval() incluso cuando el Trainer llama train(), y una prueba lo verifica
- [x] #6 state_dict() guarda y recarga pesos clásicos y cuánticos reproduciendo la misma salida para una entrada fija
- [x] #7 El modelo es sustituible por una línea base clásica en el mismo Trainer sin ramas condicionales (LSP)
- [x] #8 Se documenta la escala acotada de los logits derivada del valor esperado en [-1, 1] y su efecto sobre la confianza del softmax
- [x] #9 Hallazgo registrado en hallazgos/h1_arquitectura.tex con \label{hallazgo:task-10}, esquema de la arquitectura y conteo de parámetros por bloque
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tipado de Python 3.12 y docstrings NumPy en español latinoamericano
- [x] #2 Sin torch.save del módulo completo: persistencia únicamente con state_dict()
- [x] #3 El orden de clases del Dataset coincide con el mapeo qubit a clase de task-9 y hay una prueba que lo verifica
- [x] #4 Ninguna decisión de arquitectura queda sin justificación trazable en la bitácora
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Crear HQCNN con CabeceraReduccion, TorchLayer(set_input_argument entradas), init_method=inicializar_pesos, loop por muestra en forward (PennyLane #4462), propiedad n_capas_vqc.
2. Funciones verificar_gradiente (float64, CPU) y contar_parametros_por_bloque.
3. Pruebas en tests/test_hqcnn.py (10 casos: forma, TorchLayer, acotación, gradiente numérico, eval backbone, state_dict, mapeo clases, Trainer 1 época, conteo).
4. Hallazgo en h1_arquitectura.tex con tabla de parámetros y verificación de gradiente.
5. pytest completo y cierre de criterios.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Trampas conocidas.**

- Las claves del diccionario de formas de `TorchLayer` deben coincidir **exactamente** con los nombres de los argumentos del QNode (aquí `pesos`), y el primer argumento del QNode debe ser el de entradas. Si no coinciden, el error aparece lejos del origen y es difícil de leer.
- `TorchLayer` procesa lotes iterando internamente cuando el dispositivo no admite difusión de lotes. Con `parameter-shift` el costo crece con el tamaño del lote: **esta** es la fuente del cuello de botella que task-11 debe medir antes de lanzar 60 corridas.
- La verificación numérica del gradiente en `float32` es inestable: hacerla en CPU y, si es posible, en `float64`, o declarar una tolerancia laxa. Una prueba que falla de forma intermitente es peor que no tenerla.
- `torch.no_grad()` alrededor del backbone ahorra memoria, pero hay que comprobar que el gradiente **sí** fluye hacia `reduccion` y `vqc`. La prueba numérica del punto 2 es precisamente esa comprobación.
- No aplicar `softmax` antes de `CrossEntropyLoss`. Los valores esperados son logits de escala pequeña y eso achata la distribución; aplicar softmax dos veces la achata aún más y el modelo parece no aprender.
- No usar `torch.save(modelo)`: `TorchLayer` no serializa de forma portable. Solo `state_dict()`.
- Si se añade un factor de escala aprendible sobre los logits, es un cambio de arquitectura: debe aplicarse a **todas** las corridas o la comparación entre celdas del diseño factorial deja de ser válida.
- El orden de las clases del `Dataset` (alfabético por nombre de carpeta) debe coincidir con `MAPEO_QUBIT_CLASE` de task-9. Si no coincide, todo entrena bien y las métricas por clase quedan permutadas.

**Frontera de responsabilidades.** Este módulo resuelve el modo `eval()` del backbone porque es un detalle de **su** arquitectura. El `Trainer` no debe saber nada de ello: así se preserva DIP y no reaparece la rama condicional por tipo de modelo.

Implementado HQCNN con CabeceraReduccion, TorchLayer(set_input_argument entradas), init_method=inicializar_pesos, loop por muestra (PennyLane #4462). 12 tests test_hqcnn.py + 76 tests regresión. Gradiente: analítico=8.18e-2 vs numérico=8.17e-2. Parámetros: 4007548 congelados + 5124 reducción + 48 VQC.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modelo híbrido HQCNN end-to-end en src/models/hqcnn.py con integración TorchLayer, acotación tanh·π, verificación numérica de gradiente y conteo por bloque. Hallazgo en h1_arquitectura.tex. Verificado con uv run pytest tests/test_hqcnn.py -q (12 passed) y uv run pytest tests/ -q (76 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
