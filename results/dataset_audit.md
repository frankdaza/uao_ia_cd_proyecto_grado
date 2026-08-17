# Auditoría del Brain Tumor MRI Dataset

Generado por `uv run python -m src.data.audit`.

## Resumen ejecutivo

- Imágenes inventariadas: **7023**
- Imágenes utilizables (no excluidas): **6726**
- Imágenes excluidas: **297**
- Total declarado en anteproyecto: **7023**

## Extensiones descubiertas

| extension | conteo |
| :--- | ---: |
| .jpg | 7023 |

## Conteos

- **Total de imágenes:** 7023
- **Declaradas en anteproyecto:** 7023
- **Diferencia:** 0

### Por clase

| clase | conteo |
| :--- | ---: |
| glioma | 1621 |
| meningioma | 1645 |
| notumor | 2000 |
| pituitary | 1757 |

### Por partición

| particion | conteo |
| :--- | ---: |
| Testing | 1311 |
| Training | 5712 |

### Clase × partición

| clase | Testing | Training |
| --- | --- | --- |
| glioma | 300 | 1321 |
| meningioma | 306 | 1339 |
| notumor | 405 | 1595 |
| pituitary | 300 | 1457 |

## Duplicados exactos (SHA-256)

| Categoría | Grupos (hash con n>1) |
| :--- | ---: |
| Intra clase | 194 |
| Entre clases (etiqueta contradictoria) | 0 |
| Training ↔ Testing (fuga potencial) | 79 |

**Limitación:** el hash byte a byte no detecta la misma imagen re-codificada con distinta compresión JPEG.

## Imágenes corruptas o truncadas

Total: **0**

No se detectaron imágenes corruptas.

## Exclusiones aplicadas

| motivo_exclusion | conteo |
| :--- | ---: |
| duplicado_exacto | 194 |
| fuga_train_test | 103 |

## Heterogeneidad del dataset

### Modos de color

| modo | conteo |
| :--- | ---: |
| L | 3093 |
| P | 1 |
| RGB | 3926 |
| RGBA | 3 |

### Resoluciones más frecuentes (top 15)

| resolucion | conteo |
| :--- | ---: |
| 512×512 | 4742 |
| 225×225 | 332 |
| 630×630 | 90 |
| 236×236 | 81 |
| 201×251 | 58 |
| 228×221 | 51 |
| 232×217 | 50 |
| 300×168 | 49 |
| 442×442 | 46 |
| 150×198 | 44 |
| 200×252 | 43 |
| 428×417 | 42 |
| 227×222 | 39 |
| 173×201 | 36 |
| 206×244 | 35 |

**Implicación para TASK-5:** las imágenes en escala de grises (modo `L`) deben convertirse a 3 canales antes de la normalización ImageNet; todas las imágenes requieren redimensionado a 224×224.

## Decisión sobre `Testing/`

Unir `Training/` y `Testing/` al conjunto completo para validación cruzada estratificada k=5 (decisión D1, TASK-6). No se reserva holdout externo: el anteproyecto prescribe k-fold, no evaluación sobre el split original de Kaggle. Las copias de Testing que duplican hashes de Training quedan excluidas (`motivo_exclusion=fuga_train_test`, n=103).

## Clases y particiones inesperadas

- Clases fuera del conjunto esperado: ninguna
- Particiones inesperadas: ninguna
