---
name: agregar-cita
description: Inserta referencias bibliográficas validadas en Referencias.bib y cita en LaTeX con natbib. Usar al agregar citas, DOIs, papers de QML o neuroimagen, o cuando falte respaldo bibliográfico en el .tex.
---

# Agregar Citaciones Bibliográficas (LaTeX + BibTeX)

Inserta literatura de respaldo para afirmaciones técnicas de QML y neuroimagen. Comunicación en **Español Latinoamericano**.

## Archivos objetivo

- BibTeX: [`docs/proyecto_de_grado/Referencias.bib`](docs/proyecto_de_grado/Referencias.bib)
- LaTeX: [`docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex`](docs/proyecto_de_grado/Anteproyecto%20-%20Frank%20Daza.tex) (u otro `.tex` indicado)

## Procedimiento

### Paso 1: Obtener metadata verificable

Prioridad de fuentes:

1. **DOI** → consultar Crossref: `https://api.crossref.org/works/{doi}`
2. Metadata proporcionada por el usuario (PDF, URL de publisher).
3. Si no hay fuente verificable: **detener** y pedir al humano el DOI o PDF. No inventar entradas.

### Paso 2: Generar clave BibTeX

Formato: `apellidoAñoPalabraClave` (ej. `bergholm2018pennylane`). Verificar que no exista en `Referencias.bib`.

Tipos permitidos: `@article`, `@inproceedings`, `@book`, `@misc`.

```bibtex
@article{bergholm2018pennylane,
  author  = {Bergholm, Ville and others},
  title   = {PennyLane: Automatic differentiation of hybrid quantum-classical computations},
  journal = {arXiv preprint arXiv:1811.04968},
  year    = {2018},
  doi     = {10.48550/arXiv.1811.04968}
}
```

### Paso 3: Insertar en Referencias.bib

- No duplicar claves existentes.
- Preferir orden alfabético por clave; si no es posible, insertar sin romper entradas vecinas.

### Paso 4: Citar en el .tex

Colocar `~\cite{clave}` al final de la frase, antes del punto:

```latex
El gradiente cuántico se obtiene mediante la regla de cambio de parámetros~\cite{bergholm2018pennylane}.
```

### Paso 5: Revisión opcional

Si la cita altera la fluidez del párrafo, proponer un refinamiento manteniendo rigor académico.

## Prohibiciones

- No inventar DOIs, volúmenes, páginas ni autores.
- No migrar a `biblatex`; mantener `natbib` + `apalike`.
- No modificar el preámbulo LaTeX.

## Invocación

> "Agrega una cita a este párrafo sobre barren plateaus. DOI: 10.xxxx/yyyy"
