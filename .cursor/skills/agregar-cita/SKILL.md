---
description: Flujo de trabajo para agregar sistemáticamente nuevas referencias bibliográficas al archivo BibTeX de la tesis.
---

# Agregar Citaciones Bibliográficas (LaTeX & BibTeX)

**Objetivo:** Esta habilidad permite al Agente de Inteligencia Artificial modificar el documento principal de LaTeX para insertar literatura de respaldo para afirmaciones técnicas de Quantum Machine Learning y Neuroimagen.

> [!IMPORTANTE]
> Todas las comunicaciones y propuestas hechas usando esta habilidad deben redactarse en **Español Latinoamericano**, evitando anglicismos innecesarios en el contenido escrito de la Tesis.

## Procedimiento Paso a Paso

Si se requiere respaldar una afirmación (e.g., "El Angle Embedding permite codificar datos clásicos en qubits utilizando rotaciones locales"):

# Paso 1: Generación y Validación del Tipo BibTeX
El Agente pedirá la metadata o la identificará por el DOI, PDF abierto o instrucciones dadas.
Debe crear correctamente la estructura BibTeX usando de manera estandarizada los tipos:
`@article`, `@inproceedings`, `@book`, `@misc`.

```bibtex
@article{apellido2025titulo,
  author={Apellido, Nombre},
  title={Título del Paper de QML relacionado al Cerebro},
  journal={Nombre de la Revista (Q1)},
  year={2025},
  volume={1},
  pages={100--115},
  doi={10.XXXXXX}
}
```

# Paso 2: Inserción en Referencias.bib
El Agente inyectará el nuevo código en el archivo `docs/proyecto_de_grado/Referencias.bib`.
El Agente procurará insertar el bloque respetando el orden alfabético global guiándose por el identificador de la cita, u omitiendo el ordenamiento estricto pero garantizando que no se dupliquen identificadores únicos (como `apellido2025titulo`).

# Paso 3: Modificación del Archivo .tex
Identificar el archivo objetivo, usualmente `docs/proyecto_de_grado/Anteproyecto - Frank Daza.tex`, localizar el párrafo a respaldar y colocar estricta y únicamente el comando de LaTeX para citaciones al final de la frase, antes del punto final.

```latex
Este hallazgo se validó utilizando el marco teórico de la computación cuántica variacional~\cite{apellido2025titulo}.
```

# Paso 4: Corrección de Redacción (Opcional)
Si el Agente identifica que el párrafo es gramatical o estilísticamente inconsistente después de añadir la cita, propondrá un refinamiento del párrafo original, garantizando la rigurosidad científica exigida en Colombia y la Región.

## Cómo Invocar Esta Habilidad
**Tú**: "Agrega una cita a este párrafo sobre el teorema de Ansatz, aquí está el paper: [URL/DOI]"
