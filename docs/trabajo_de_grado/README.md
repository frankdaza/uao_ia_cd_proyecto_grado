# Documentación LaTeX del Trabajo de Grado

Carpeta de la tesis final y la bitácora metodológica de hallazgos.

## Artefactos

| Archivo / carpeta | Rol |
| :--- | :--- |
| `preambulo_uao.tex` | Preámbulo UAO copiado del anteproyecto (paquetes, geometría, `natbib` + `apalike`, `listings`, encabezados). |
| `Bitacora Metodologica de Hallazgos.tex` | Documento vivo que acumula hallazgos por fase conforme se ejecutan las tareas del backlog. |
| `Trabajo de Grado - Frank Daza.tex` | Esqueleto del documento final (A12), alimentado por la bitácora en la tarea 17. |
| `hallazgos/h0_fundamentos.tex` … `h3_analisis.tex` | Fragmentos `\input{}` por fase; una subsección por tarea con evidencia. |
| `capitulos/` | Esqueleto vacío del documento final (introducción, marco, método, resultados, discusión, conclusiones). |
| `Figuras/` | Figuras propias del trabajo de grado, incluido el logo UAO. |
| `../proyecto_de_grado/Referencias.bib` | Bibliografía **compartida**; prohibido duplicar entradas. |

## Flujo documental

```
Anteproyecto (proyecto_de_grado/)
    → Bitácora de hallazgos (esta carpeta)
        → Trabajo de Grado (tarea 17)
            → Artículo científico (tarea 18)
            → Ponencia (tarea 19)
```

El anteproyecto es **prospectivo** (problema, marco, diseño).
La bitácora es el **registro empírico** de lo ejecutado, con trazabilidad
`task-N` ↔ artefacto en `results/`.
Al llegar a A12, la redacción se convierte en síntesis de evidencia ya escrita.

## Convención de trazabilidad

Toda tarea del backlog que produzca evidencia debe registrar un hallazgo con:

```latex
\label{hallazgo:task-N}
```

en el fragmento `hallazgos/` de su fase. Una tarea sin esta etiqueta se considera
no documentada, aunque haya generado CSV en `results/`.

## Política de bibliografía

- **Única fuente:** `docs/proyecto_de_grado/Referencias.bib`
- **Prohibido** crear un `.bib` local ni duplicar entradas.
- **Nuevas citas:** usar el skill `agregar-cita` sobre el archivo compartido.
- **Motor:** `natbib` + estilo `apalike`.

## Compilación

Ejecutar **desde esta carpeta** (`docs/trabajo_de_grado/`):

```bash
cd docs/trabajo_de_grado

# Bitácora
pdflatex "Bitacora Metodologica de Hallazgos.tex"
bibtex "Bitacora Metodologica de Hallazgos"
pdflatex "Bitacora Metodologica de Hallazgos.tex"
pdflatex "Bitacora Metodologica de Hallazgos.tex"

# Trabajo de grado (cuando tenga contenido)
pdflatex "Trabajo de Grado - Frank Daza.tex"
bibtex "Trabajo de Grado - Frank Daza"
pdflatex "Trabajo de Grado - Frank Daza.tex"
pdflatex "Trabajo de Grado - Frank Daza.tex"
```

**Importante:**

- Los nombres de archivo con espacios requieren comillas en la línea de comandos.
- `bibtex` debe ejecutarse con el directorio de trabajo en `docs/trabajo_de_grado/`
  para que `\bibliography{../proyecto_de_grado/Referencias}` resuelva correctamente.
- La tercera pasada de `pdflatex` resuelve referencias cruzadas (`\ref{hallazgo:task-N}`).
- Los auxiliares LaTeX (`.aux`, `.bbl`, `.log`) están en `.gitignore` y no se versionan.

## Mapeo de fragmentos por fase

| Fragmento | Tareas |
| :--- | :--- |
| `h0_fundamentos.tex` | 1, 2, 3, 4, 5, 6 |
| `h1_arquitectura.tex` | 7, 8, 9, 10, 11 |
| `h2_experimentacion.tex` | 12, 13 |
| `h3_analisis.tex` | 14, 15, 16 |
