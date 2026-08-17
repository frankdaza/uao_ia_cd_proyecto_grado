"""Auditoría del Brain Tumor MRI Dataset (TASK-3 / A4)."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)

TOTAL_DECLARADO = 7023
CLASES_ESPERADAS = frozenset({"glioma", "meningioma", "notumor", "pituitary"})
PARTICIONES_ESPERADAS = frozenset({"Testing", "Training"})
COLUMNAS_MANIFIESTO = [
    "ruta_relativa",
    "particion_origen",
    "clase",
    "modo",
    "ancho",
    "alto",
    "corrupta",
    "sha256",
    "excluida",
    "motivo_exclusion",
]


@dataclass(frozen=True, slots=True)
class ResumenDuplicados:
    """Conteos de duplicados exactos por categoría.

    Attributes
    ----------
    intra_clase : int
        Grupos con mismo hash y misma clase.
    entre_clases : int
        Grupos con mismo hash y clases distintas (etiqueta contradictoria).
    train_test : int
        Grupos con mismo hash presentes en Training y Testing (fuga potencial).
    """

    intra_clase: int
    entre_clases: int
    train_test: int


def descubrir_extensiones(raiz: Path) -> dict[str, int]:
    """Inventaría las extensiones de archivo presentes bajo la raíz del dataset.

    Parameters
    ----------
    raiz : Path
        Directorio raíz que contiene ``Training/`` y ``Testing/``.

    Returns
    -------
    dict[str, int]
        Conteo por extensión en minúsculas (p. ej. ``{".jpg": 7023}``).
    """
    conteo: Counter[str] = Counter()
    for ruta in raiz.rglob("*"):
        if ruta.is_file():
            conteo[ruta.suffix.lower()] += 1
    return dict(sorted(conteo.items()))


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    """Calcula el hash SHA-256 del contenido binario de una imagen.

    Parameters
    ----------
    ruta : Path
        Ruta al archivo de imagen.
    bloque : int
        Tamaño del bloque de lectura en bytes.

    Returns
    -------
    str
        Digest hexadecimal del contenido.
    """
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for trozo in iter(lambda: archivo.read(bloque), b""):
            digest.update(trozo)
    return digest.hexdigest()


def verificar_imagen(ruta: Path) -> tuple[str, int, int, bool]:
    """Verifica integridad y extrae metadatos de una imagen.

    ``Image.verify()`` invalida el objeto; por eso se reabre el archivo
    para leer modo y dimensiones. ``Image.load()`` detecta truncamiento.

    Parameters
    ----------
    ruta : Path
        Ruta al archivo de imagen.

    Returns
    -------
    tuple[str, int, int, bool]
        Modo PIL, ancho, alto y bandera de corrupción.
    """
    try:
        with Image.open(ruta) as imagen:
            imagen.verify()
        with Image.open(ruta) as imagen:
            imagen.load()
            return imagen.mode, imagen.width, imagen.height, False
    except Exception:
        return "", 0, 0, True


def _extensiones_imagen(extensiones: dict[str, int]) -> set[str]:
    """Devuelve extensiones de imagen conocidas presentes en el inventario."""
    candidatas = {ext for ext in extensiones if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    if candidatas:
        return candidatas
    return {ext for ext in extensiones if ext}


def construir_manifiesto(raiz: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recorre el dataset y devuelve una fila por imagen.

    El recorrido se ordena por ruta relativa para que los índices
    posicionales del manifiesto sean estables entre ejecuciones.

    Parameters
    ----------
    raiz : Path
        Raíz del dataset (``Training/`` y ``Testing/``).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Manifiesto y conteo de extensiones descubiertas.
    """
    extensiones = descubrir_extensiones(raiz)
    extensiones_img = _extensiones_imagen(extensiones)

    rutas: list[Path] = []
    for ext in sorted(extensiones_img):
        rutas.extend(sorted(raiz.rglob(f"*{ext}")))

    filas: list[dict[str, object]] = []
    for ruta in sorted(set(rutas)):
        relativa = ruta.relative_to(raiz)
        partes = relativa.parts
        if len(partes) < 3:
            logger.warning("Ruta fuera de estructura esperada: %s", relativa)
            continue

        particion, clase = partes[0], partes[1]
        modo, ancho, alto, corrupta = verificar_imagen(ruta)
        digest = sha256_archivo(ruta)

        filas.append(
            {
                "ruta_relativa": relativa.as_posix(),
                "particion_origen": particion,
                "clase": clase,
                "modo": modo,
                "ancho": ancho,
                "alto": alto,
                "corrupta": corrupta,
                "sha256": digest,
                "excluida": False,
                "motivo_exclusion": "",
            }
        )

    return pd.DataFrame(filas, columns=COLUMNAS_MANIFIESTO), extensiones


def analizar_duplicados(manifiesto: pd.DataFrame) -> ResumenDuplicados:
    """Clasifica duplicados exactos por hash en tres categorías.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Manifiesto con columna ``sha256``.

    Returns
    -------
    ResumenDuplicados
        Conteo de grupos duplicados por categoría.
    """
    grupos = manifiesto.groupby("sha256", sort=False)
    intra = entre = train_test = 0

    for _, grupo in grupos:
        if len(grupo) < 2:
            continue
        if grupo["clase"].nunique() > 1:
            entre += 1
        if grupo["clase"].nunique() == 1:
            intra += 1
        if grupo["particion_origen"].nunique() > 1:
            train_test += 1

    return ResumenDuplicados(intra_clase=intra, entre_clases=entre, train_test=train_test)


def aplicar_exclusiones(manifiesto: pd.DataFrame) -> pd.DataFrame:
    """Marca filas excluidas según reglas deterministas y auditables.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Manifiesto sin columna de exclusión aplicada.

    Returns
    -------
    pd.DataFrame
        Copia del manifiesto con ``excluida`` y ``motivo_exclusion`` actualizados.
    """
    df = manifiesto.copy()
    df["excluida"] = False
    df["motivo_exclusion"] = ""

    def _excluir(indices: pd.Index, motivo: str) -> None:
        pendientes = indices[~df.loc[indices, "excluida"]]
        df.loc[pendientes, "excluida"] = True
        df.loc[pendientes, "motivo_exclusion"] = motivo

    corruptas = df.index[df["corrupta"]]
    _excluir(corruptas, "corrupta")

    for _, grupo in df.groupby("sha256", sort=False):
        if len(grupo) < 2:
            continue

        if grupo["clase"].nunique() > 1:
            _excluir(grupo.index, "etiqueta_contradictoria")
            continue

        training = grupo[grupo["particion_origen"] == "Training"]
        testing = grupo[grupo["particion_origen"] == "Testing"]
        if not training.empty and not testing.empty:
            _excluir(testing.index, "fuga_train_test")

        activos = grupo[~df.loc[grupo.index, "excluida"]]
        if len(activos) < 2:
            continue

        ordenado = activos.sort_values("ruta_relativa")
        _excluir(ordenado.index[1:], "duplicado_exacto")

    return df


def _serie_a_markdown(serie: pd.Series, titulo_indice: str = "clave") -> str:
    """Convierte una Serie a tabla Markdown sin dependencia de tabulate."""
    lineas = [f"| {titulo_indice} | conteo |", "| :--- | ---: |"]
    for clave, valor in serie.items():
        lineas.append(f"| {clave} | {valor} |")
    return "\n".join(lineas)


def _crosstab_a_markdown(tabla: pd.DataFrame) -> str:
    """Convierte una tabla cruzada a Markdown."""
    columnas = ["clase", *tabla.columns.astype(str)]
    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join(["---"] * len(columnas)) + " |",
    ]
    for indice, fila in tabla.iterrows():
        celdas = [str(indice), *[str(fila[col]) for col in tabla.columns]]
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def _df_a_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame pequeño a tabla Markdown."""
    columnas = list(df.columns)
    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join(["---"] * len(columnas)) + " |",
    ]
    for _, fila in df.iterrows():
        lineas.append("| " + " | ".join(str(fila[col]) for col in columnas) + " |")
    return "\n".join(lineas)


def generar_figura(manifiesto: pd.DataFrame, ruta: Path) -> None:
    """Genera barras apiladas de distribución de clases por partición.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Manifiesto auditado (solo filas no excluidas para el gráfico).
    ruta : Path
        Ruta de salida del PNG.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    usable = manifiesto[~manifiesto["excluida"]]
    tabla = (
        usable.groupby(["clase", "particion_origen"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(sorted(CLASES_ESPERADAS))
    )

    fig, eje = plt.subplots(figsize=(8, 5))
    tabla.plot(kind="bar", stacked=True, ax=eje, colormap="tab10")
    eje.set_title("Distribución de clases por partición (imágenes no excluidas)")
    eje.set_xlabel("Clase")
    eje.set_ylabel("Número de imágenes")
    eje.legend(title="Partición")
    eje.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _tabla_conteos(manifiesto: pd.DataFrame) -> str:
    """Formatea tablas de conteo para el informe Markdown."""
    total = len(manifiesto)
    por_clase = manifiesto["clase"].value_counts().sort_index()
    por_particion = manifiesto["particion_origen"].value_counts().sort_index()
    cruzada = pd.crosstab(manifiesto["clase"], manifiesto["particion_origen"]).sort_index()

    lineas = [
        f"- **Total de imágenes:** {total}",
        f"- **Declaradas en anteproyecto:** {TOTAL_DECLARADO}",
        f"- **Diferencia:** {total - TOTAL_DECLARADO}",
        "",
        "### Por clase",
        "",
        _serie_a_markdown(por_clase, "clase"),
        "",
        "### Por partición",
        "",
        _serie_a_markdown(por_particion, "particion"),
        "",
        "### Clase × partición",
        "",
        _crosstab_a_markdown(cruzada),
    ]
    return "\n".join(lineas)


def _tabla_heterogeneidad(manifiesto: pd.DataFrame) -> str:
    """Resume modos de color y resoluciones del dataset."""
    modos = manifiesto["modo"].value_counts().sort_index()
    resoluciones = (
        manifiesto.assign(resolucion=manifiesto["ancho"].astype(str) + "×" + manifiesto["alto"].astype(str))
        .groupby("resolucion")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    return "\n".join(
        [
            "### Modos de color",
            "",
            _serie_a_markdown(modos, "modo"),
            "",
            "### Resoluciones más frecuentes (top 15)",
            "",
            _serie_a_markdown(resoluciones, "resolucion"),
            "",
            "**Implicación para TASK-5:** las imágenes en escala de grises (modo `L`) "
            "deben convertirse a 3 canales antes de la normalización ImageNet; "
            "todas las imágenes requieren redimensionado a 224×224.",
        ]
    )


def generar_informe_md(
    manifiesto: pd.DataFrame,
    extensiones: dict[str, int],
    duplicados: ResumenDuplicados,
    ruta: Path,
) -> str:
    """Escribe el informe de auditoría en Markdown.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Manifiesto con exclusiones aplicadas.
    extensiones : dict[str, int]
        Extensiones descubiertas en el dataset.
    duplicados : ResumenDuplicados
        Resumen de duplicados por categoría.
    ruta : Path
        Ruta de salida del informe.

    Returns
    -------
    str
        Texto del informe generado.
    """
    total = len(manifiesto)
    excluidas = int(manifiesto["excluida"].sum())
    utilizables = total - excluidas
    corruptas = manifiesto[manifiesto["corrupta"]]

    decision_testing = (
        "Unir `Training/` y `Testing/` al conjunto completo para validación cruzada "
        "estratificada k=5 (decisión D1, TASK-6). No se reserva holdout externo: "
        "el anteproyecto prescribe k-fold, no evaluación sobre el split original de Kaggle. "
        "Las copias de Testing que duplican hashes de Training quedan excluidas "
        f"(`motivo_exclusion=fuga_train_test`, n={int((manifiesto['motivo_exclusion'] == 'fuga_train_test').sum())})."
    )

    texto = f"""# Auditoría del Brain Tumor MRI Dataset

Generado por `uv run python -m src.data.audit`.

## Resumen ejecutivo

- Imágenes inventariadas: **{total}**
- Imágenes utilizables (no excluidas): **{utilizables}**
- Imágenes excluidas: **{excluidas}**
- Total declarado en anteproyecto: **{TOTAL_DECLARADO}**

## Extensiones descubiertas

{_serie_a_markdown(pd.Series(extensiones, name="conteo"), "extension")}

## Conteos

{_tabla_conteos(manifiesto)}

## Duplicados exactos (SHA-256)

| Categoría | Grupos (hash con n>1) |
| :--- | ---: |
| Intra clase | {duplicados.intra_clase} |
| Entre clases (etiqueta contradictoria) | {duplicados.entre_clases} |
| Training ↔ Testing (fuga potencial) | {duplicados.train_test} |

**Limitación:** el hash byte a byte no detecta la misma imagen re-codificada con distinta compresión JPEG.

## Imágenes corruptas o truncadas

Total: **{len(corruptas)}**

"""
    if corruptas.empty:
        texto += "No se detectaron imágenes corruptas.\n"
    else:
        texto += _df_a_markdown(corruptas[["ruta_relativa", "particion_origen", "clase"]])
        texto += "\n"

    exclusiones = (
        manifiesto[manifiesto["excluida"]]
        .groupby("motivo_exclusion")
        .size()
        .sort_values(ascending=False)
    )
    texto += f"""
## Exclusiones aplicadas

{_serie_a_markdown(exclusiones, "motivo_exclusion")}

## Heterogeneidad del dataset

{_tabla_heterogeneidad(manifiesto)}

## Decisión sobre `Testing/`

{decision_testing}

## Clases y particiones inesperadas

- Clases fuera del conjunto esperado: {sorted(set(manifiesto['clase']) - CLASES_ESPERADAS) or 'ninguna'}
- Particiones inesperadas: {sorted(set(manifiesto['particion_origen']) - PARTICIONES_ESPERADAS) or 'ninguna'}
"""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return texto


def ejecutar_auditoria(cfg: ExperimentConfig) -> pd.DataFrame:
    """Orquesta la auditoría completa y persiste artefactos en ``results/``.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con rutas de datos y resultados.

    Returns
    -------
    pd.DataFrame
        Manifiesto final con exclusiones aplicadas.
    """
    cfg.ensure_layout()
    raiz = cfg.raiz_datos.resolve()

    if not raiz.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {cfg.raiz_datos}. "
            "Enlaza notebooks/data/ a data/brain_tumor_mri/."
        )

    logger.info("Auditando dataset en %s", cfg.raiz_datos)
    manifiesto, extensiones = construir_manifiesto(raiz)
    duplicados = analizar_duplicados(manifiesto)
    manifiesto = aplicar_exclusiones(manifiesto)

    ruta_manifiesto = cfg.raiz_resultados / "dataset_manifest.csv"
    ruta_informe = cfg.raiz_resultados / "dataset_audit.md"
    ruta_figura = cfg.raiz_figuras / "distribucion_clases.png"

    manifiesto.to_csv(ruta_manifiesto, index=False)
    generar_informe_md(manifiesto, extensiones, duplicados, ruta_informe)
    generar_figura(manifiesto, ruta_figura)

    logger.info("Manifiesto: %s (%d filas)", ruta_manifiesto, len(manifiesto))
    logger.info("Informe: %s", ruta_informe)
    logger.info("Figura: %s", ruta_figura)
    logger.info(
        "Duplicados — intra: %d, entre clases: %d, train/test: %d",
        duplicados.intra_clase,
        duplicados.entre_clases,
        duplicados.train_test,
    )
    logger.info(
        "Utilizables: %d / %d (excluidas: %d)",
        int((~manifiesto["excluida"]).sum()),
        len(manifiesto),
        int(manifiesto["excluida"].sum()),
    )

    return manifiesto


def main() -> None:
    """Punto de entrada: ``uv run python -m src.data.audit``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ejecutar_auditoria(ExperimentConfig())


if __name__ == "__main__":
    main()
