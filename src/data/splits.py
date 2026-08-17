"""Protocolo de particiones: k-fold estratificado y fracciones anidadas (TASK-6 / A7)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.config import ExperimentConfig
from src.data.dataset import MAPEO_CLASE, cargar_manifiesto_utilizable
from src.logging.records import CLASES_ORDEN

logger = logging.getLogger(__name__)

FRACCIONES: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
TOLERANCIA_PROPORCION: float = 0.02
VERSION_SPLITS: int = 1
ORDENAMIENTO_MANIFIESTO: str = "ruta_relativa mergesort"
LIMITACION_FOLDS: str = (
    "Los 5 folds comparten observaciones entre sí: no son 5 muestras independientes. "
    "Esta limitación condiciona el análisis estadístico posterior (ANOVA, TASK-15)."
)

_CLAVE_FRACCION: dict[float, str] = {
    0.10: "0.10",
    0.25: "0.25",
    0.50: "0.50",
    1.00: "1.00",
    1.0: "1.00",
}


def _clave_fraccion(fraccion: float) -> str:
    """Convierte una fracción numérica a la clave canónica del JSON."""
    clave = _CLAVE_FRACCION.get(fraccion)
    if clave is None:
        clave = f"{fraccion:.2f}"
        if clave not in {v for v in _CLAVE_FRACCION.values()}:
            raise ValueError(f"Fracción no soportada: {fraccion}")
    return clave


def hash_manifiesto_utilizable(manifiesto: pd.DataFrame) -> str:
    """Calcula el hash SHA-256 del manifiesto utilizable ordenado.

    Parameters
    ----------
    manifiesto : pd.DataFrame
        Manifiesto ya filtrado y ordenado por ``ruta_relativa``.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.
    """
    columnas = ["ruta_relativa", "clase"]
    contenido = manifiesto[columnas].to_csv(index=False)
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def _conteos_por_clase(etiquetas: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    """Cuenta imágenes por clase para un subconjunto de índices."""
    if len(indices) == 0:
        return {clase: 0 for clase in CLASES_ORDEN}
    valores, conteos = np.unique(etiquetas[indices], return_counts=True)
    inverso = {indice: nombre for nombre, indice in MAPEO_CLASE.items()}
    resultado = {clase: 0 for clase in CLASES_ORDEN}
    for valor, conteo in zip(valores, conteos, strict=True):
        resultado[inverso[int(valor)]] = int(conteo)
    return resultado


def _submuestreo_anidado(
    indices: np.ndarray,
    etiquetas: np.ndarray,
    semilla: int,
) -> dict[str, list[int]]:
    """Submuestrea en cascada descendente para garantizar anidamiento.

    Parameters
    ----------
    indices : np.ndarray
        Índices de entrenamiento del fold.
    etiquetas : np.ndarray
        Etiquetas de todo el conjunto, indexables por ``indices``.
    semilla : int
        Semilla del submuestreo.

    Returns
    -------
    dict[str, list[int]]
        Índices de entrenamiento por fracción, con el 10 % contenido en el
        25 %, este en el 50 % y este en el 100 %.
    """
    n_total = len(indices)
    actual = indices.copy()
    resultado: dict[str, list[int]] = {"1.00": sorted(actual.tolist())}
    for fraccion in (0.50, 0.25, 0.10):
        objetivo = int(round(fraccion * n_total))
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=objetivo,
            random_state=semilla,
        )
        (seleccion, _), = sss.split(np.zeros(len(actual)), etiquetas[actual])
        actual = actual[seleccion]
        resultado[f"{fraccion:.2f}"] = sorted(actual.tolist())
    return resultado


def construir_particiones(
    etiquetas: np.ndarray,
    *,
    n_folds: int,
    semilla: int,
) -> list[dict[str, Any]]:
    """Construye todas las particiones del proyecto según la decisión D1.

    Parameters
    ----------
    etiquetas : np.ndarray
        Etiquetas enteras indexadas por posición en el manifiesto ordenado.
    n_folds : int
        Número de folds estratificados.
    semilla : int
        Semilla global para ``StratifiedKFold``.

    Returns
    -------
    list[dict[str, Any]]
        Lista de folds con validación fija y entrenamiento anidado por fracción.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    folds: list[dict[str, Any]] = []
    for k, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(etiquetas)), etiquetas)):
        train_por_fraccion = _submuestreo_anidado(train_idx, etiquetas, semilla + k)
        conteos_train = {
            fraccion: _conteos_por_clase(etiquetas, np.array(indices))
            for fraccion, indices in train_por_fraccion.items()
        }
        folds.append(
            {
                "fold": k,
                "val": sorted(val_idx.tolist()),
                "train": train_por_fraccion,
                "conteos": {
                    "val": _conteos_por_clase(etiquetas, val_idx),
                    "train": conteos_train,
                },
            }
        )
    return folds


def _serializar_splits(artefacto: dict[str, Any]) -> str:
    """Serializa el artefacto de particiones de forma determinista."""
    return json.dumps(artefacto, sort_keys=True, separators=(",", ":"))


def generar_splits(cfg: ExperimentConfig) -> dict[str, Any]:
    """Genera y persiste ``results/splits.json`` desde el manifiesto auditado.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con rutas y semilla global.

    Returns
    -------
    dict[str, Any]
        Artefacto de particiones listo para serializar o consumir.
    """
    cfg.ensure_layout()
    ruta_manifiesto = cfg.raiz_resultados / "dataset_manifest.csv"
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(
            f"No se encontró el manifiesto en {ruta_manifiesto}. "
            "Ejecuta primero: uv run python -m src.data.audit"
        )

    manifiesto = cargar_manifiesto_utilizable(ruta_manifiesto)
    etiquetas = manifiesto["clase"].map(MAPEO_CLASE).to_numpy()
    manifest_hash = hash_manifiesto_utilizable(manifiesto)
    folds = construir_particiones(etiquetas, n_folds=cfg.n_folds, semilla=cfg.semilla)

    artefacto: dict[str, Any] = {
        "version": VERSION_SPLITS,
        "semilla": cfg.semilla,
        "n_folds": cfg.n_folds,
        "fracciones": list(FRACCIONES),
        "n_total": len(manifiesto),
        "manifest_hash": manifest_hash,
        "ordenamiento": ORDENAMIENTO_MANIFIESTO,
        "limitacion": LIMITACION_FOLDS,
        "folds": folds,
    }

    ruta_json = cfg.raiz_resultados / "splits.json"
    ruta_json.write_text(_serializar_splits(artefacto) + "\n", encoding="utf-8")
    _escribir_resumen_markdown(artefacto, cfg.raiz_resultados / "splits_summary.md")

    logger.info("Particiones guardadas en %s (%d folds)", ruta_json, len(folds))
    logger.info("Hash del manifiesto: %s", manifest_hash)
    return artefacto


def _escribir_resumen_markdown(artefacto: dict[str, Any], ruta: Path) -> None:
    """Escribe un resumen legible de conteos por fold y fracción."""
    lineas = [
        "# Resumen de particiones (TASK-6)",
        "",
        f"- Semilla: **{artefacto['semilla']}**",
        f"- Folds: **{artefacto['n_folds']}**",
        f"- Total de imágenes: **{artefacto['n_total']}**",
        f"- Hash del manifiesto: `{artefacto['manifest_hash']}`",
        "",
        f"**Limitación:** {artefacto['limitacion']}",
        "",
    ]
    for fold in artefacto["folds"]:
        k = fold["fold"]
        lineas.append(f"## Fold {k}")
        lineas.append("")
        lineas.append(f"- Validación: {len(fold['val'])} imágenes")
        lineas.append("")
        lineas.append("| Fracción | Total train | glioma | meningioma | pituitary | notumor |")
        lineas.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for fraccion in ("0.10", "0.25", "0.50", "1.00"):
            conteos = fold["conteos"]["train"][fraccion]
            total = sum(conteos.values())
            lineas.append(
                f"| {fraccion} | {total} | {conteos['glioma']} | {conteos['meningioma']} "
                f"| {conteos['pituitary']} | {conteos['notumor']} |"
            )
        lineas.append("")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def cargar_splits(
    ruta: Path,
    *,
    ruta_manifiesto: Path | None = None,
    validar_hash: bool = True,
) -> dict[str, Any]:
    """Carga ``splits.json`` y valida el hash del manifiesto si se solicita.

    Parameters
    ----------
    ruta : Path
        Ruta a ``splits.json``.
    ruta_manifiesto : Path | None
        Manifiesto actual para validar integridad. Si es ``None``, se infiere
        desde el directorio padre de ``ruta``.
    validar_hash : bool
        Si es ``True``, falla cuando el manifiesto cambió desde la generación.

    Returns
    -------
    dict[str, Any]
        Artefacto de particiones cargado desde disco.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo de particiones.
    ValueError
        Si el hash del manifiesto no coincide.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}. Ejecuta: uv run python -m src.data.splits")

    artefacto: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    if not validar_hash:
        return artefacto

    if ruta_manifiesto is None:
        ruta_manifiesto = ruta.parent / "dataset_manifest.csv"
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No se encontró el manifiesto en {ruta_manifiesto}")

    manifiesto = cargar_manifiesto_utilizable(ruta_manifiesto)
    hash_actual = hash_manifiesto_utilizable(manifiesto)
    hash_guardado = artefacto.get("manifest_hash")
    if hash_actual != hash_guardado:
        raise ValueError(
            "El manifiesto cambió desde la generación de splits.json. "
            f"Esperado: {hash_guardado}; actual: {hash_actual}. "
            "Regenera con: uv run python -m src.data.splits"
        )
    return artefacto


def obtener_indices(
    splits: dict[str, Any],
    fold: int,
    fraccion: float,
) -> tuple[list[int], list[int]]:
    """Devuelve los índices de entrenamiento y validación para una corrida.

    Parameters
    ----------
    splits : dict[str, Any]
        Artefacto cargado desde ``splits.json``.
    fold : int
        Índice del fold (0 a ``n_folds - 1``).
    fraccion : float
        Fracción de escasez del entrenamiento (0.10, 0.25, 0.50 o 1.00).

    Returns
    -------
    tuple[list[int], list[int]]
        Par ``(train_idx, val_idx)`` sobre el manifiesto ordenado.

    Raises
    ------
    KeyError
        Si el fold o la fracción no existen en el artefacto.
    """
    clave = _clave_fraccion(fraccion)
    for fold_info in splits["folds"]:
        if fold_info["fold"] == fold:
            return fold_info["train"][clave], fold_info["val"]
    raise KeyError(f"Fold {fold} no encontrado en splits.json")


def main() -> None:
    """Punto de entrada: ``uv run python -m src.data.splits``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generar_splits(ExperimentConfig())


if __name__ == "__main__":
    main()
