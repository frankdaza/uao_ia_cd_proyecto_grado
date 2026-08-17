"""Escritura de registros a CSV, wandb e historial JSON (TASK-4)."""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path

import wandb

from src.config import ExperimentConfig
from src.logging.records import COLUMNAS_CSV, EpochRecord, RunRecord, aplanar

logger = logging.getLogger(__name__)


def obtener_commit_sha() -> str:
    """Devuelve el SHA corto del commit actual, con sufijo ``-dirty`` si aplica.

    Returns
    -------
    str
        Identificador de versión del código, o ``"unknown"`` si git no está disponible.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        sucio = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if sucio:
            return f"{sha}-dirty"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _verificar_cabecera_csv(ruta: Path) -> None:
    """Compara la cabecera existente con ``COLUMNAS_CSV``."""
    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        cabecera_existente = next(lector, None)
    if cabecera_existente is None:
        return
    if tuple(cabecera_existente) != COLUMNAS_CSV:
        raise ValueError(
            "La cabecera de experiments.csv no coincide con COLUMNAS_CSV. "
            f"Esperado: {COLUMNAS_CSV}; encontrado: {tuple(cabecera_existente)}"
        )


def escribir_corrida_csv(registro: RunRecord, ruta: Path) -> None:
    """Añade una fila a ``experiments.csv`` tras validar el registro.

    Parameters
    ----------
    registro : RunRecord
        Corrida completa a persistir.
    ruta : Path
        Ruta al CSV (típicamente ``results/experiments.csv``).

    Notes
    -----
    Abre en modo append con ``newline=""``. Escribe la cabecera solo si el
    archivo no existe. Si la cabecera difiere de ``COLUMNAS_CSV``, aborta.
    """
    registro.validar()
    fila = aplanar(registro)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escribir_cabecera = not ruta.exists() or ruta.stat().st_size == 0
    if not escribir_cabecera:
        _verificar_cabecera_csv(ruta)

    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        if escribir_cabecera:
            escritor.writeheader()
        escritor.writerow(fila)

    logger.info("Corrida registrada en %s", ruta)


def registrar_corrida_wandb(registro: RunRecord, paso: int | None = None) -> None:
    """Envía la misma fila tidy del CSV a wandb.

    Parameters
    ----------
    registro : RunRecord
        Corrida completa a registrar.
    paso : int | None
        Paso opcional para ``wandb.log``.

    Notes
    -----
    No invoca ``wandb.init()``: el orquestador de la corrida debe inicializar
    la sesión. Respeta ``WANDB_MODE=offline`` del entorno.
    """
    fila = aplanar(registro)
    if paso is None:
        wandb.log(fila)
    else:
        wandb.log(fila, step=paso)


def nombre_historial(registro: RunRecord) -> str:
    """Construye el nombre de archivo JSON del historial por época.

    Parameters
    ----------
    registro : RunRecord
        Corrida cuyo historial se va a persistir.

    Returns
    -------
    str
        Nombre con modelo, fracción, fold y semilla.
    """
    fraccion = str(registro.data_fraction).replace(".", "p")
    base = f"{registro.modelo}_{fraccion}_f{registro.fold}_s{registro.semilla}"
    if registro.n_capas_vqc is not None:
        return f"{base}_L{registro.n_capas_vqc}.json"
    return f"{base}.json"


def escribir_historial_json(
    registro: RunRecord,
    historial: list[EpochRecord],
    cfg: ExperimentConfig,
) -> Path:
    """Persiste el historial por época en ``results/history/``.

    Parameters
    ----------
    registro : RunRecord
        Corrida asociada al historial.
    historial : list[EpochRecord]
        Métricas por época.
    cfg : ExperimentConfig
        Configuración con la ruta raíz de resultados.

    Returns
    -------
    Path
        Ruta del archivo JSON escrito.
    """
    for epoca in historial:
        epoca.validar()

    directorio = cfg.raiz_resultados / "history"
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre_historial(registro)
    payload = [asdict(epoca) for epoca in historial]
    ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Historial registrado en %s", ruta)
    return ruta


def corrida_existe(registro: RunRecord, ruta: Path) -> bool:
    """Indica si la celda ``(modelo, fracción, fold, semilla)`` ya está en el CSV.

    Parameters
    ----------
    registro : RunRecord
        Corrida a consultar.
    ruta : Path
        Ruta a ``experiments.csv``.

    Returns
    -------
    bool
        ``True`` si ya existe una fila con la misma clave de identidad.
    """
    if not ruta.exists():
        return False

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            if (
                fila.get("modelo") == registro.modelo
                and float(fila.get("data_fraction", -1)) == registro.data_fraction
                and int(fila.get("fold", -1)) == registro.fold
                and int(fila.get("semilla", -1)) == registro.semilla
            ):
                return True
    return False
