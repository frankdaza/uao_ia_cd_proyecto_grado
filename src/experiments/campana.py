"""Campaña experimental factorial k-fold en escenarios de escasez (TASK-13 / A8)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pennylane as qml
import torch

from src.config import ExperimentConfig, cargar_hparams_congelados, n_capas_congelada
from src.data.splits import FRACCIONES, cargar_splits
from src.experiments.ablacion_L import get_device_hqcnn
from src.experiments.baselines import FRACCION_BASELINE, MODELOS_BASELINE, verificar_indices_fold
from src.logging.records import COLUMNAS_CSV, RunRecord
from src.logging.sinks import (
    escribir_corrida_csv,
    escribir_historial_json,
    nombre_historial,
    obtener_commit_sha,
)
from src.models.factory import build_model
from src.train.dataloading import construir_loaders_para_fold
from src.train.trainer import Trainer
from src.utils.device import get_device, log_dispositivo
from src.utils.logging import configurar_logging_cli
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

MODELOS: tuple[str, ...] = ("efficientnet_b0", "resnet50", "hqcnn")
FRACCION_COMPLETA: float = 1.0
N_CELDAS_TOTAL: int = len(MODELOS) * len(FRACCIONES) * 5
N_CELDAS_BASELINE_TASK12: int = len(MODELOS_BASELINE) * 5
RUTA_ESTADO: str = "campana_estado.json"
RUTA_HPARAMS: str = "selected_hparams.json"

DISPOSITIVO_CAMPANA: str = "cuda"
RUTA_HISTORICO_NO_CUDA: str = "historico_mps.csv"
RUTA_PRUEBAS_INFORMALES: str = "pruebas_informales.csv"
DIR_HISTORIAL_ARCHIVADO: str = "history_mps"

ClaveCelda = tuple[str, float, int]
EstadoCelda = Literal["completada", "omitida", "fallida", "pendiente"]


class FilaEstadoCelda(TypedDict, total=False):
    """Entrada de estado por celda del diseño factorial."""

    modelo: str
    data_fraction: float
    fold: int
    estado: EstadoCelda
    motivo: str
    train_time_s: float
    timestamp: str


class ResumenCampana(TypedDict):
    """Conteo agregado del progreso de la campaña."""

    completadas: int
    omitidas: int
    pendientes: int
    fallidas: int


class CostoCampana(TypedDict):
    """Comparación de costo real frente a la estimación de TASK-11."""

    horas_reales: float
    horas_estimadas: float
    desviacion_pct: float | None


class EstadoCampana(TypedDict):
    """Artefacto completo de estado de ejecución."""

    celdas: list[FilaEstadoCelda]
    resumen: ResumenCampana
    costo: CostoCampana
    timestamp: str
    commit_sha: str


def orden_modelos(fraccion: float) -> tuple[str, ...]:
    """Devuelve el orden de ejecución de modelos para una fracción.

    Parameters
    ----------
    fraccion : float
        Fracción de datos del escenario de escasez.

    Returns
    -------
    tuple[str, ...]
        Orden fijo; en fracción 1.00 el HQCNN va al final (gestión de riesgo).
    """
    if fraccion >= FRACCION_COMPLETA:
        return ("efficientnet_b0", "resnet50", "hqcnn")
    return MODELOS


def dispositivo_para_modelo(nombre: str) -> torch.device:
    """Selecciona el dispositivo PyTorch según el tipo de modelo.

    Parameters
    ----------
    nombre : str
        Identificador del modelo.

    Returns
    -------
    torch.device
        ``get_device_hqcnn()`` para HQCNN; ``get_device()`` para baselines.
    """
    if nombre == "hqcnn":
        return get_device_hqcnn()
    return get_device()


def generar_celdas_design(
    cfg: ExperimentConfig,
    *,
    fraccion: float | None = None,
    modelo: str | None = None,
    fold: int | None = None,
) -> list[ClaveCelda]:
    """Enumera las celdas del diseño factorial según filtros opcionales.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con ``n_folds``.
    fraccion : float | None
        Si se indica, solo esa fracción.
    modelo : str | None
        Si se indica, solo ese modelo.
    fold : int | None
        Si se indica, solo ese fold.

    Returns
    -------
    list[ClaveCelda]
        Lista de tuplas ``(modelo, data_fraction, fold)``.
    """
    fracciones = FRACCIONES
    if fraccion is not None:
        if fraccion not in FRACCIONES:
            raise ValueError(f"Fraccion no soportada: {fraccion}. Opciones: {FRACCIONES}")
        fracciones = (fraccion,)

    modelos = MODELOS
    if modelo is not None:
        if modelo not in MODELOS:
            raise ValueError(f"Modelo no soportado: {modelo}. Opciones: {MODELOS}")
        modelos = (modelo,)

    folds = range(cfg.n_folds)
    if fold is not None:
        if fold < 0 or fold >= cfg.n_folds:
            raise ValueError(f"Fold fuera de rango: {fold}")
        folds = range(fold, fold + 1)

    celdas: list[ClaveCelda] = []
    modelos_filtrados = set(modelos)
    for fracc in fracciones:
        for nombre in orden_modelos(fracc):
            if nombre not in modelos_filtrados:
                continue
            for k in folds:
                celdas.append((nombre, fracc, k))
    return celdas


def _contar_baselines_al_100(ruta_csv: Path) -> int:
    """Cuenta celdas de líneas base al 100 % en ``experiments.csv``."""
    if not ruta_csv.exists():
        return 0

    conteo = 0
    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["modelo"] not in MODELOS_BASELINE:
                continue
            if float(fila["data_fraction"]) != FRACCION_BASELINE:
                continue
            conteo += 1
    return conteo


def _ruta_hparams(cfg: ExperimentConfig) -> Path:
    """Resuelve ``selected_hparams.json`` desde la configuración, no desde el CWD.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con la raíz de resultados.

    Returns
    -------
    Path
        Ruta al JSON de hiperparámetros congelados.

    Notes
    -----
    En Colab el cuaderno puede montar ``results/`` fuera del directorio de trabajo;
    resolver la ruta desde ``cfg`` evita que la campaña arranque con la ``L``
    equivocada o falle en silencio.
    """
    return cfg.raiz_resultados / RUTA_HPARAMS


def verificar_precondiciones(
    cfg: ExperimentConfig,
    *,
    exigir_baselines: bool = True,
) -> dict[str, Any]:
    """Comprueba artefactos previos antes de lanzar la campaña.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con rutas de resultados.
    exigir_baselines : bool
        Si ``True``, exige las celdas baseline al 100 % de TASK-12 en el CSV.
        Se desactiva cuando esas corridas fueron archivadas por heterogeneidad
        de hardware y se reejecutarán dentro de la propia campaña (decisión D3).
        La validación de ``L`` congelada y del hash de ``splits.json`` se
        mantiene en ambos casos.

    Returns
    -------
    dict[str, Any]
        Metadatos de precondiciones (``L``, decisión de presupuesto, etc.).

    Raises
    ------
    FileNotFoundError
        Si faltan ``selected_hparams.json`` o ``splits.json``.
    ValueError
        Si se exigen las celdas baseline de TASK-12 y no están completas.
    """
    cfg.ensure_layout()
    hparams = cargar_hparams_congelados(_ruta_hparams(cfg))
    l_congelada = int(hparams["n_capas"])

    ruta_splits = cfg.raiz_resultados / "splits.json"
    cargar_splits(ruta_splits, validar_hash=True)

    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    n_baselines = _contar_baselines_al_100(ruta_csv)
    if exigir_baselines and n_baselines < N_CELDAS_BASELINE_TASK12:
        raise ValueError(
            f"Faltan celdas baseline al 100%: {n_baselines}/{N_CELDAS_BASELINE_TASK12}. "
            "Ejecuta TASK-12 antes de la campaña, o usa --sin-baselines-previas "
            "si fueron archivadas para reejecutarse en CUDA."
        )

    presupuesto = hparams.get("presupuesto", {})
    decision = presupuesto.get("decision", "desconocida")
    if decision == "no-go":
        logger.warning(
            "TASK-11 registro decision=no-go (%.1f h estimadas). "
            "La campaña procede bajo go condicionado a sonda en Colab.",
            float(presupuesto.get("horas_campana_estimadas", 0.0)),
        )

    logger.info(
        "Precondiciones OK: L=%d, baselines al 100%%=%d, splits validados",
        l_congelada,
        n_baselines,
    )
    return {
        "n_capas": l_congelada,
        "n_baselines_100": n_baselines,
        "presupuesto": presupuesto,
    }


def _run_record_desde_fila(fila: dict[str, str]) -> RunRecord:
    """Reconstruye un ``RunRecord`` desde una fila del CSV."""
    n_capas_raw = fila.get("n_capas_vqc", "")
    n_capas_vqc = int(n_capas_raw) if n_capas_raw not in ("", None) else None
    return RunRecord(
        modelo=fila["modelo"],
        data_fraction=float(fila["data_fraction"]),
        fold=int(fila["fold"]),
        semilla=int(fila["semilla"]),
        dispositivo=fila["dispositivo"],
        n_train=int(fila["n_train"]),
        n_val=int(fila["n_val"]),
        epocas=int(fila["epocas"]),
        n_params_entrenables=int(fila["n_params_entrenables"]),
        n_capas_vqc=n_capas_vqc,
        commit_sha=fila["commit_sha"],
        timestamp=fila["timestamp"],
        accuracy_train=float(fila["accuracy_train"]),
        accuracy_val=float(fila["accuracy_val"]),
        loss_train=float(fila["loss_train"]),
        loss_val=float(fila["loss_val"]),
        f1_val_weighted=float(fila["f1_val_weighted"]),
        f1_val_macro=float(fila["f1_val_macro"]),
        sensibilidad_por_clase={
            "glioma": float(fila["sens_glioma"]),
            "meningioma": float(fila["sens_meningioma"]),
            "pituitary": float(fila["sens_pituitary"]),
            "notumor": float(fila["sens_notumor"]),
        },
        especificidad_por_clase={
            "glioma": float(fila["spec_glioma"]),
            "meningioma": float(fila["spec_meningioma"]),
            "pituitary": float(fila["spec_pituitary"]),
            "notumor": float(fila["spec_notumor"]),
        },
        train_time_s=float(fila["train_time_s"]),
        inference_ms_per_batch=float(fila["inference_ms_per_batch"]),
    )


def _cargar_filas_campana(ruta_csv: Path) -> list[dict[str, str]]:
    """Lee del CSV solo filas que pertenecen al diseño factorial TASK-13."""
    if not ruta_csv.exists():
        return []

    filas: list[dict[str, str]] = []
    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["modelo"] not in MODELOS:
                continue
            if float(fila["data_fraction"]) not in FRACCIONES:
                continue
            filas.append(fila)
    return filas


def _clave_celda(fila: dict[str, str]) -> ClaveCelda:
    """Extrae la clave de celda sin semilla."""
    return (fila["modelo"], float(fila["data_fraction"]), int(fila["fold"]))


def comparar_costo(
    cfg: ExperimentConfig,
    hparams: dict[str, Any] | None = None,
) -> CostoCampana:
    """Compara el costo real acumulado con la estimación de TASK-11.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con ruta de resultados.
    hparams : dict[str, Any] | None
        Hiperparámetros congelados; se cargan si es ``None``.

    Returns
    -------
    CostoCampana
        Horas reales, estimadas y desviación porcentual.
    """
    hparams = hparams or cargar_hparams_congelados(_ruta_hparams(cfg))
    horas_estimadas = float(hparams.get("presupuesto", {}).get("horas_campana_estimadas", 0.0))

    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    segundos_reales = sum(
        float(fila["train_time_s"]) for fila in _cargar_filas_campana(ruta_csv)
    )
    horas_reales = segundos_reales / 3600.0

    desviacion: float | None
    if horas_estimadas > 0:
        desviacion = ((horas_reales - horas_estimadas) / horas_estimadas) * 100.0
    else:
        desviacion = None

    return {
        "horas_reales": round(horas_reales, 4),
        "horas_estimadas": round(horas_estimadas, 4),
        "desviacion_pct": round(desviacion, 2) if desviacion is not None else None,
    }


def verificar_integridad(cfg: ExperimentConfig) -> dict[str, Any]:
    """Verifica que el diseño factorial esté completo y sin duplicados.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con rutas de resultados.

    Returns
    -------
    dict[str, Any]
        Resultado con ``ok``, conteos y listas de celdas faltantes o duplicadas.

    Raises
    ------
    AssertionError
        Si el diseño no cumple integridad (60 celdas únicas + historial).
    """
    cfg.ensure_layout()
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    filas = _cargar_filas_campana(ruta_csv)

    claves_vistas: dict[ClaveCelda, int] = {}
    for fila in filas:
        clave = _clave_celda(fila)
        claves_vistas[clave] = claves_vistas.get(clave, 0) + 1

    duplicadas = [clave for clave, n in claves_vistas.items() if n > 1]
    esperadas = set(generar_celdas_design(cfg))
    presentes = set(claves_vistas.keys())
    faltantes = sorted(esperadas - presentes)
    sobrantes = sorted(presentes - esperadas)

    historial_faltante: list[str] = []
    directorio_historial = cfg.raiz_resultados / "history"
    for fila in filas:
        clave = _clave_celda(fila)
        if clave in faltantes:
            continue
        registro = _run_record_desde_fila(fila)
        ruta_hist = directorio_historial / nombre_historial(registro)
        if not ruta_hist.is_file():
            historial_faltante.append(nombre_historial(registro))

    ok = (
        len(presentes) == N_CELDAS_TOTAL
        and not duplicadas
        and not faltantes
        and not historial_faltante
    )

    resultado = {
        "ok": ok,
        "n_celdas_presentes": len(presentes),
        "n_celdas_esperadas": N_CELDAS_TOTAL,
        "duplicadas": [list(c) for c in duplicadas],
        "faltantes": [list(c) for c in faltantes],
        "sobrantes": [list(c) for c in sobrantes],
        "historial_faltante": historial_faltante,
    }

    if not ok:
        raise AssertionError(
            f"Integridad de campaña fallida: {len(presentes)}/{N_CELDAS_TOTAL} celdas, "
            f"{len(duplicadas)} duplicadas, {len(faltantes)} faltantes, "
            f"{len(historial_faltante)} historiales ausentes"
        )

    logger.info("Integridad OK: %d celdas, sin duplicados, historial completo", N_CELDAS_TOTAL)
    return resultado


def _ruta_estado(cfg: ExperimentConfig) -> Path:
    return cfg.raiz_resultados / RUTA_ESTADO


def cargar_estado_campana(cfg: ExperimentConfig) -> EstadoCampana | None:
    """Carga el estado persistido de la campaña, si existe."""
    ruta = _ruta_estado(cfg)
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _construir_resumen(celdas: list[FilaEstadoCelda]) -> ResumenCampana:
    """Calcula conteos por estado a partir de la lista de celdas."""
    conteos: dict[EstadoCelda, int] = {
        "completada": 0,
        "omitida": 0,
        "pendiente": 0,
        "fallida": 0,
    }
    for celda in celdas:
        estado = celda.get("estado", "pendiente")
        conteos[estado] = conteos.get(estado, 0) + 1
    return {
        "completadas": conteos["completada"],
        "omitidas": conteos["omitida"],
        "pendientes": conteos["pendiente"],
        "fallidas": conteos["fallida"],
    }


def _indice_celda(celdas: list[FilaEstadoCelda], clave: ClaveCelda) -> int | None:
    """Busca el índice de una celda en la lista de estado."""
    for i, celda in enumerate(celdas):
        if (
            celda["modelo"] == clave[0]
            and celda["data_fraction"] == clave[1]
            and celda["fold"] == clave[2]
        ):
            return i
    return None


def guardar_estado_campana(
    cfg: ExperimentConfig,
    celdas: list[FilaEstadoCelda],
    *,
    hparams: dict[str, Any] | None = None,
) -> Path:
    """Persiste el estado de ejecución de la campaña.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con ruta de resultados.
    celdas : list[FilaEstadoCelda]
        Estado por celda.
    hparams : dict[str, Any] | None
        Hiperparámetros para comparación de costo.

    Returns
    -------
    Path
        Ruta del JSON escrito.
    """
    cfg.ensure_layout()
    payload: EstadoCampana = {
        "celdas": celdas,
        "resumen": _construir_resumen(celdas),
        "costo": comparar_costo(cfg, hparams=hparams),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "commit_sha": obtener_commit_sha(),
    }
    ruta = _ruta_estado(cfg)
    ruta.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Estado de campaña: completadas=%d omitidas=%d pendientes=%d fallidas=%d",
        payload["resumen"]["completadas"],
        payload["resumen"]["omitidas"],
        payload["resumen"]["pendientes"],
        payload["resumen"]["fallidas"],
    )
    return ruta


def _inicializar_estado_celdas(cfg: ExperimentConfig) -> list[FilaEstadoCelda]:
    """Crea entradas pendientes para todo el diseño factorial."""
    return [
        {
            "modelo": nombre,
            "data_fraction": fracc,
            "fold": k,
            "estado": "pendiente",
        }
        for nombre, fracc, k in generar_celdas_design(cfg)
    ]


def _actualizar_celda_estado(
    celdas: list[FilaEstadoCelda],
    clave: ClaveCelda,
    *,
    estado: EstadoCelda,
    motivo: str | None = None,
    train_time_s: float | None = None,
) -> None:
    """Actualiza o inserta el estado de una celda."""
    indice = _indice_celda(celdas, clave)
    entrada: FilaEstadoCelda = {
        "modelo": clave[0],
        "data_fraction": clave[1],
        "fold": clave[2],
        "estado": estado,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    if motivo is not None:
        entrada["motivo"] = motivo
    if train_time_s is not None:
        entrada["train_time_s"] = train_time_s

    if indice is None:
        celdas.append(entrada)
    else:
        celdas[indice] = {**celdas[indice], **entrada}


def _escribir_archivo_corridas(filas: list[dict[str, str]], ruta: Path) -> None:
    """Añade filas archivadas a un CSV auxiliar, creando la cabecera si hace falta."""
    if not filas:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escribir_cabecera = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        if escribir_cabecera:
            escritor.writeheader()
        escritor.writerows(filas)


def archivar_corridas_no_cuda(
    cfg: ExperimentConfig,
    *,
    dispositivo_objetivo: str = DISPOSITIVO_CAMPANA,
) -> dict[str, Any]:
    """Retira del CSV oficial las corridas ajenas a la campaña en CUDA (decisión D3).

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con rutas de resultados y el presupuesto de épocas del protocolo.
    dispositivo_objetivo : str
        Dispositivo con el que se ejecuta la campaña completa.

    Returns
    -------
    dict[str, Any]
        Conteos por categoría y rutas de los archivos generados.

    Notes
    -----
    Separa en dos categorías y **no elimina** ninguna evidencia:

    - Pruebas informales: presupuesto de épocas distinto al del protocolo
      (por ejemplo la sonda de 1 época). Van a ``pruebas_informales.csv``.
    - Corridas heterogéneas: presupuesto correcto pero dispositivo distinto al
      objetivo (las líneas base de TASK-12 en ``mps``). Van a ``historico_mps.csv``.

    Los historiales por época de ambas categorías se mueven a ``history_mps/``.
    Las celdas afectadas vuelven a ``pendiente`` en ``campana_estado.json``, de modo
    que ``Trainer.corrida_completada()`` no las omita por reanudabilidad.

    Los pesos en ``models/`` no se archivan: la convención de nombres de TASK-4 no
    incluye el dispositivo, así que la reejecución los sobrescribe. La evidencia
    conservada de las corridas retiradas es su fila en el CSV y su historial.
    """
    cfg.ensure_layout()
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    if not ruta_csv.exists():
        logger.info("No hay experiments.csv que archivar en %s", ruta_csv)
        return {"conservadas": 0, "historicas": 0, "informales": 0, "historiales_movidos": 0}

    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    conservadas: list[dict[str, str]] = []
    historicas: list[dict[str, str]] = []
    informales: list[dict[str, str]] = []

    for fila in filas:
        if int(fila["epocas"]) != cfg.epocas:
            informales.append(fila)
        elif fila["dispositivo"] != dispositivo_objetivo:
            historicas.append(fila)
        else:
            conservadas.append(fila)

    if not historicas and not informales:
        logger.info("Nada que archivar: las %d filas ya son de la campaña", len(conservadas))
        return {
            "conservadas": len(conservadas),
            "historicas": 0,
            "informales": 0,
            "historiales_movidos": 0,
        }

    _escribir_archivo_corridas(historicas, cfg.raiz_resultados / RUTA_HISTORICO_NO_CUDA)
    _escribir_archivo_corridas(informales, cfg.raiz_resultados / RUTA_PRUEBAS_INFORMALES)

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        escritor.writerows(conservadas)

    directorio_origen = cfg.raiz_resultados / "history"
    directorio_destino = cfg.raiz_resultados / DIR_HISTORIAL_ARCHIVADO
    directorio_destino.mkdir(parents=True, exist_ok=True)

    movidos = 0
    for fila in historicas + informales:
        nombre = nombre_historial(_run_record_desde_fila(fila))
        origen = directorio_origen / nombre
        if origen.is_file():
            origen.replace(directorio_destino / nombre)
            movidos += 1

    estado_previo = cargar_estado_campana(cfg)
    celdas_estado: list[FilaEstadoCelda] = (
        estado_previo["celdas"] if estado_previo else _inicializar_estado_celdas(cfg)
    )
    for fila in historicas + informales:
        _actualizar_celda_estado(celdas_estado, _clave_celda(fila), estado="pendiente")
    guardar_estado_campana(cfg, celdas_estado)

    logger.info(
        "Archivado D3: %d corridas heterogéneas, %d pruebas informales, "
        "%d historiales movidos; quedan %d filas en el CSV oficial",
        len(historicas),
        len(informales),
        movidos,
        len(conservadas),
    )
    return {
        "conservadas": len(conservadas),
        "historicas": len(historicas),
        "informales": len(informales),
        "historiales_movidos": movidos,
        "rutas": {
            "historico": cfg.raiz_resultados / RUTA_HISTORICO_NO_CUDA,
            "informales": cfg.raiz_resultados / RUTA_PRUEBAS_INFORMALES,
            "historiales": directorio_destino,
        },
    }


def ejecutar_campana(
    cfg_base: ExperimentConfig | None = None,
    *,
    fraccion: float | None = None,
    modelo: str | None = None,
    fold: int | None = None,
    max_epocas: int | None = None,
    verificar_pre: bool = True,
    exigir_baselines: bool = True,
) -> dict[str, Any]:
    """Ejecuta el diseño factorial con reanudabilidad por celda.

    Parameters
    ----------
    cfg_base : ExperimentConfig | None
        Configuración base; usa valores por defecto si es ``None``.
    fraccion : float | None
        Si se indica, limita a esa fracción (bloque Colab).
    modelo : str | None
        Si se indica, limita a ese modelo.
    fold : int | None
        Si se indica, limita a ese fold.
    max_epocas : int | None
        Presupuesto reducido de épocas (sondas).
    verificar_pre : bool
        Si ``True``, ejecuta ``verificar_precondiciones`` al inicio.
    exigir_baselines : bool
        Si ``False``, no exige las celdas baseline de TASK-12 en el CSV porque
        fueron archivadas para reejecutarse en CUDA (decisión D3). Sigue validando
        la ``L`` congelada y el hash de ``splits.json``.

    Returns
    -------
    dict[str, Any]
        Registros nuevos, conteos y rutas de artefactos.
    """
    cfg_base = cfg_base or ExperimentConfig()
    cfg_base.ensure_layout()

    hparams: dict[str, Any] | None = None
    if verificar_pre:
        hparams = verificar_precondiciones(cfg_base, exigir_baselines=exigir_baselines)

    L = n_capas_congelada(_ruta_hparams(cfg_base))
    if max_epocas is not None:
        cfg_base = replace(cfg_base, epocas=max_epocas)

    qml.qnn.TorchLayer.set_input_argument("entradas")

    estado_previo = cargar_estado_campana(cfg_base)
    celdas_estado: list[FilaEstadoCelda] = (
        estado_previo["celdas"] if estado_previo else _inicializar_estado_celdas(cfg_base)
    )

    ruta_csv = cfg_base.raiz_resultados / "experiments.csv"
    celdas_a_correr = generar_celdas_design(
        cfg_base,
        fraccion=fraccion,
        modelo=modelo,
        fold=fold,
    )

    logger.info(
        "Protocolo TASK-13: L=%d, epocas=%d, semilla=%d, celdas=%d",
        L,
        cfg_base.epocas,
        cfg_base.semilla,
        len(celdas_a_correr),
    )

    registros_nuevos: list[RunRecord] = []
    omitidas = 0
    fallidas = 0

    for nombre, fracc, k in celdas_a_correr:
        clave: ClaveCelda = (nombre, fracc, k)
        cfg = replace(
            cfg_base,
            modelo=nombre,
            data_fraction=fracc,
            n_capas=L,
        )
        dispositivo = dispositivo_para_modelo(nombre)
        log_dispositivo(dispositivo)

        try:
            set_seed(cfg.semilla)
            verificar_indices_fold(cfg, k)

            modelo_nn = build_model(nombre, cfg)
            entrenador = Trainer(modelo_nn, cfg, dispositivo, fold=k)

            if entrenador.corrida_completada():
                logger.info(
                    "Celda omitida (ya en CSV): modelo=%s fraccion=%.2f fold=%d",
                    nombre,
                    fracc,
                    k,
                )
                _actualizar_celda_estado(celdas_estado, clave, estado="omitida")
                omitidas += 1
                continue

            cargador_train, cargador_val = construir_loaders_para_fold(cfg, k)
            registro, historial = entrenador.ajustar(cargador_train, cargador_val)
            entrenador.guardar_pesos()
            escribir_corrida_csv(registro, ruta_csv)
            escribir_historial_json(registro, historial, cfg)

            _actualizar_celda_estado(
                celdas_estado,
                clave,
                estado="completada",
                train_time_s=registro.train_time_s,
            )
            registros_nuevos.append(registro)

            logger.info(
                "%s fraccion=%.2f fold=%d: acc_val=%.4f tiempo=%.1fs",
                nombre,
                fracc,
                k,
                registro.accuracy_val,
                registro.train_time_s,
            )
        except Exception as exc:
            motivo = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Celda fallida: modelo=%s fraccion=%.2f fold=%d — %s",
                nombre,
                fracc,
                k,
                motivo,
            )
            _actualizar_celda_estado(
                celdas_estado,
                clave,
                estado="fallida",
                motivo=motivo,
            )
            fallidas += 1

    ruta_estado = guardar_estado_campana(cfg_base, celdas_estado, hparams=hparams)
    costo = comparar_costo(cfg_base, hparams=hparams)

    return {
        "registros_nuevos": registros_nuevos,
        "omitidas": omitidas,
        "fallidas": fallidas,
        "n_capas": L,
        "costo": costo,
        "rutas": {
            "csv": ruta_csv,
            "estado": ruta_estado,
        },
    }


def _parsear_args() -> argparse.Namespace:
    """Define y parsea argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        description="Campana experimental factorial k-fold (TASK-13 / A8).",
    )
    parser.add_argument(
        "--fraccion",
        type=float,
        default=None,
        choices=FRACCIONES,
        help="Ejecuta solo el bloque de esta fraccion.",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        choices=MODELOS,
        help="Ejecuta solo el modelo indicado.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Ejecuta solo el fold indicado.",
    )
    parser.add_argument(
        "--max-epocas",
        type=int,
        default=None,
        help="Limita el presupuesto de epocas (sonda de tiempo).",
    )
    parser.add_argument(
        "--sonda",
        action="store_true",
        help="1 epoca, efficientnet_b0, 10%%, fold 0.",
    )
    parser.add_argument(
        "--verificar",
        action="store_true",
        help="Solo verifica integridad del diseno (60 celdas).",
    )
    parser.add_argument(
        "--sin-baselines-previas",
        action="store_true",
        help=(
            "No exige las 10 celdas baseline al 100%% de TASK-12 en el CSV: "
            "fueron archivadas y se reejecutan en CUDA (decision D3). "
            "Sigue validando L congelada y el hash de splits.json."
        ),
    )
    parser.add_argument(
        "--archivar-no-cuda",
        action="store_true",
        help=(
            "Solo archiva las corridas ajenas a la campana en CUDA "
            "(dispositivo distinto o presupuesto de epocas distinto) y termina."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI para la campaña TASK-13."""
    configurar_logging_cli()
    args = _parsear_args()

    cfg = ExperimentConfig()

    if args.archivar_no_cuda:
        archivar_corridas_no_cuda(cfg)
        return

    if args.verificar:
        verificar_precondiciones(cfg)
        resultado = verificar_integridad(cfg)
        costo = comparar_costo(cfg)
        logger.info("Verificacion OK: %s", resultado)
        logger.info(
            "Costo: %.2f h reales vs %.2f h estimadas (desviacion %.1f%%)",
            costo["horas_reales"],
            costo["horas_estimadas"],
            costo["desviacion_pct"] or 0.0,
        )
        return

    fraccion = args.fraccion
    modelo = args.modelo
    fold = args.fold
    max_epocas = args.max_epocas

    if args.sonda:
        fraccion = 0.10
        modelo = "efficientnet_b0"
        fold = 0
        max_epocas = 1
        logger.info("Modo sonda: 1 epoca, efficientnet_b0, 10%%, fold=0")

    ejecutar_campana(
        cfg,
        fraccion=fraccion,
        modelo=modelo,
        fold=fold,
        max_epocas=max_epocas,
        exigir_baselines=not args.sin_baselines_previas,
    )


if __name__ == "__main__":
    main()
