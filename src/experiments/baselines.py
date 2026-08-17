"""Entrenamiento de líneas base clásicas al 100% del conjunto (TASK-12 / A6)."""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

from src.config import ExperimentConfig
from src.data.dataset import cargar_manifiesto_utilizable
from src.data.splits import cargar_splits, obtener_indices
from src.logging.records import COLUMNAS_CSV, RunRecord
from src.logging.sinks import escribir_corrida_csv, escribir_historial_json
from src.models.backbones import BACKBONES_SOPORTADOS, VERSIONES_PESOS, obtener_version_pesos
from src.models.factory import build_model
from src.train.dataloading import construir_loaders_para_fold
from src.train.trainer import Trainer
from src.utils.device import get_device, log_dispositivo
from src.utils.logging import configurar_logging_cli
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

MODELOS_BASELINE: tuple[str, ...] = ("efficientnet_b0", "resnet50")
FRACCION_BASELINE: float = 1.0

METRICAS_AGREGADAS: tuple[str, ...] = (
    "accuracy_val",
    "f1_val_weighted",
    "f1_val_macro",
    "loss_val",
    "train_time_s",
    "inference_ms_per_batch",
    "brecha_g",
)


class ResumenModelo(TypedDict):
    """Media y desviación estándar de una métrica agregada."""

    media: float
    desv_std: float


def verificar_indices_fold(cfg: ExperimentConfig, fold: int) -> tuple[list[int], list[int]]:
    """Comprueba que los índices del fold coinciden con ``results/splits.json``.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con fracción y rutas de resultados.
    fold : int
        Índice del fold estratificado.

    Returns
    -------
    tuple[list[int], list[int]]
        Par ``(train_idx, val_idx)`` leído del artefacto de particiones.

    Raises
    ------
    ValueError
        Si los índices cargados no coinciden con los esperados.
  """
    ruta_splits = cfg.raiz_resultados / "splits.json"
    splits = cargar_splits(ruta_splits)
    esperado_train, esperado_val = obtener_indices(splits, fold, cfg.data_fraction)

    ruta_manifiesto = cfg.raiz_resultados / "dataset_manifest.csv"
    manifiesto = cargar_manifiesto_utilizable(ruta_manifiesto)
    _, cargador_val = construir_loaders_para_fold(cfg, fold)
    n_train_esperado = len(esperado_train)
    n_val_esperado = len(esperado_val)

    if len(cargador_val.dataset) != n_val_esperado:
        raise ValueError(
            f"Fold {fold}: val esperado {n_val_esperado} muestras, "
            f"cargador tiene {len(cargador_val.dataset)}"
        )

    cargador_train, cargador_val = construir_loaders_para_fold(cfg, fold)
    if len(cargador_train.dataset) != n_train_esperado:
        raise ValueError(
            f"Fold {fold}: train esperado {n_train_esperado} muestras, "
            f"cargador tiene {len(cargador_train.dataset)}"
        )

    rutas_train_esperadas = set(manifiesto.iloc[esperado_train]["ruta_relativa"])
    rutas_val_esperadas = set(manifiesto.iloc[esperado_val]["ruta_relativa"])
    rutas_train_cargadas = set(cargador_train.dataset._filas["ruta_relativa"])
    rutas_val_cargadas = set(cargador_val.dataset._filas["ruta_relativa"])

    if rutas_train_esperadas != rutas_train_cargadas:
        raise ValueError(f"Fold {fold}: indices de entrenamiento no coinciden con el manifiesto")
    if rutas_val_esperadas != rutas_val_cargadas:
        raise ValueError(f"Fold {fold}: indices de validacion no coinciden con el manifiesto")

    logger.info(
        "Fold %d verificado: train=%d, val=%d (fraccion=%.2f)",
        fold,
        n_train_esperado,
        n_val_esperado,
        cfg.data_fraction,
    )
    return esperado_train, esperado_val


def consolidar_metricas(
    registros: list[RunRecord],
) -> dict[str, dict[str, ResumenModelo]]:
    """Calcula media y desviación estándar por modelo sobre los folds.

    Parameters
    ----------
    registros : list[RunRecord]
        Corridas completadas de las líneas base.

    Returns
    -------
    dict[str, dict[str, ResumenModelo]]
        Métricas agregadas indexadas por modelo y nombre de métrica.
    """
    por_modelo: dict[str, list[RunRecord]] = {}
    for registro in registros:
        por_modelo.setdefault(registro.modelo, []).append(registro)

    resultado: dict[str, dict[str, ResumenModelo]] = {}
    for modelo, filas in por_modelo.items():
        resultado[modelo] = {}
        for metrica in METRICAS_AGREGADAS:
            valores = [getattr(fila, metrica) for fila in filas]
            if len(valores) < 2:
                desv = 0.0
            else:
                desv = float(statistics.stdev(valores))
            resultado[modelo][metrica] = {
                "media": float(statistics.mean(valores)),
                "desv_std": desv,
            }
    return resultado


def guardar_resumen_csv(
    resumen: dict[str, dict[str, ResumenModelo]],
    ruta: Path,
) -> None:
    """Persiste la tabla agregada media ± desviación estándar.

    Parameters
    ----------
    resumen : dict[str, dict[str, ResumenModelo]]
        Salida de ``consolidar_metricas``.
    ruta : Path
        Destino CSV (``results/baselines_summary.csv``).
    """
    columnas = ["modelo", "metrica", "media", "desv_std"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for modelo in sorted(resumen):
            for metrica in METRICAS_AGREGADAS:
                fila = resumen[modelo][metrica]
                escritor.writerow(
                    {
                        "modelo": modelo,
                        "metrica": metrica,
                        "media": f"{fila['media']:.6f}",
                        "desv_std": f"{fila['desv_std']:.6f}",
                    }
                )
    logger.info("Resumen de baselines guardado en %s", ruta)


def ejecutar_baselines(
    cfg_base: ExperimentConfig | None = None,
    *,
    modelos: tuple[str, ...] = MODELOS_BASELINE,
    max_epocas: int | None = None,
    solo_fold: int | None = None,
    solo_modelo: str | None = None,
) -> dict[str, Any]:
    """Entrena las líneas base clásicas al 100% con k-fold estratificado.

    Parameters
    ----------
    cfg_base : ExperimentConfig | None
        Configuración base; usa valores por defecto si es ``None``.
    modelos : tuple[str, ...]
        Arquitecturas a entrenar.
    max_epocas : int | None
        Si se indica, limita el presupuesto de épocas (sonda de tiempo).
    solo_fold : int | None
        Si se indica, ejecuta solo ese fold (útil para sondas).
    solo_modelo : str | None
        Si se indica, ejecuta solo ese modelo.

    Returns
    -------
    dict[str, Any]
        Registros, resumen agregado y rutas de artefactos.
    """
    cfg_base = cfg_base or ExperimentConfig()
    cfg_base.ensure_layout()

    if max_epocas is not None:
        cfg_base = replace(cfg_base, epocas=max_epocas)

    dispositivo = get_device()
    nombre_dispositivo = log_dispositivo(dispositivo)
    ruta_csv = cfg_base.raiz_resultados / "experiments.csv"

    modelos_a_correr = modelos
    if solo_modelo is not None:
        if solo_modelo not in modelos:
            raise ValueError(f"Modelo no soportado para baselines: {solo_modelo}")
        modelos_a_correr = (solo_modelo,)

    folds = range(cfg_base.n_folds)
    if solo_fold is not None:
        if solo_fold < 0 or solo_fold >= cfg_base.n_folds:
            raise ValueError(f"Fold fuera de rango: {solo_fold}")
        folds = range(solo_fold, solo_fold + 1)

    logger.info(
        "Protocolo TASK-12: fraccion=%.2f, epocas=%d, semilla=%d, dispositivo=%s",
        FRACCION_BASELINE,
        cfg_base.epocas,
        cfg_base.semilla,
        nombre_dispositivo,
    )
    for nombre in modelos_a_correr:
        logger.info(
            "Version de pesos %s: %s",
            nombre,
            obtener_version_pesos(nombre),
        )

    registros: list[RunRecord] = []
    omitidas = 0

    for nombre in modelos_a_correr:
        for fold in folds:
            cfg = replace(
                cfg_base,
                modelo=nombre,
                data_fraction=FRACCION_BASELINE,
            )
            set_seed(cfg.semilla)
            verificar_indices_fold(cfg, fold)

            modelo = build_model(nombre, cfg)
            entrenador = Trainer(modelo, cfg, dispositivo, fold=fold)

            if entrenador.corrida_completada():
                logger.info(
                    "Celda ya completada: modelo=%s, fraccion=%.2f, fold=%d — omitida",
                    nombre,
                    FRACCION_BASELINE,
                    fold,
                )
                omitidas += 1
                continue

            cargador_train, cargador_val = construir_loaders_para_fold(cfg, fold)
            registro, historial = entrenador.ajustar(cargador_train, cargador_val)
            entrenador.guardar_pesos()
            escribir_corrida_csv(registro, ruta_csv)
            escribir_historial_json(registro, historial, cfg)
            registros.append(registro)

            logger.info(
                "%s fold=%d: acc_val=%.4f, f1_macro=%.4f, tiempo=%.1fs",
                nombre,
                fold,
                registro.accuracy_val,
                registro.f1_val_macro,
                registro.train_time_s,
            )

    todos_registros = _deduplicar_csv_baselines(ruta_csv)
    resumen = consolidar_metricas(todos_registros)
    ruta_resumen = cfg_base.raiz_resultados / "baselines_summary.csv"
    if resumen:
        guardar_resumen_csv(resumen, ruta_resumen)

    return {
        "registros_nuevos": registros,
        "registros_totales": todos_registros,
        "resumen": resumen,
        "omitidas": omitidas,
        "versiones_pesos": dict(VERSIONES_PESOS),
        "dispositivo": nombre_dispositivo,
        "rutas": {
            "csv": ruta_csv,
            "resumen": ruta_resumen,
        },
    }


def _deduplicar_csv_baselines(ruta_csv: Path) -> list[RunRecord]:
    """Elimina filas duplicadas del CSV y conserva la última por celda.

    Parameters
    ----------
    ruta_csv : Path
        Ruta a ``experiments.csv``.

    Returns
    -------
    list[RunRecord]
        Registros únicos de líneas base al 100%.
    """
    if not ruta_csv.exists():
        return []

    filas_raw = list(csv.DictReader(ruta_csv.open(newline="", encoding="utf-8")))
    vistas: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for fila in filas_raw:
        if fila["modelo"] not in MODELOS_BASELINE:
            continue
        if float(fila["data_fraction"]) != FRACCION_BASELINE:
            continue
        clave = (fila["modelo"], fila["data_fraction"], fila["fold"], fila["semilla"])
        vistas[clave] = fila

    unicas = sorted(vistas.values(), key=lambda f: (f["modelo"], int(f["fold"])))
    if len(unicas) < len([f for f in filas_raw if f["modelo"] in MODELOS_BASELINE]):
        with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV)
            escritor.writeheader()
            escritor.writerows(unicas)
        logger.warning(
            "CSV deduplicado: %d filas unicas de baselines al 100%%",
            len(unicas),
        )

    return [_run_record_desde_fila(fila) for fila in unicas]


def _cargar_registros_baselines(ruta_csv: Path) -> list[RunRecord]:
    """Lee del CSV las corridas de líneas base al 100%."""
    if not ruta_csv.exists():
        return []

    registros: list[RunRecord] = []
    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            if fila["modelo"] not in MODELOS_BASELINE:
                continue
            if float(fila["data_fraction"]) != FRACCION_BASELINE:
                continue
            registros.append(_run_record_desde_fila(fila))
    return registros


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


def _parsear_args() -> argparse.Namespace:
    """Define y parsea argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        description="Entrena lineas base clasicas al 100%% (TASK-12).",
    )
    parser.add_argument(
        "--sonda",
        action="store_true",
        help="Ejecuta 1 epoca, 1 fold, efficientnet_b0 para calibrar tiempo.",
    )
    parser.add_argument(
        "--max-epocas",
        type=int,
        default=None,
        help="Limita el presupuesto de epocas (sonda de tiempo).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Ejecuta solo el fold indicado.",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        choices=MODELOS_BASELINE,
        help="Ejecuta solo el modelo indicado.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI para las líneas base TASK-12."""
    configurar_logging_cli()
    args = _parsear_args()

    max_epocas = args.max_epocas
    solo_fold = args.fold
    solo_modelo = args.modelo

    if args.sonda:
        max_epocas = 1
        solo_fold = 0
        solo_modelo = "efficientnet_b0"
        logger.info("Modo sonda: 1 epoca, fold=0, efficientnet_b0")

    ejecutar_baselines(
        max_epocas=max_epocas,
        solo_fold=solo_fold,
        solo_modelo=solo_modelo,
    )


if __name__ == "__main__":
    main()
