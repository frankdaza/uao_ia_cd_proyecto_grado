"""Ablación económica de profundidad L y compuerta de presupuesto (TASK-11)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pennylane as qml
import torch

from src.config import ExperimentConfig
from src.data.splits import FRACCIONES
from src.logging.records import EpochRecord, RunRecord
from src.logging.sinks import escribir_historial_json, obtener_commit_sha
from src.models.hqcnn import HQCNN
from src.models.vqc import N_QUBITS, norma_gradiente_inicial
from src.train.dataloading import construir_loaders_para_fold
from src.train.trainer import Trainer
from src.utils.device import log_dispositivo
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

FOLD_ABLACION: int = 0
FRACCION_ABLACION: float = 0.25
EPOCAS_ABLACION: int = 5
PROFUNDIDADES: tuple[int, ...] = (2, 4, 6)
UMBRAL_HORAS_CAMPANA: float = 72.0
N_MODELOS_CAMPANA: int = 3
N_FOLDS_CAMPANA: int = 5
N_CELDAS_CAMPANA: int = N_MODELOS_CAMPANA * len(FRACCIONES) * N_FOLDS_CAMPANA
SIMULADOR_CUANTICO: str = "default.qubit"

CRITERIO_SELECCION: str = (
    "Mayor F1 macro de validacion al final del presupuesto reducido; "
    "ante diferencias <= 0.02, preferir la L mas baja por costo y entrenabilidad."
)
TOLERANCIA_F1: float = 0.02

COLUMNAS_ABLACION: tuple[str, ...] = (
    "n_capas",
    "f1_val_macro",
    "accuracy_val",
    "loss_val",
    "segundos_por_epoca",
    "n_params_cuanticos",
    "norma_gradiente_inicial",
    "dispositivo",
    "n_train",
    "epocas",
)

MITIGACIONES_EVALUADAS: tuple[dict[str, str], ...] = (
    {
        "nombre": "precalcular_caracteristicas_backbone",
        "impacto": (
            "Incompatible con aumento de datos en cada epoca; invalida comparacion "
            "con el metodo declarado salvo desactivar augmentation (TASK-5)."
        ),
    },
    {
        "nombre": "reducir_epocas_campana",
        "impacto": (
            "Reduce costo pero puede dejar modelos sin converger; sesga la "
            "comparacion frente a baselines con el mismo presupuesto."
        ),
    },
    {
        "nombre": "reducir_k_folds",
        "impacto": (
            "Menos folds debilita la estimacion de varianza y condiciona la "
            "ANOVA de dos vias (TASK-15)."
        ),
    },
    {
        "nombre": "restringir_diseno_factorial",
        "impacto": (
            "Eliminar celdas rompe el balance del diseno y el termino de "
            "interaccion modelo x fraccion (decision D2)."
        ),
    },
)


class FilaAblacion(TypedDict):
    """Fila de la tabla comparativa de profundidades."""

    n_capas: int
    f1_val_macro: float
    accuracy_val: float
    loss_val: float
    segundos_por_epoca: float
    n_params_cuanticos: int
    norma_gradiente_inicial: float
    dispositivo: str
    n_train: int
    epocas: int


@dataclass(frozen=True, slots=True)
class ResultadoViabilidad:
    """Decisión go/no-go y metadatos de presupuesto."""

    decision: str
    horas_estimadas: float
    umbral_horas: float
    decision_d2: str
    mitigaciones_evaluadas: tuple[dict[str, str], ...]
    mitigacion_adoptada: str | None
    notas: str


def get_device_hqcnn() -> torch.device:
    """Selecciona el dispositivo PyTorch para entrenar el HQCNN.

    Returns
    -------
    torch.device
        ``cuda`` si está disponible; en caso contrario ``cpu``.

    Notes
    -----
    ``TorchLayer`` con ``parameter-shift`` no es estable en ``mps`` (PennyLane
    0.45 + PyTorch 2.9): los logits pueden quedar sin almacenamiento válido y
    ``CrossEntropyLoss`` falla. El simulador ``default.qubit`` corre en CPU;
    usar ``cpu`` en macOS garantiza tiempos extrapolables y reproducibles.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info(
            "HQCNN: se usa CPU porque TorchLayer + parameter-shift no es estable en MPS"
        )
    return torch.device("cpu")


def params_cuanticos(n_capas: int, n_qubits: int = N_QUBITS) -> int:
    """Devuelve el número de parámetros variacionales del ansatz.

    Parameters
    ----------
    n_capas : int
        Profundidad ``L`` de ``StronglyEntanglingLayers``.
    n_qubits : int
        Número de qubits del circuito.

    Returns
    -------
    int
        Parámetros del bloque cuántico ($L \\times n \\times 3$).
    """
    return n_capas * n_qubits * 3


def estimar_horas_campana(
    segundos_por_epoca: float,
    epocas_campana: int,
    fraccion_referencia: float,
    *,
    fracciones: tuple[float, ...] = FRACCIONES,
    n_modelos: int = N_MODELOS_CAMPANA,
    n_folds: int = N_FOLDS_CAMPANA,
) -> float:
    """Estima las horas de cómputo de la campaña factorial completa.

    Parameters
    ----------
    segundos_por_epoca : float
        Tiempo por época medido en la ablación (HQCNN).
    epocas_campana : int
        Presupuesto de épocas de la campaña (TASK-13).
    fraccion_referencia : float
        Fracción de datos usada en la medición de referencia.
    fracciones : tuple[float, ...]
        Fracciones del diseño factorial.
    n_modelos : int
        Modelos por celda (HQCNN + 2 baselines).
    n_folds : int
        Folds estratificados por fracción.

    Returns
    -------
    float
        Horas estimadas para las 60 celdas.

    Notes
    -----
    Escala el tiempo por época con ``fraccion / fraccion_referencia`` porque
    más datos implican más lotes por época. Usa el tiempo del HQCNN como
    cota conservadora: las baselines clásicas son más rápidas.
    """
    if fraccion_referencia <= 0:
        raise ValueError(f"fraccion_referencia debe ser > 0: {fraccion_referencia}")

    total_segundos = 0.0
    for fraccion in fracciones:
        factor_muestras = fraccion / fraccion_referencia
        segundos_fraccion = segundos_por_epoca * factor_muestras * epocas_campana
        total_segundos += segundos_fraccion * n_modelos * n_folds
    return total_segundos / 3600.0


def seleccionar_profundidad(
    filas: list[FilaAblacion],
    *,
    tolerancia: float = TOLERANCIA_F1,
) -> int:
    """Aplica el criterio pre-declarado para congelar ``L``.

    Parameters
    ----------
    filas : list[FilaAblacion]
        Resultados de la ablación por profundidad.
    tolerancia : float
        Empate en F1 macro dentro de este margen favorece la ``L`` más baja.

    Returns
    -------
    int
        Profundidad seleccionada.

    Raises
    ------
    ValueError
        Si ``filas`` está vacía.
    """
    if not filas:
        raise ValueError("No hay filas de ablacion para seleccionar L.")

    mejor_f1 = max(fila["f1_val_macro"] for fila in filas)
    candidatas = [
        fila for fila in filas if fila["f1_val_macro"] >= mejor_f1 - tolerancia
    ]
    return min(fila["n_capas"] for fila in candidatas)


def decidir_viabilidad(
    horas_estimadas: float,
    *,
    umbral_horas: float = UMBRAL_HORAS_CAMPANA,
) -> ResultadoViabilidad:
    """Toma la decisión go/no-go sobre la campaña factorial.

    Parameters
    ----------
    horas_estimadas : float
        Horas extrapoladas para las 60 celdas.
    umbral_horas : float
        Límite de viabilidad acordado (72 h).

    Returns
    -------
    ResultadoViabilidad
        Decisión, estado de D2 y mitigaciones evaluadas.
    """
    if horas_estimadas <= umbral_horas:
        return ResultadoViabilidad(
            decision="go",
            horas_estimadas=horas_estimadas,
            umbral_horas=umbral_horas,
            decision_d2="confirmada",
            mitigaciones_evaluadas=MITIGACIONES_EVALUADAS,
            mitigacion_adoptada=None,
            notas=(
                f"La extrapolacion ({horas_estimadas:.1f} h) queda dentro del umbral "
                f"de {umbral_horas:.0f} h. Se confirma entrenar HQCNN al 100 % (D2)."
            ),
        )

    return ResultadoViabilidad(
        decision="no-go",
        horas_estimadas=horas_estimadas,
        umbral_horas=umbral_horas,
        decision_d2="ajustada",
        mitigaciones_evaluadas=MITIGACIONES_EVALUADAS,
        mitigacion_adoptada="reducir_epocas_campana",
        notas=(
            f"La extrapolacion ({horas_estimadas:.1f} h) supera el umbral de "
            f"{umbral_horas:.0f} h. Se adopta reducir epocas de campana como "
            "mitigacion minima; D2 (HQCNN al 100 %) queda sujeta a revision antes "
            "de TASK-13."
        ),
    )


def guardar_ablacion_csv(filas: list[FilaAblacion], ruta: Path) -> None:
    """Persiste la tabla comparativa de profundidades.

    Parameters
    ----------
    filas : list[FilaAblacion]
        Filas ordenadas por ``n_capas``.
    ruta : Path
        Destino CSV (``results/ablacion_L.csv``).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_ABLACION)
        escritor.writeheader()
        for fila in sorted(filas, key=lambda item: item["n_capas"]):
            escritor.writerow(fila)
    logger.info("Tabla de ablacion guardada en %s", ruta)


def guardar_selected_hparams(
    *,
    n_capas: int,
    criterio: str,
    protocolo: dict[str, Any],
    presupuesto: dict[str, Any],
    ruta: Path,
) -> None:
    """Congela ``L`` y la decisión de viabilidad antes de TASK-13.

    Parameters
    ----------
    n_capas : int
        Profundidad seleccionada.
    criterio : str
        Criterio de selección pre-declarado.
    protocolo : dict[str, Any]
        Metadatos del protocolo económico.
    presupuesto : dict[str, Any]
        Extrapolación de costo y decisión go/no-go.
    ruta : Path
        Destino JSON (``results/selected_hparams.json``).
    """
    payload = {
        "n_capas": n_capas,
        "criterio_seleccion": criterio,
        "fecha": datetime.now(tz=UTC).isoformat(),
        "commit_sha": obtener_commit_sha(),
        "protocolo_ablacion": protocolo,
        "presupuesto": presupuesto,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Hiperparametros congelados en %s (L=%d)", ruta, n_capas)


def graficar_curvas_perdida(
    historiales: dict[int, list[EpochRecord]],
    ruta: Path,
) -> None:
    """Genera curvas de pérdida train/val por profundidad.

    Parameters
    ----------
    historiales : dict[int, list[EpochRecord]]
        Historial por época indexado por ``L``.
    ruta : Path
        Ruta PNG de salida.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for n_capas in sorted(historiales):
        historial = historiales[n_capas]
        epocas = [registro.epoca + 1 for registro in historial]
        loss_train = [registro.loss_train for registro in historial]
        loss_val = [registro.loss_val for registro in historial]
        ejes[0].plot(epocas, loss_train, marker="o", label=f"L={n_capas}")
        ejes[1].plot(epocas, loss_val, marker="o", label=f"L={n_capas}")

    ejes[0].set_title("Perdida de entrenamiento")
    ejes[1].set_title("Perdida de validacion")
    for eje in ejes:
        eje.set_xlabel("Epoca")
        eje.set_ylabel("CrossEntropyLoss")
        eje.grid(True, alpha=0.3)
        eje.legend()

    figura.suptitle(
        "Ablacion TASK-11: curvas de perdida (presupuesto reducido, no A8)",
        fontsize=11,
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(figura)
    logger.info("Curvas de perdida guardadas en %s", ruta)


def _fila_desde_registro(
    registro: RunRecord,
    *,
    n_capas: int,
    norma_gradiente: float,
) -> FilaAblacion:
    """Construye una fila de ablación a partir del ``RunRecord``."""
    segundos_por_epoca = registro.train_time_s / registro.epocas
    return {
        "n_capas": n_capas,
        "f1_val_macro": registro.f1_val_macro,
        "accuracy_val": registro.accuracy_val,
        "loss_val": registro.loss_val,
        "segundos_por_epoca": segundos_por_epoca,
        "n_params_cuanticos": params_cuanticos(n_capas),
        "norma_gradiente_inicial": norma_gradiente,
        "dispositivo": registro.dispositivo,
        "n_train": registro.n_train,
        "epocas": registro.epocas,
    }


def ejecutar_ablacion(cfg_base: ExperimentConfig | None = None) -> dict[str, Any]:
    """Ejecuta la ablación económica y congela la decisión de ``L``.

    Parameters
    ----------
    cfg_base : ExperimentConfig | None
        Configuración base; usa valores por defecto del proyecto si es ``None``.

    Returns
    -------
    dict[str, Any]
        Resumen con filas, ``L`` seleccionada, viabilidad y rutas de artefactos.
    """
    cfg_base = cfg_base or ExperimentConfig()
    cfg_base.ensure_layout()

    qml.qnn.TorchLayer.set_input_argument("entradas")

    filas: list[FilaAblacion] = []
    historiales: dict[int, list[EpochRecord]] = {}
    registros: dict[int, RunRecord] = {}
    dispositivo = get_device_hqcnn()
    nombre_dispositivo = log_dispositivo(dispositivo)

    logger.info("Protocolo TASK-11: fold=%d, fraccion=%.2f, epocas=%d, semilla=%d",
                FOLD_ABLACION, FRACCION_ABLACION, EPOCAS_ABLACION, cfg_base.semilla)
    logger.info("Criterio de seleccion (pre-declarado): %s", CRITERIO_SELECCION)
    logger.info("Umbral go/no-go: %.0f h para %d celdas", UMBRAL_HORAS_CAMPANA, N_CELDAS_CAMPANA)

    for n_capas in PROFUNDIDADES:
        cfg = replace(
            cfg_base,
            modelo="hqcnn",
            n_capas=n_capas,
            epocas=EPOCAS_ABLACION,
            data_fraction=FRACCION_ABLACION,
        )
        set_seed(cfg.semilla)
        norma = norma_gradiente_inicial(n_capas, semilla=cfg.semilla)

        cargador_train, cargador_val = construir_loaders_para_fold(cfg, FOLD_ABLACION)
        modelo = HQCNN(cfg)
        entrenador = Trainer(modelo, cfg, dispositivo, fold=FOLD_ABLACION)
        registro, historial = entrenador.ajustar(cargador_train, cargador_val)

        escribir_historial_json(registro, historial, cfg)
        fraccion_txt = str(cfg.data_fraction).replace(".", "p")
        ruta_hist_estandar = (
            cfg.raiz_resultados
            / "history"
            / f"hqcnn_{fraccion_txt}_f{FOLD_ABLACION}_s{cfg.semilla}.json"
        )
        if ruta_hist_estandar.exists():
            ruta_hist_ablacion = (
                cfg.raiz_resultados
                / "history"
                / f"hqcnn_ablacion_L{n_capas}_{fraccion_txt}_f{FOLD_ABLACION}_s{cfg.semilla}.json"
            )
            ruta_hist_ablacion.write_text(
                ruta_hist_estandar.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        fila = _fila_desde_registro(registro, n_capas=n_capas, norma_gradiente=norma)
        filas.append(fila)
        historiales[n_capas] = historial
        registros[n_capas] = registro

        logger.info(
            "L=%d: F1 macro=%.4f, seg/epoch=%.1f, norma grad=%.4f",
            n_capas,
            fila["f1_val_macro"],
            fila["segundos_por_epoca"],
            fila["norma_gradiente_inicial"],
        )

    n_capas_seleccionada = seleccionar_profundidad(filas)
    fila_seleccionada = next(
        fila for fila in filas if fila["n_capas"] == n_capas_seleccionada
    )
    horas_estimadas = estimar_horas_campana(
        fila_seleccionada["segundos_por_epoca"],
        epocas_campana=cfg_base.epocas,
        fraccion_referencia=FRACCION_ABLACION,
    )
    viabilidad = decidir_viabilidad(horas_estimadas)

    ruta_csv = cfg_base.raiz_resultados / "ablacion_L.csv"
    ruta_json = cfg_base.raiz_resultados / "selected_hparams.json"
    ruta_figura = cfg_base.raiz_figuras / "ablacion_L_curvas_perdida.png"

    guardar_ablacion_csv(filas, ruta_csv)
    graficar_curvas_perdida(historiales, ruta_figura)

    protocolo = {
        "fold": FOLD_ABLACION,
        "data_fraction": FRACCION_ABLACION,
        "epocas_ablacion": EPOCAS_ABLACION,
        "epocas_campana": cfg_base.epocas,
        "semilla": cfg_base.semilla,
        "simulador_cuantico": SIMULADOR_CUANTICO,
        "dispositivo": nombre_dispositivo,
        "profundidades_evaluadas": list(PROFUNDIDADES),
        "nota_metodologica": (
            "Decision de diseno instrumentada; no tiene significancia estadistica (no A8)."
        ),
    }
    presupuesto = {
        "segundos_por_epoca_referencia": fila_seleccionada["segundos_por_epoca"],
        "L_referencia": n_capas_seleccionada,
        "horas_campana_estimadas": round(horas_estimadas, 2),
        "n_celdas": N_CELDAS_CAMPANA,
        "umbral_horas": UMBRAL_HORAS_CAMPANA,
        "decision": viabilidad.decision,
        "decision_d2": viabilidad.decision_d2,
        "mitigaciones_evaluadas": list(viabilidad.mitigaciones_evaluadas),
        "mitigacion_adoptada": viabilidad.mitigacion_adoptada,
        "notas": viabilidad.notas,
    }

    guardar_selected_hparams(
        n_capas=n_capas_seleccionada,
        criterio=CRITERIO_SELECCION,
        protocolo=protocolo,
        presupuesto=presupuesto,
        ruta=ruta_json,
    )

    logger.info(
        "L congelada=%d | decision=%s | horas estimadas=%.1f",
        n_capas_seleccionada,
        viabilidad.decision,
        horas_estimadas,
    )

    return {
        "filas": filas,
        "n_capas_seleccionada": n_capas_seleccionada,
        "viabilidad": viabilidad,
        "horas_estimadas": horas_estimadas,
        "rutas": {
            "csv": ruta_csv,
            "json": ruta_json,
            "figura": ruta_figura,
        },
    }


def main() -> None:
    """Punto de entrada CLI para la ablación TASK-11."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ejecutar_ablacion()


if __name__ == "__main__":
    main()
