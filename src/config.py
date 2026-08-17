"""Configuración centralizada del experimento."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUTA_HPARAMS_CONGELADOS: Path = Path("results/selected_hparams.json")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Única fuente de verdad de hiperparámetros y rutas del experimento.

    Attributes
    ----------
    modelo : str
        Identificador que consume la fábrica de modelos.
    n_clases : int
        Número de clases del problema de clasificación.
    n_qubits : int
        Número de qubits del VQC; coincide con el número de clases.
    n_capas : int
        Profundidad ``L`` de las StronglyEntanglingLayers.
    data_fraction : float
        Fracción del entrenamiento usada en el escenario de escasez.
    n_folds : int
        Número de folds para validación cruzada estratificada.
    epocas : int
        Número de épocas de entrenamiento por corrida.
    batch_size : int
        Tamaño de lote para el DataLoader.
    lr : float
        Tasa de aprendizaje del optimizador.
    semilla : int
        Semilla global para reproducibilidad.
    raiz_datos : Path
        Raíz del conjunto de datos MRI.
    raiz_resultados : Path
        Raíz de métricas, historiales y artefactos tabulares.
    raiz_modelos : Path
        Raíz de pesos guardados (``state_dict``).
    raiz_figuras : Path
        Raíz de figuras y fragmentos LaTeX generados.
    """

    modelo: str = "hqcnn"
    n_clases: int = 4
    n_qubits: int = 4
    n_capas: int = 6
    data_fraction: float = 1.0
    n_folds: int = 5
    epocas: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    semilla: int = 42
    raiz_datos: Path = Path("data/brain_tumor_mri")
    raiz_resultados: Path = Path("results")
    raiz_modelos: Path = Path("models")
    raiz_figuras: Path = Path("results/figures")

    def ensure_layout(self) -> None:
        """Crea las carpetas de datos, resultados, modelos y figuras si no existen."""
        for ruta in (
            self.raiz_datos,
            self.raiz_resultados,
            self.raiz_modelos,
            self.raiz_figuras,
        ):
            ruta.mkdir(parents=True, exist_ok=True)


def cargar_hparams_congelados(
    ruta: Path = RUTA_HPARAMS_CONGELADOS,
) -> dict[str, Any]:
    """Lee hiperparámetros congelados tras la ablación TASK-11.

    Parameters
    ----------
    ruta : Path
        Ruta a ``selected_hparams.json``.

    Returns
    -------
    dict[str, Any]
        Contenido del JSON con ``n_capas``, protocolo y presupuesto.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe (ablación aún no ejecutada).
    ValueError
        Si falta ``n_capas`` en el payload.
    """
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta la ablacion TASK-11 antes de la campaña."
        )
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    if "n_capas" not in payload:
        raise ValueError(f"El archivo {ruta} no contiene 'n_capas'.")
    return payload


def n_capas_congelada(ruta: Path = RUTA_HPARAMS_CONGELADOS) -> int:
    """Devuelve la profundidad ``L`` congelada en ``selected_hparams.json``.

    Parameters
    ----------
    ruta : Path
        Ruta al JSON de hiperparámetros.

    Returns
    -------
    int
        Valor de ``n_capas`` congelado.
    """
    return int(cargar_hparams_congelados(ruta)["n_capas"])
