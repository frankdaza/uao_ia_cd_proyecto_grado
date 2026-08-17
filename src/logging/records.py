"""Contrato de registros de corrida e historial por época (TASK-4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import recall_score

CLASES_ORDEN: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "notumor")

COLUMNAS_CSV: tuple[str, ...] = (
    "modelo",
    "data_fraction",
    "fold",
    "semilla",
    "dispositivo",
    "n_train",
    "n_val",
    "epocas",
    "n_params_entrenables",
    "n_capas_vqc",
    "commit_sha",
    "timestamp",
    "accuracy_train",
    "accuracy_val",
    "loss_train",
    "loss_val",
    "f1_val_weighted",
    "f1_val_macro",
    "sens_glioma",
    "sens_meningioma",
    "sens_pituitary",
    "sens_notumor",
    "spec_glioma",
    "spec_meningioma",
    "spec_pituitary",
    "spec_notumor",
    "train_time_s",
    "inference_ms_per_batch",
    "brecha_g",
)

_METRICAS_UNITARIAS = (
    "accuracy_train",
    "accuracy_val",
    "f1_val_weighted",
    "f1_val_macro",
)


def sensibilidad_por_clase(
    y_verdadero: np.ndarray,
    y_predicho: np.ndarray,
    clases: tuple[str, ...] = CLASES_ORDEN,
) -> dict[str, float]:
    """Calcula la sensibilidad (recall) por clase con orden fijo.

    Parameters
    ----------
    y_verdadero : np.ndarray
        Etiquetas reales codificadas como enteros en ``[0, n_clases)``.
    y_predicho : np.ndarray
        Etiquetas predichas con la misma codificación.
    clases : tuple[str, ...]
        Nombres de clase en el orden canónico del proyecto.

    Returns
    -------
    dict[str, float]
        Sensibilidad por clase, con una entrada por cada nombre en ``clases``.
    """
    recalls = recall_score(
        y_verdadero,
        y_predicho,
        labels=list(range(len(clases))),
        average=None,
        zero_division=0.0,
    )
    return {clase: float(valor) for clase, valor in zip(clases, recalls, strict=True)}


def especificidad_por_clase(
    matriz: np.ndarray,
    clases: tuple[str, ...] = CLASES_ORDEN,
) -> dict[str, float]:
    """Calcula la especificidad uno-contra-resto desde la matriz de confusión.

    Parameters
    ----------
    matriz : np.ndarray
        Matriz de confusión cuadrada con filas/columnas alineadas a ``clases``.
    clases : tuple[str, ...]
        Nombres de clase en el orden canónico del proyecto.

    Returns
    -------
    dict[str, float]
        Especificidad por clase.
    """
    total = matriz.sum()
    resultado: dict[str, float] = {}
    for i, clase in enumerate(clases):
        vp = matriz[i, i]
        fn = matriz[i, :].sum() - vp
        fp = matriz[:, i].sum() - vp
        vn = total - vp - fn - fp
        resultado[clase] = float(vn / (vn + fp)) if (vn + fp) else 0.0
    return resultado


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Métricas de una época para curvas de aprendizaje (A11).

    Attributes
    ----------
    epoca : int
        Índice de época, comenzando en 0.
    loss_train : float
        Pérdida media en entrenamiento.
    loss_val : float
        Pérdida media en validación.
    accuracy_train : float
        Exactitud en entrenamiento, en ``[0, 1]``.
    accuracy_val : float
        Exactitud en validación, en ``[0, 1]``.
    """

    epoca: int
    loss_train: float
    loss_val: float
    accuracy_train: float
    accuracy_val: float

    def validar(self) -> None:
        """Rechaza registros de época con exactitudes fuera de rango."""
        if self.epoca < 0:
            raise ValueError(f"epoca debe ser >= 0: {self.epoca}")
        for nombre in ("accuracy_train", "accuracy_val"):
            valor = getattr(self, nombre)
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} fuera de [0, 1]: {valor}")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Una corrida completa: unidad de observación del análisis estadístico.

    Notes
    -----
    La tupla ``(modelo, data_fraction, fold, semilla)`` identifica la fila de
    forma única. Cualquier campo faltante o fuera de rango invalida el registro
    antes de que llegue a disco.
    """

    modelo: str
    data_fraction: float
    fold: int
    semilla: int
    dispositivo: str
    n_train: int
    n_val: int
    epocas: int
    n_params_entrenables: int
    n_capas_vqc: int | None
    commit_sha: str
    timestamp: str
    accuracy_train: float
    accuracy_val: float
    loss_train: float
    loss_val: float
    f1_val_weighted: float
    f1_val_macro: float
    sensibilidad_por_clase: dict[str, float]
    especificidad_por_clase: dict[str, float]
    train_time_s: float
    inference_ms_per_batch: float

    @property
    def brecha_g(self) -> float:
        """Brecha de generalización |Acc_train - Acc_val| exigida por A11."""
        return abs(self.accuracy_train - self.accuracy_val)

    def validar(self) -> None:
        """Rechaza registros incompletos o con métricas fuera de rango."""
        if not self.modelo.strip():
            raise ValueError("modelo no puede estar vacío")
        if not self.commit_sha.strip():
            raise ValueError("commit_sha no puede estar vacío")
        if not self.timestamp.strip():
            raise ValueError("timestamp no puede estar vacío")
        if self.fold < 0:
            raise ValueError(f"fold debe ser >= 0: {self.fold}")
        if self.n_train < 0 or self.n_val < 0:
            raise ValueError("n_train y n_val deben ser >= 0")
        if self.epocas <= 0:
            raise ValueError(f"epocas debe ser > 0: {self.epocas}")
        if self.n_params_entrenables < 0:
            raise ValueError("n_params_entrenables debe ser >= 0")
        if self.train_time_s < 0.0 or self.inference_ms_per_batch < 0.0:
            raise ValueError("los tiempos deben ser >= 0")
        for nombre in _METRICAS_UNITARIAS:
            valor = getattr(self, nombre)
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nombre} fuera de [0, 1]: {valor}")
        _validar_metricas_por_clase(self.sensibilidad_por_clase, "sensibilidad")
        _validar_metricas_por_clase(self.especificidad_por_clase, "especificidad")


def _validar_metricas_por_clase(metricas: dict[str, float], etiqueta: str) -> None:
    """Verifica que existan exactamente las cuatro clases canónicas en rango."""
    if set(metricas) != set(CLASES_ORDEN):
        raise ValueError(
            f"Se requieren {etiqueta} para las 4 clases {CLASES_ORDEN}; "
            f"recibido: {sorted(metricas)}"
        )
    for clase in CLASES_ORDEN:
        valor = metricas[clase]
        if not 0.0 <= valor <= 1.0:
            raise ValueError(f"{etiqueta}[{clase}] fuera de [0, 1]: {valor}")


def aplanar(registro: RunRecord) -> dict[str, Any]:
    """Convierte el registro en una fila tidy con columnas por clase.

    Parameters
    ----------
    registro : RunRecord
        Registro validado de una corrida completa.

    Returns
    -------
    dict[str, Any]
        Diccionario con las columnas de ``COLUMNAS_CSV``, listo para CSV o wandb.
    """
    registro.validar()
    fila = asdict(registro)
    for clase, valor in fila.pop("sensibilidad_por_clase").items():
        fila[f"sens_{clase}"] = valor
    for clase, valor in fila.pop("especificidad_por_clase").items():
        fila[f"spec_{clase}"] = valor
    fila["brecha_g"] = registro.brecha_g
    if fila["n_capas_vqc"] is None:
        fila["n_capas_vqc"] = ""
    return {columna: fila[columna] for columna in COLUMNAS_CSV}
