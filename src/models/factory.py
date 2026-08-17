"""Fábrica unificada de modelos clásicos e híbridos (TASK-12 / TASK-13)."""

from __future__ import annotations

import torch.nn as nn

from src.config import ExperimentConfig
from src.models.backbones import BACKBONES_SOPORTADOS
from src.models.baseline import ClassicalBaseline
from src.models.hqcnn import HQCNN


def build_model(nombre: str, cfg: ExperimentConfig) -> nn.Module:
    """Construye el modelo indicado sin ramas ad hoc en el orquestador.

    Parameters
    ----------
    nombre : str
        Identificador: ``hqcnn``, ``efficientnet_b0`` o ``resnet50``.
    cfg : ExperimentConfig
        Configuración del experimento.

    Returns
    -------
    nn.Module
        Instancia lista para el ``Trainer``.

    Raises
    ------
    ValueError
        Si ``nombre`` no corresponde a un modelo soportado.
    """
    if nombre == "hqcnn":
        return HQCNN(cfg)
    if nombre in BACKBONES_SOPORTADOS:
        return ClassicalBaseline(cfg, backbone=nombre)
    raise ValueError(
        f"Modelo no soportado: {nombre}. "
        f"Opciones: hqcnn, {', '.join(BACKBONES_SOPORTADOS)}"
    )
