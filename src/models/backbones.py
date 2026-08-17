"""Fábrica de extractores preentrenados y congelados (TASK-7 / A1)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

BACKBONES_SOPORTADOS: tuple[str, ...] = ("efficientnet_b0", "resnet50")

VERSIONES_PESOS: dict[str, str] = {
    "efficientnet_b0": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
}


def obtener_version_pesos(nombre: str) -> str:
    """Devuelve el identificador de la versión de pesos ImageNet usada.

    Parameters
    ----------
    nombre : str
        Identificador del backbone (``efficientnet_b0`` o ``resnet50``).

    Returns
    -------
    str
        Nombre de la enumeración de pesos (p. ej. ``IMAGENET1K_V1``).

    Raises
    ------
    ValueError
        Si el nombre no está en ``BACKBONES_SOPORTADOS``.
    """
    if nombre not in VERSIONES_PESOS:
        raise ValueError(f"Backbone no soportado: {nombre}")
    return VERSIONES_PESOS[nombre]


def build_backbone(nombre: str) -> tuple[nn.Module, int]:
    """Construye un extractor preentrenado y congelado.

    Parameters
    ----------
    nombre : str
        Identificador de la arquitectura: ``efficientnet_b0`` o ``resnet50``.

    Returns
    -------
    tuple[nn.Module, int]
        El backbone sin cabeza de clasificación y la dimensión de su salida.

    Notes
    -----
    El backbone se devuelve en modo evaluación para que las capas de
    normalización por lotes no actualicen sus estadísticas móviles con los
    lotes pequeños del escenario de escasez. Mantener ``eval()`` cuando el
    ``Trainer`` llama ``model.train()`` es responsabilidad del módulo
    contenedor (TASK-10).
    """
    if nombre == "efficientnet_b0":
        modelo = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        dimension = modelo.classifier[1].in_features
        modelo.classifier = nn.Identity()
    elif nombre == "resnet50":
        modelo = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        dimension = modelo.fc.in_features
        modelo.fc = nn.Identity()
    else:
        raise ValueError(f"Backbone no soportado: {nombre}")

    for parametro in modelo.parameters():
        parametro.requires_grad = False
    modelo.eval()
    return modelo, dimension


def contar_parametros(modelo: nn.Module) -> dict[str, int]:
    """Cuenta parámetros entrenables y congelados de un módulo.

    Parameters
    ----------
    modelo : nn.Module
        Módulo PyTorch cuyos parámetros se contabilizan.

    Returns
    -------
    dict[str, int]
        Diccionario con claves ``entrenables`` y ``congelados``.
    """
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    congelados = sum(p.numel() for p in modelo.parameters() if not p.requires_grad)
    return {"entrenables": entrenables, "congelados": congelados}


def parametros_entrenables(modelo: nn.Module) -> list[nn.Parameter]:
    """Devuelve solo los parámetros con ``requires_grad=True``.

    Parameters
    ----------
    modelo : nn.Module
        Módulo cuyos parámetros entrenables se filtran.

    Returns
    -------
    list[nn.Parameter]
        Lista de tensores que deben entrar al optimizador.

    Notes
    -----
    Pasar ``modelo.parameters()`` completo a Adam mantiene estado (momentos)
    para tensores congelados: gasta memoria y arranca con estado espurio si
    alguien reactiva ``requires_grad``.
    """
    return [p for p in modelo.parameters() if p.requires_grad]
