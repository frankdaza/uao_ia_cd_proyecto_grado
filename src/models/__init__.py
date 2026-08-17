"""Modelos clásicos e híbridos para clasificación de MRI."""

from src.models.backbones import (
    BACKBONES_SOPORTADOS,
    VERSIONES_PESOS,
    build_backbone,
    contar_parametros,
    obtener_version_pesos,
    parametros_entrenables,
)
from src.models.heads import CabeceraReduccion

__all__ = [
    "BACKBONES_SOPORTADOS",
    "VERSIONES_PESOS",
    "CabeceraReduccion",
    "build_backbone",
    "contar_parametros",
    "obtener_version_pesos",
    "parametros_entrenables",
]
