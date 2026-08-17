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
from src.models.vqc import (
    ESCALA_INICIALIZACION,
    MAPEO_QUBIT_CLASE,
    N_QUBITS,
    PROFUNDIDADES_DIAGNOSTICO,
    RUTA_DIAGRAMA_CIRCUITO,
    circuito_vqc,
    diagnosticar_normas_gradiente,
    forma_pesos_vqc,
    generar_diagrama_circuito,
    inicializar_pesos,
    norma_gradiente_inicial,
)

__all__ = [
    "BACKBONES_SOPORTADOS",
    "ESCALA_INICIALIZACION",
    "MAPEO_QUBIT_CLASE",
    "N_QUBITS",
    "PROFUNDIDADES_DIAGNOSTICO",
    "RUTA_DIAGRAMA_CIRCUITO",
    "VERSIONES_PESOS",
    "CabeceraReduccion",
    "build_backbone",
    "circuito_vqc",
    "contar_parametros",
    "diagnosticar_normas_gradiente",
    "forma_pesos_vqc",
    "generar_diagrama_circuito",
    "inicializar_pesos",
    "norma_gradiente_inicial",
    "obtener_version_pesos",
    "parametros_entrenables",
]
