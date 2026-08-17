"""Contrato de métricas, logging y artefactos experimentales."""

from src.logging.records import (
    CLASES_ORDEN,
    COLUMNAS_CSV,
    EpochRecord,
    RunRecord,
    aplanar,
    especificidad_por_clase,
    sensibilidad_por_clase,
)
from src.logging.sinks import (
    corrida_existe,
    escribir_corrida_csv,
    escribir_historial_json,
    nombre_historial,
    obtener_commit_sha,
    registrar_corrida_wandb,
)
from src.logging.timing import (
    LOTES_CALENTAMIENTO,
    LOTES_MEDICION,
    medir_inferencia_ms_por_lote,
    sincronizar_dispositivo,
)

__all__ = [
    "CLASES_ORDEN",
    "COLUMNAS_CSV",
    "EpochRecord",
    "LOTES_CALENTAMIENTO",
    "LOTES_MEDICION",
    "RunRecord",
    "aplanar",
    "corrida_existe",
    "escribir_corrida_csv",
    "escribir_historial_json",
    "especificidad_por_clase",
    "medir_inferencia_ms_por_lote",
    "nombre_historial",
    "obtener_commit_sha",
    "registrar_corrida_wandb",
    "sensibilidad_por_clase",
    "sincronizar_dispositivo",
]
