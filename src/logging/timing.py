"""Protocolo de medición de tiempos de entrenamiento e inferencia (TASK-4)."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

LOTES_CALENTAMIENTO = 3
LOTES_MEDICION = 10


def sincronizar_dispositivo(dispositivo: torch.device) -> None:
    """Fuerza la finalización de kernels asíncronos antes de medir tiempos.

    Parameters
    ----------
    dispositivo : torch.device
        Dispositivo donde se ejecuta el modelo.

    Notes
    -----
    Sin sincronización, ``perf_counter`` mide el encolado del kernel y no el
    cómputo real. En HQCNN el sesgo sería especialmente grande por la
    simulación del circuito cuántico.
    """
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)
    elif dispositivo.type == "mps":
        torch.mps.synchronize()


def _iterar_lotes(cargador: DataLoader) -> Iterator[tuple]:
    """Reinicia el cargador indefinidamente para obtener lotes de medición."""
    while True:
        yield from cargador


def medir_inferencia_ms_por_lote(
    modelo: nn.Module,
    cargador: DataLoader,
    dispositivo: torch.device,
    *,
    lotes_calentamiento: int = LOTES_CALENTAMIENTO,
    lotes_medicion: int = LOTES_MEDICION,
) -> float:
    """Mide la mediana de milisegundos por lote en inferencia.

    Parameters
    ----------
    modelo : nn.Module
        Modelo en modo evaluación.
    cargador : DataLoader
        Cargador de validación o prueba.
    dispositivo : torch.device
        Dispositivo de cómputo.
    lotes_calentamiento : int
        Número de lotes descartados antes de medir (mínimo 3).
    lotes_medicion : int
        Número de lotes usados para calcular la mediana.

    Returns
    -------
    float
        Mediana de milisegundos por lote.

    Notes
    -----
    Protocolo exigido por el contrato de resultados:

    1. Descartar al menos ``lotes_calentamiento`` lotes (calentamiento).
    2. Sincronizar el dispositivo antes y después de cada medición.
    3. Reportar la **mediana** sobre varios lotes, no el promedio.
    """
    if lotes_calentamiento < LOTES_CALENTAMIENTO:
        raise ValueError(
            f"lotes_calentamiento debe ser >= {LOTES_CALENTAMIENTO}: {lotes_calentamiento}"
        )
    if lotes_medicion <= 0:
        raise ValueError(f"lotes_medicion debe ser > 0: {lotes_medicion}")

    modelo.eval()
    tiempos_ms: list[float] = []
    iterador = _iterar_lotes(cargador)

    for _ in range(lotes_calentamiento):
        lote = next(iterador)
        entradas = lote[0].to(dispositivo)
        sincronizar_dispositivo(dispositivo)
        with torch.inference_mode():
            _ = modelo(entradas)
        sincronizar_dispositivo(dispositivo)

    for _ in range(lotes_medicion):
        lote = next(iterador)
        entradas = lote[0].to(dispositivo)
        sincronizar_dispositivo(dispositivo)
        inicio = time.perf_counter()
        with torch.inference_mode():
            _ = modelo(entradas)
        sincronizar_dispositivo(dispositivo)
        fin = time.perf_counter()
        tiempos_ms.append((fin - inicio) * 1000.0)

    return float(statistics.median(tiempos_ms))
