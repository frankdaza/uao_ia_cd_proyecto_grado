"""Cabezas intercambiables para líneas base y modelo híbrido (TASK-7)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CabeceraReduccion(nn.Module):
    """Capa lineal que reduce el espacio latente a la dimensión del clasificador.

    Parameters
    ----------
    dim_entrada : int
        Dimensión del vector de características del backbone (1280 o 2048).
    dim_salida : int
        Dimensión de salida; por defecto 4 (número de clases / qubits del VQC).

    Notes
    -----
    La misma cabeza alimenta la línea base clásica (TASK-12) y la reducción
    previa al VQC (TASK-10), garantizando que HQCNN y baselines comparten el
    mismo extractor congelado.
    """

    def __init__(self, dim_entrada: int, dim_salida: int = 4) -> None:
        super().__init__()
        self.lineal = nn.Linear(dim_entrada, dim_salida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la reducción lineal al vector de características.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de forma ``(B, dim_entrada)``.

        Returns
        -------
        torch.Tensor
            Tensor de forma ``(B, dim_salida)``.
        """
        return self.lineal(x)
