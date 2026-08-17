"""Siembra determinista para reproducibilidad de corridas."""

import random
from collections.abc import Callable

import numpy as np
import torch


def set_seed(semilla: int) -> None:
    """Fija todas las fuentes de aleatoriedad de una corrida.

    Parameters
    ----------
    semilla : int
        Valor de semilla tomado de ``ExperimentConfig.semilla``.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_worker_init_fn(semilla: int) -> Callable[[int], None]:
    """Construye ``worker_init_fn`` determinista para ``DataLoader``.

    Parameters
    ----------
    semilla : int
        Semilla base de la corrida; se desplaza por el identificador del worker.

    Returns
    -------
    Callable[[int], None]
        Función compatible con ``DataLoader(worker_init_fn=...)``.
    """
    def worker_init_fn(worker_id: int) -> None:
        set_seed(semilla + worker_id)

    return worker_init_fn
