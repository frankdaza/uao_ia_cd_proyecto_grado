"""Siembra determinista para reproducibilidad de corridas."""

import random

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


class WorkerInitFn:
    """Inicializador determinista de workers para ``DataLoader``.

    Se implementa como clase de módulo (no cierre local) para que sea
    serializable con el backend ``spawn`` de macOS y Windows.
    """

    __slots__ = ("_semilla",)

    def __init__(self, semilla: int) -> None:
        self._semilla = semilla

    def __call__(self, worker_id: int) -> None:
        set_seed(self._semilla + worker_id)


def make_worker_init_fn(semilla: int) -> WorkerInitFn:
    """Construye ``worker_init_fn`` determinista para ``DataLoader``.

    Parameters
    ----------
    semilla : int
        Semilla base de la corrida; se desplaza por el identificador del worker.

    Returns
    -------
    WorkerInitFn
        Objeto invocable compatible con ``DataLoader(worker_init_fn=...)``.
    """
    return WorkerInitFn(semilla)
