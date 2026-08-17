"""Selección y registro del dispositivo de cómputo."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Devuelve el mejor dispositivo disponible: cuda > mps > cpu.

    Returns
    -------
    torch.device
        Dispositivo seleccionado según la prelación del proyecto.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_dispositivo(dispositivo: torch.device) -> str:
    """Registra en el log el dispositivo elegido para la corrida.

    Parameters
    ----------
    dispositivo : torch.device
        Dispositivo devuelto por ``get_device()`` o equivalente.

    Returns
    -------
    str
        Nombre canónico del dispositivo (``cuda``, ``mps`` o ``cpu``).
    """
    nombre = dispositivo.type
    logger.info("Dispositivo de cómputo: %s", nombre)
    return nombre
