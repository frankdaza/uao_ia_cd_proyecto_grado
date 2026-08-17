"""Utilidades transversales: semilla y dispositivo."""

from src.utils.device import get_device, log_dispositivo
from src.utils.seed import make_worker_init_fn, set_seed

__all__ = [
    "get_device",
    "log_dispositivo",
    "make_worker_init_fn",
    "set_seed",
]
