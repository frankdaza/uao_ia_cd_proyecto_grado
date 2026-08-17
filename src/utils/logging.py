"""Configuración de logging para scripts CLI del proyecto."""

from __future__ import annotations

import logging
import sys


def configurar_logging_cli(nivel: int = logging.INFO) -> None:
    """Configura logging en stderr para scripts ejecutados como módulo.

    Parameters
    ----------
    nivel : int
        Nivel de logging (por defecto ``logging.INFO``).

    Notes
    -----
    Usa ``force=True`` para que la configuración aplique aunque otro módulo
  ya haya configurado el root logger. El handler apunta a ``stderr`` para no
    interferir con barras ``tqdm`` en la misma stream.
    """
    logging.basicConfig(
        level=nivel,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
