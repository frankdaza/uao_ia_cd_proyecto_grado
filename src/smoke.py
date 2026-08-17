"""Verificación mínima de reproducibilidad del entorno."""

import logging
import math

import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.utils.device import get_device, log_dispositivo
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _ejecutar_mini_entrenamiento(cfg: ExperimentConfig, dispositivo: torch.device) -> float:
    """Entrena un paso de un modelo lineal en datos sintéticos fijos.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración del experimento con la semilla y tamaño de lote.
    dispositivo : torch.device
        Dispositivo donde se ejecuta el mini entrenamiento.

    Returns
    -------
    float
        Pérdida tras un paso de optimización.
    """
    set_seed(cfg.semilla)
    generador = torch.Generator()
    generador.manual_seed(cfg.semilla)

    entradas = torch.randn(cfg.batch_size, 16, generator=generador).to(dispositivo)
    etiquetas = torch.randint(
        0,
        cfg.n_clases,
        (cfg.batch_size,),
        generator=generador,
    ).to(dispositivo)

    modelo = nn.Linear(16, cfg.n_clases).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=cfg.lr)

    optimizador.zero_grad()
    salida = modelo(entradas)
    perdida = criterio(salida, etiquetas)
    perdida.backward()
    optimizador.step()

    return float(perdida.item())


def verificar_reproducibilidad(cfg: ExperimentConfig) -> tuple[bool, float, str]:
    """Compara dos mini entrenamientos con la misma semilla y dispositivo.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración del experimento.

    Returns
    -------
    tuple[bool, float, str]
        Tupla con el resultado PASS/FAIL, la pérdida de referencia y el dispositivo.
    """
    cfg.ensure_layout()
    dispositivo = get_device()
    nombre_dispositivo = log_dispositivo(dispositivo)

    perdida_1 = _ejecutar_mini_entrenamiento(cfg, dispositivo)
    perdida_2 = _ejecutar_mini_entrenamiento(cfg, dispositivo)

    reproducible = math.isclose(perdida_1, perdida_2, rel_tol=0.0, abs_tol=0.0)
    return reproducible, perdida_1, nombre_dispositivo


def main() -> None:
    """Ejecuta la verificación de reproducibilidad y reporta PASS o FAIL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = ExperimentConfig()
    reproducible, perdida, dispositivo = verificar_reproducibilidad(cfg)

    estado = "PASS" if reproducible else "FAIL"
    logger.info("Pérdida de referencia: %.8f", perdida)
    logger.info("Resultado reproducibilidad: %s (dispositivo=%s)", estado, dispositivo)

    if not reproducible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
