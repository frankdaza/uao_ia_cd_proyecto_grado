"""Línea base clásica con extractor congelado (TASK-12 / A6)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.models.backbones import build_backbone
from src.models.heads import CabeceraReduccion


class ClassicalBaseline(nn.Module):
    """Línea base clásica: backbone ImageNet congelado + cabeza lineal.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con ``n_clases``.
    backbone : str
        Identificador del extractor (``efficientnet_b0`` o ``resnet50``).

    Notes
    -----
    Sobrescribe ``train`` para mantener el backbone en ``eval()``: de lo contrario
    BatchNorm actualizaría estadísticas móviles con lotes pequeños del escenario
    de escasez (misma restricción que HQCNN, TASK-7/10).
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        backbone: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        self._nombre_backbone = backbone
        self.backbone, dim_latente = build_backbone(backbone)
        self.cabecera = CabeceraReduccion(dim_latente, cfg.n_clases)

    @property
    def nombre_backbone(self) -> str:
        """Identificador del backbone usado (p. ej. ``efficientnet_b0``)."""
        return self._nombre_backbone

    def train(self, modo: bool = True) -> ClassicalBaseline:
        """Cambia de modo manteniendo el backbone congelado en evaluación."""
        super().train(modo)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga de la imagen a los logits de clasificación.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de forma ``(B, 3, 224, 224)`` o ``(3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Logits de forma ``(B, n_clases)`` o ``(n_clases,)``.
        """
        unica_muestra = x.ndim == 3
        if unica_muestra:
            x = x.unsqueeze(0)

        with torch.no_grad():
            latente = self.backbone(x)
        logits = self.cabecera(latente)

        if unica_muestra:
            return logits.squeeze(0)
        return logits
