"""Dataset de MRI cerebral construido desde el manifiesto auditado (TASK-5)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.logging.records import CLASES_ORDEN

MAPEO_CLASE: dict[str, int] = {nombre: indice for indice, nombre in enumerate(CLASES_ORDEN)}


def cargar_manifiesto_utilizable(ruta: Path) -> pd.DataFrame:
    """Carga el manifiesto auditado y devuelve solo filas utilizables.

    Parameters
    ----------
    ruta : Path
        Ruta a ``dataset_manifest.csv``.

    Returns
    -------
    pd.DataFrame
        Filas no excluidas, ordenadas por ``ruta_relativa`` para índices
        posicionales estables (contrato con TASK-6).
    """
    manifiesto = pd.read_csv(ruta)
    utilizables = manifiesto[~manifiesto["excluida"]].copy()
    return utilizables.sort_values("ruta_relativa", kind="mergesort").reset_index(drop=True)


class MRIDataset(Dataset):
    """Dataset de MRI cerebral construido desde el manifiesto auditado."""

    def __init__(
        self,
        manifiesto: pd.DataFrame,
        raiz: Path,
        transformacion: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        """Inicializa el dataset.

        Parameters
        ----------
        manifiesto : pd.DataFrame
            Subconjunto del manifiesto con columnas ``ruta_relativa`` y ``clase``.
        raiz : Path
            Raíz del dataset (``ExperimentConfig.raiz_datos``).
        transformacion : Callable
            Pipeline de transformaciones inyectado (fábrica de TASK-5).
        """
        self._filas = manifiesto.reset_index(drop=True)
        self._raiz = raiz
        self._transformacion = transformacion

    def __len__(self) -> int:
        return len(self._filas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        fila = self._filas.iloc[indice]
        with Image.open(self._raiz / fila["ruta_relativa"]) as img:
            imagen = img.convert("RGB")
        clase_id = MAPEO_CLASE[str(fila["clase"])]
        return self._transformacion(imagen), clase_id
