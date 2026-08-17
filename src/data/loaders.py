"""Constructores deterministas de DataLoader (TASK-5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MRIDataset
from src.data.transforms import construir_transformaciones
from src.utils.seed import make_worker_init_fn


def construir_loader(
    dataset: MRIDataset,
    *,
    batch_size: int,
    mezclar: bool,
    semilla: int,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """Construye un DataLoader reproducible incluso con varios workers.

    Parameters
    ----------
    dataset : MRIDataset
        Dataset de entrenamiento o validación.
    batch_size : int
        Tamaño de lote.
    mezclar : bool
        Si se mezclan los índices en cada época.
    semilla : int
        Semilla base de la corrida.
    num_workers : int
        Número de procesos de carga en paralelo.
    drop_last : bool
        Si es ``True``, descarta el último lote incompleto (recomendado en
        entrenamiento con lotes pequeños; TASK-6 / TASK-8).

    Returns
    -------
    DataLoader
        Cargador con ``generator`` y ``worker_init_fn`` explícitos.
    """
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=mezclar,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(semilla),
        generator=generador,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def construir_datasets(
    manifiesto_train: pd.DataFrame,
    manifiesto_val: pd.DataFrame,
    raiz: Path,
) -> tuple[MRIDataset, MRIDataset]:
    """Construye datasets de entrenamiento y validación con transformaciones distintas.

    Parameters
    ----------
    manifiesto_train : pd.DataFrame
        Subconjunto de entrenamiento del manifiesto.
    manifiesto_val : pd.DataFrame
        Subconjunto de validación del manifiesto.
    raiz : Path
        Raíz del dataset.

    Returns
    -------
    tuple[MRIDataset, MRIDataset]
        Par ``(train, val)``; solo el de entrenamiento aplica aumento.
    """
    transform_train = construir_transformaciones(aumentar=True)
    transform_val = construir_transformaciones(aumentar=False)
    return (
        MRIDataset(manifiesto_train, raiz, transform_train),
        MRIDataset(manifiesto_val, raiz, transform_val),
    )
