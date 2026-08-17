"""Construcción de DataLoaders desde particiones fijas (TASK-8)."""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import cargar_manifiesto_utilizable
from src.data.loaders import construir_datasets, construir_loader
from src.data.splits import cargar_splits, obtener_indices


def construir_loaders_para_fold(
    cfg: ExperimentConfig,
    fold: int,
    *,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Construye cargadores de entrenamiento y validación para un fold.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuración con fracción de datos, lote y semilla.
    fold : int
        Índice del fold estratificado (0 a ``n_folds - 1``).
    num_workers : int
        Procesos de carga en paralelo para cada DataLoader.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Par ``(train, val)`` con índices leídos de ``results/splits.json``.

    Notes
    -----
    No remuestrea: consume ``obtener_indices()`` tal cual. El cargador de
    entrenamiento usa ``drop_last=True`` para evitar lotes de tamaño 1 con
    BatchNorm (decisión documentada en TASK-6).
    """
    ruta_splits = cfg.raiz_resultados / "splits.json"
    splits = cargar_splits(ruta_splits)
    train_idx, val_idx = obtener_indices(splits, fold, cfg.data_fraction)

    ruta_manifiesto = cfg.raiz_resultados / "dataset_manifest.csv"
    manifiesto = cargar_manifiesto_utilizable(ruta_manifiesto)
    manifiesto_train = manifiesto.iloc[train_idx].reset_index(drop=True)
    manifiesto_val = manifiesto.iloc[val_idx].reset_index(drop=True)

    dataset_train, dataset_val = construir_datasets(
        manifiesto_train,
        manifiesto_val,
        cfg.raiz_datos,
    )

    cargador_train = construir_loader(
        dataset_train,
        batch_size=cfg.batch_size,
        mezclar=True,
        semilla=cfg.semilla,
        num_workers=num_workers,
        drop_last=True,
    )
    cargador_val = construir_loader(
        dataset_val,
        batch_size=cfg.batch_size,
        mezclar=False,
        semilla=cfg.semilla,
        num_workers=num_workers,
    )
    return cargador_train, cargador_val
