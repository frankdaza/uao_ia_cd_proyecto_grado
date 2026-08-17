"""Entrenamiento unificado para modelos clásicos e híbridos (TASK-8)."""

from src.train.dataloading import construir_loaders_para_fold
from src.train.trainer import Trainer, nombre_pesos

__all__ = [
    "Trainer",
    "construir_loaders_para_fold",
    "nombre_pesos",
]
