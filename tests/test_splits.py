"""Pruebas del protocolo de particiones (TASK-6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.data.dataset import MAPEO_CLASE
from src.data.splits import (
    FRACCIONES,
    TOLERANCIA_PROPORCION,
    _serializar_splits,
    cargar_splits,
    construir_particiones,
    generar_splits,
    hash_manifiesto_utilizable,
    obtener_indices,
)
from src.logging.records import CLASES_ORDEN


def _manifiesto_sintetico(n_por_clase: int = 25) -> pd.DataFrame:
  """Genera un manifiesto balanceado suficiente para k-fold estratificado."""
  filas: list[dict[str, object]] = []
  for clase in CLASES_ORDEN:
    for i in range(n_por_clase):
      filas.append(
        {
          "ruta_relativa": f"Training/{clase}/img_{clase}_{i:03d}.jpg",
          "particion_origen": "Training",
          "clase": clase,
          "modo": "RGB",
          "ancho": 32,
          "alto": 32,
          "corrupta": False,
          "sha256": f"{clase}_{i:03d}".ljust(64, "0"),
          "excluida": False,
          "motivo_exclusion": "",
        }
      )
  return pd.DataFrame(filas).sort_values("ruta_relativa", kind="mergesort").reset_index(drop=True)


@pytest.fixture
def particiones_sinteticas() -> tuple[np.ndarray, list[dict]]:
  manifiesto = _manifiesto_sintetico()
  etiquetas = manifiesto["clase"].map(MAPEO_CLASE).to_numpy()
  folds = construir_particiones(etiquetas, n_folds=5, semilla=42)
  return etiquetas, folds


def test_val_identico_en_todas_fracciones(particiones_sinteticas) -> None:
  _, folds = particiones_sinteticas
  for fold in folds:
    val = fold["val"]
    for fraccion in ("0.10", "0.25", "0.50", "1.00"):
      train, val_obtenido = obtener_indices({"folds": [fold]}, fold["fold"], float(fraccion))
      assert val_obtenido == val
      assert len(train) > 0


def test_anidamiento_train(particiones_sinteticas) -> None:
  _, folds = particiones_sinteticas
  for fold in folds:
    train_10, _ = obtener_indices({"folds": [fold]}, fold["fold"], 0.10)
    train_25, _ = obtener_indices({"folds": [fold]}, fold["fold"], 0.25)
    train_50, _ = obtener_indices({"folds": [fold]}, fold["fold"], 0.50)
    train_100, _ = obtener_indices({"folds": [fold]}, fold["fold"], 1.00)
    assert set(train_10) <= set(train_25) <= set(train_50) <= set(train_100)


def test_estratificacion_dentro_tolerancia(particiones_sinteticas) -> None:
  etiquetas, folds = particiones_sinteticas
  proporciones_globales = {
    int(valor): float(conteo) / len(etiquetas)
    for valor, conteo in zip(*np.unique(etiquetas, return_counts=True), strict=True)
  }
  for fold in folds:
    for fraccion in ("0.10", "0.25", "0.50", "1.00"):
      indices = np.array(fold["train"][fraccion])
      valores, conteos = np.unique(etiquetas[indices], return_counts=True)
      for valor, conteo in zip(valores, conteos, strict=True):
        prop_observada = conteo / len(indices)
        prop_global = proporciones_globales[int(valor)]
        assert abs(prop_observada - prop_global) <= TOLERANCIA_PROPORCION


def test_train_val_disjuntos(particiones_sinteticas) -> None:
  _, folds = particiones_sinteticas
  for fold in folds:
    val = set(fold["val"])
    for fraccion in ("0.10", "0.25", "0.50", "1.00"):
      train = set(fold["train"][fraccion])
      assert train.isdisjoint(val)


def test_hash_falla_si_manifiesto_cambia(tmp_path: Path) -> None:
  manifiesto = _manifiesto_sintetico()
  ruta_csv = tmp_path / "dataset_manifest.csv"
  manifiesto.to_csv(ruta_csv, index=False)

  cfg = ExperimentConfig(raiz_resultados=tmp_path, semilla=42, n_folds=5)
  generar_splits(cfg)
  ruta_json = tmp_path / "splits.json"

  cargar_splits(ruta_json, ruta_manifiesto=ruta_csv, validar_hash=True)

  manifiesto_mod = manifiesto.copy()
  manifiesto_mod.loc[0, "clase"] = "notumor"
  manifiesto_mod.to_csv(ruta_csv, index=False)

  with pytest.raises(ValueError, match="El manifiesto cambió"):
    cargar_splits(ruta_json, ruta_manifiesto=ruta_csv, validar_hash=True)


def test_determinismo_regeneracion(tmp_path: Path) -> None:
  manifiesto = _manifiesto_sintetico()
  ruta_csv = tmp_path / "dataset_manifest.csv"
  manifiesto.to_csv(ruta_csv, index=False)

  cfg = ExperimentConfig(raiz_resultados=tmp_path, semilla=42, n_folds=5)
  generar_splits(cfg)
  hash_1 = hashlib.sha256((tmp_path / "splits.json").read_bytes()).hexdigest()
  generar_splits(cfg)
  hash_2 = hashlib.sha256((tmp_path / "splits.json").read_bytes()).hexdigest()
  assert hash_1 == hash_2


@pytest.mark.skipif(
  not Path("results/dataset_manifest.csv").exists(),
  reason="Manifiesto real no disponible",
)
def test_piso_10_porciento() -> None:
  ruta_json = Path("results/splits.json")
  if not ruta_json.exists():
    generar_splits(ExperimentConfig())

  splits = cargar_splits(ruta_json)
  for fold in splits["folds"]:
    conteos = fold["conteos"]["train"]["0.10"]
    for clase in CLASES_ORDEN:
      assert conteos[clase] >= 130, f"Fold {fold['fold']}, clase {clase}: {conteos[clase]}"


def test_obtener_indices_acepta_fraccion_1_como_float() -> None:
  manifiesto = _manifiesto_sintetico()
  etiquetas = manifiesto["clase"].map(MAPEO_CLASE).to_numpy()
  folds = construir_particiones(etiquetas, n_folds=5, semilla=42)
  splits = {"folds": folds}
  train_a, val_a = obtener_indices(splits, 0, 1.0)
  train_b, val_b = obtener_indices(splits, 0, 1.00)
  assert train_a == train_b
  assert val_a == val_b


def test_serializacion_json_determinista() -> None:
  manifiesto = _manifiesto_sintetico()
  etiquetas = manifiesto["clase"].map(MAPEO_CLASE).to_numpy()
  folds = construir_particiones(etiquetas, n_folds=5, semilla=42)
  artefacto = {
    "version": 1,
    "semilla": 42,
    "n_folds": 5,
    "fracciones": list(FRACCIONES),
    "n_total": len(manifiesto),
    "manifest_hash": hash_manifiesto_utilizable(manifiesto),
    "folds": folds,
  }
  texto_1 = _serializar_splits(artefacto)
  texto_2 = _serializar_splits(json.loads(texto_1))
  assert texto_1 == texto_2
