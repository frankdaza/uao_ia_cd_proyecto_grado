"""Pruebas del Dataset y DataLoader (TASK-5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import MAPEO_CLASE, MRIDataset, cargar_manifiesto_utilizable
from src.data.loaders import construir_loader
from src.data.transforms import TAMANO_ENTRADA, construir_transformaciones
from src.logging.records import CLASES_ORDEN


def _crear_imagen(ruta: Path, modo: str = "RGB", tamano: int = 32) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if modo == "L":
        datos = [(i * 5) % 256 for i in range(tamano * tamano)]
        Image.frombytes("L", (tamano, tamano), bytes(datos)).save(ruta)
    else:
        Image.new("RGB", (tamano, tamano), color=(50, 100, 150)).save(ruta)


@pytest.fixture
def entorno_mini(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    raiz = tmp_path / "datos"
    filas = [
        {
            "ruta_relativa": "Training/glioma/img_000.jpg",
            "particion_origen": "Training",
            "clase": "glioma",
            "modo": "L",
            "ancho": 32,
            "alto": 32,
            "corrupta": False,
            "sha256": "a" * 64,
            "excluida": False,
            "motivo_exclusion": "",
        },
        {
            "ruta_relativa": "Training/meningioma/img_001.jpg",
            "particion_origen": "Training",
            "clase": "meningioma",
            "modo": "RGB",
            "ancho": 32,
            "alto": 32,
            "corrupta": False,
            "sha256": "b" * 64,
            "excluida": False,
            "motivo_exclusion": "",
        },
        {
            "ruta_relativa": "Testing/pituitary/img_002.jpg",
            "particion_origen": "Testing",
            "clase": "pituitary",
            "modo": "RGB",
            "ancho": 32,
            "alto": 32,
            "corrupta": False,
            "sha256": "c" * 64,
            "excluida": True,
            "motivo_exclusion": "duplicado_exacto",
        },
    ]
    for fila in filas:
        _crear_imagen(raiz / fila["ruta_relativa"], modo=str(fila["modo"]))

    manifiesto = pd.DataFrame(filas)
    ruta_csv = tmp_path / "manifest.csv"
    manifiesto.to_csv(ruta_csv, index=False)
    return raiz, manifiesto


def test_cargar_manifiesto_utilizable_filtra_y_ordena(tmp_path: Path, entorno_mini) -> None:
    _, manifiesto = entorno_mini
    ruta_csv = tmp_path / "manifest.csv"
    utilizables = cargar_manifiesto_utilizable(ruta_csv)

    assert len(utilizables) == 2
    assert not utilizables["excluida"].any()
    assert list(utilizables["ruta_relativa"]) == sorted(
        utilizables["ruta_relativa"],
        key=str,
    )


def test_mapeo_clases_alineado_con_records() -> None:
    assert list(MAPEO_CLASE.keys()) == list(CLASES_ORDEN)
    assert MAPEO_CLASE["glioma"] == 0
    assert MAPEO_CLASE["notumor"] == 3


def test_validacion_sin_aumento_es_determinista(entorno_mini) -> None:
    raiz, manifiesto = entorno_mini
    utilizables = manifiesto[~manifiesto["excluida"]].reset_index(drop=True)
    transform = construir_transformaciones(aumentar=False)
    dataset = MRIDataset(utilizables, raiz, transform)

    tensor_1, _ = dataset[0]
    tensor_2, _ = dataset[0]
    assert torch.equal(tensor_1, tensor_2)


def test_dataset_contrato_tensor(entorno_mini) -> None:
    raiz, manifiesto = entorno_mini
    utilizables = manifiesto[~manifiesto["excluida"]].reset_index(drop=True)
    dataset = MRIDataset(
        utilizables,
        raiz,
        construir_transformaciones(aumentar=False),
    )

    tensor, clase_id = dataset[0]
    assert tensor.shape == (3, TAMANO_ENTRADA, TAMANO_ENTRADA)
    assert tensor.dtype == torch.float32
    assert clase_id == MAPEO_CLASE["glioma"]


def test_loader_determinista_con_workers(entorno_mini) -> None:
    raiz, manifiesto = entorno_mini
    utilizables = manifiesto[~manifiesto["excluida"]].reset_index(drop=True)
    dataset = MRIDataset(
        utilizables,
        raiz,
        construir_transformaciones(aumentar=False),
    )

    def primer_lote(semilla: int) -> torch.Tensor:
        loader = construir_loader(
            dataset,
            batch_size=2,
            mezclar=True,
            semilla=semilla,
            num_workers=2,
        )
        imagenes, _ = next(iter(loader))
        return imagenes

    assert torch.equal(primer_lote(42), primer_lote(42))
