"""Pruebas de auditoría del dataset (TASK-3)."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.audit import (
    COLUMNAS_MANIFIESTO,
    analizar_duplicados,
    aplicar_exclusiones,
    construir_manifiesto,
    sha256_archivo,
)


def _guardar_jpg(ruta: Path, color: tuple[int, int, int] = (10, 20, 30)) -> str:
    """Crea un JPEG mínimo y devuelve su hash SHA-256."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(ruta, format="JPEG")
    return sha256_archivo(ruta)


def _fila(
    ruta: str,
    *,
    particion: str = "Training",
    clase: str = "glioma",
    digest: str = "a" * 64,
    corrupta: bool = False,
) -> dict[str, object]:
    return {
        "ruta_relativa": ruta,
        "particion_origen": particion,
        "clase": clase,
        "modo": "RGB",
        "ancho": 8,
        "alto": 8,
        "corrupta": corrupta,
        "sha256": digest,
        "excluida": False,
        "motivo_exclusion": "",
    }


@pytest.fixture
def raiz_mini(tmp_path: Path) -> Path:
    """Dataset sintético con duplicados, fuga y corrupta."""
    raiz = tmp_path / "dataset"
    digest_dup = _guardar_jpg(raiz / "Training/glioma/original.jpg", (1, 2, 3))
    _guardar_jpg(raiz / "Training/glioma/copia.jpg", (1, 2, 3))
    _guardar_jpg(raiz / "Testing/glioma/fuga.jpg", (1, 2, 3))
    _guardar_jpg(raiz / "Training/meningioma/unica.jpg", (4, 5, 6))

    # Mismo contenido, etiquetas distintas
    digest_conflicto = _guardar_jpg(raiz / "Training/pituitary/conf_a.jpg", (7, 8, 9))
    _guardar_jpg(raiz / "Training/notumor/conf_b.jpg", (7, 8, 9))
    assert sha256_archivo(raiz / "Training/notumor/conf_b.jpg") == digest_conflicto

    (raiz / "Training/glioma/corrupta.jpg").write_bytes(b"no-es-jpeg")
    return raiz


def test_construir_manifiesto_rutas_relativas_y_estables(raiz_mini: Path) -> None:
    manifiesto, extensiones = construir_manifiesto(raiz_mini)

    assert extensiones == {".jpg": manifiesto.shape[0]}
    assert (manifiesto["ruta_relativa"].str.contains("/Users|Drive|^/", regex=True)).sum() == 0
    assert list(manifiesto.columns) == COLUMNAS_MANIFIESTO
    assert manifiesto["ruta_relativa"].is_monotonic_increasing


def test_aplicar_exclusiones_reglas_auditables(raiz_mini: Path) -> None:
    manifiesto, _ = construir_manifiesto(raiz_mini)
    duplicados = analizar_duplicados(manifiesto)
    resultado = aplicar_exclusiones(manifiesto)

    assert duplicados.entre_clases >= 1
    assert duplicados.train_test >= 1

    corruptas = resultado[resultado["motivo_exclusion"] == "corrupta"]
    assert len(corruptas) == 1

    contradictorias = resultado[resultado["motivo_exclusion"] == "etiqueta_contradictoria"]
    assert len(contradictorias) == 2

    fuga = resultado[resultado["motivo_exclusion"] == "fuga_train_test"]
    assert len(fuga) >= 1
    assert (fuga["particion_origen"] == "Testing").all()

    duplicados_exactos = resultado[resultado["motivo_exclusion"] == "duplicado_exacto"]
    assert len(duplicados_exactos) >= 1

    utilizables = resultado[~resultado["excluida"]]
    hashes_activos = utilizables.groupby("sha256").size()
    assert (hashes_activos <= 1).all()


def test_una_copia_retenida_por_hash_en_grupo_intra_clase() -> None:
    digest = "d" * 64
    manifiesto = pd.DataFrame(
        [
            _fila("Training/glioma/a.jpg", digest=digest),
            _fila("Training/glioma/b.jpg", digest=digest),
            _fila("Training/glioma/c.jpg", digest=digest),
        ]
    )
    resultado = aplicar_exclusiones(manifiesto)
    activos = resultado[~resultado["excluida"]]

    assert len(activos) == 1
    assert activos.iloc[0]["ruta_relativa"] == "Training/glioma/a.jpg"
    assert (resultado.loc[resultado["excluida"], "motivo_exclusion"] == "duplicado_exacto").all()


def test_fuga_train_test_excluye_solo_testing() -> None:
    digest = "e" * 64
    manifiesto = pd.DataFrame(
        [
            _fila("Training/glioma/train.jpg", digest=digest),
            _fila("Testing/glioma/test.jpg", particion="Testing", digest=digest),
        ]
    )
    resultado = aplicar_exclusiones(manifiesto)

    fila_train = resultado[resultado["ruta_relativa"] == "Training/glioma/train.jpg"].iloc[0]
    fila_test = resultado[resultado["ruta_relativa"] == "Testing/glioma/test.jpg"].iloc[0]
    assert not fila_train["excluida"]
    assert fila_test["excluida"]
    assert fila_test["motivo_exclusion"] == "fuga_train_test"


def test_sha256_archivo_coincide_con_contenido(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="JPEG")
    contenido = buffer.getvalue()
    digest_esperado = hashlib.sha256(contenido).hexdigest()

    ruta = tmp_path / "prueba.jpg"
    ruta.write_bytes(contenido)
    assert sha256_archivo(ruta) == digest_esperado
