"""Pruebas de la fábrica de transformaciones (TASK-5)."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.data.transforms import (
    DESV_IMAGENET,
    MEDIA_IMAGENET,
    TAMANO_ENTRADA,
    construir_transformaciones,
)


def _imagen_pil(modo: str = "RGB", tamano: int = 64) -> Image.Image:
    if modo == "L":
        datos = [(i * 3) % 256 for i in range(tamano * tamano)]
        return Image.frombytes("L", (tamano, tamano), bytes(datos))
    return Image.new(modo, (tamano, tamano), color=(120, 80, 200))


def test_validacion_sin_aumento_es_determinista() -> None:
    transform = construir_transformaciones(aumentar=False)
    imagen = _imagen_pil()
    tensor_1 = transform(imagen)
    tensor_2 = transform(imagen)
    assert torch.equal(tensor_1, tensor_2)


def test_entrenamiento_con_aumento_varia() -> None:
    transform = construir_transformaciones(aumentar=True)
    imagen = _imagen_pil(tamano=128)
    torch.manual_seed(0)
    tensor_1 = transform(imagen)
    torch.manual_seed(1)
    tensor_2 = transform(imagen)
    assert not torch.equal(tensor_1, tensor_2)


def test_tensor_contrato_forma_y_tipo() -> None:
    transform = construir_transformaciones(aumentar=False)
    tensor = transform(_imagen_pil())
    assert tensor.shape == (3, TAMANO_ENTRADA, TAMANO_ENTRADA)
    assert tensor.dtype == torch.float32


def test_imagen_gris_salida_tres_canales() -> None:
    transform = construir_transformaciones(aumentar=False)
    tensor = transform(_imagen_pil(modo="L"))
    assert tensor.shape == (3, TAMANO_ENTRADA, TAMANO_ENTRADA)


def test_normalizacion_imagenet_aplicada() -> None:
    transform = construir_transformaciones(aumentar=False)
    imagen = Image.new("RGB", (TAMANO_ENTRADA, TAMANO_ENTRADA), color=(255, 255, 255))
    tensor = transform(imagen)
    for canal, media, desv in zip(tensor, MEDIA_IMAGENET, DESV_IMAGENET, strict=True):
        assert canal.mean().item() == pytest.approx((1.0 - media) / desv)
