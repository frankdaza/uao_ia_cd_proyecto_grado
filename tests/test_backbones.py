"""Pruebas de la fábrica de backbones y cabeza intercambiable (TASK-7)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.backbones import (
    BACKBONES_SOPORTADOS,
    VERSIONES_PESOS,
    build_backbone,
    contar_parametros,
    obtener_version_pesos,
    parametros_entrenables,
)
from src.models.heads import CabeceraReduccion

_BATCH = 2
_ENTRADA = (3, 224, 224)
_DIM_SALIDA = 4


@pytest.fixture(scope="module")
def backbone_efficientnet() -> tuple[nn.Module, int]:
    return build_backbone("efficientnet_b0")


@pytest.fixture(scope="module")
def backbone_resnet() -> tuple[nn.Module, int]:
    return build_backbone("resnet50")


def test_versiones_pesos_registradas() -> None:
    for nombre in BACKBONES_SOPORTADOS:
        assert obtener_version_pesos(nombre) == VERSIONES_PESOS[nombre]


def test_dimension_salida_efficientnet_b0(backbone_efficientnet: tuple[nn.Module, int]) -> None:
    backbone, dimension = backbone_efficientnet
    assert dimension == 1280
    x = torch.randn(_BATCH, *_ENTRADA)
    salida = backbone(x)
    assert salida.shape == (_BATCH, 1280)


def test_dimension_salida_resnet50(backbone_resnet: tuple[nn.Module, int]) -> None:
    backbone, dimension = backbone_resnet
    assert dimension == 2048
    x = torch.randn(_BATCH, *_ENTRADA)
    salida = backbone(x)
    assert salida.shape == (_BATCH, 2048)


def test_backbone_congelado_sin_gradiente(backbone_efficientnet: tuple[nn.Module, int]) -> None:
    backbone, dimension = backbone_efficientnet
    cabecera = CabeceraReduccion(dimension, _DIM_SALIDA)
    x = torch.randn(_BATCH, *_ENTRADA)
    logits = cabecera(backbone(x))
    perdida = logits.sum()
    perdida.backward()

    for parametro in backbone.parameters():
        assert parametro.grad is None
    for parametro in cabecera.parameters():
        assert parametro.grad is not None


def test_backbone_en_eval(backbone_efficientnet: tuple[nn.Module, int]) -> None:
    backbone, _ = backbone_efficientnet
    assert not backbone.training


def test_contar_parametros_backbone_solo_congelados(
    backbone_efficientnet: tuple[nn.Module, int],
    backbone_resnet: tuple[nn.Module, int],
) -> None:
    for backbone, _ in (backbone_efficientnet, backbone_resnet):
        conteo = contar_parametros(backbone)
        assert conteo["entrenables"] == 0
        assert conteo["congelados"] > 0


def test_contar_parametros_con_cabecera(backbone_efficientnet: tuple[nn.Module, int]) -> None:
    backbone, dimension = backbone_efficientnet
    modelo = nn.Sequential(backbone, CabeceraReduccion(dimension, _DIM_SALIDA))
    conteo = contar_parametros(modelo)
    assert conteo["entrenables"] == dimension * _DIM_SALIDA + _DIM_SALIDA
    assert conteo["congelados"] > 0


def test_parametros_entrenables_para_optimizador(backbone_efficientnet: tuple[nn.Module, int]) -> None:
    backbone, dimension = backbone_efficientnet
    cabecera = CabeceraReduccion(dimension, _DIM_SALIDA)
    modelo = nn.Sequential(backbone, cabecera)
    entrenables = parametros_entrenables(modelo)
    conteo = contar_parametros(modelo)

    assert len(entrenables) == 2  # peso y sesgo de la capa lineal
    assert sum(p.numel() for p in entrenables) == conteo["entrenables"]
    assert all(p.requires_grad for p in entrenables)
    assert not any(p.requires_grad for p in backbone.parameters())

    optimizador = torch.optim.Adam(entrenables, lr=1e-3)
    assert len(optimizador.param_groups[0]["params"]) == len(entrenables)


def test_cabecera_reduccion_intercambiable(
    backbone_efficientnet: tuple[nn.Module, int],
    backbone_resnet: tuple[nn.Module, int],
) -> None:
    for backbone, dimension in (backbone_efficientnet, backbone_resnet):
        cabecera = CabeceraReduccion(dimension, _DIM_SALIDA)
        x = torch.randn(_BATCH, *_ENTRADA)
        salida = cabecera(backbone(x))
        assert salida.shape == (_BATCH, _DIM_SALIDA)


def test_backbone_no_soportado() -> None:
    with pytest.raises(ValueError, match="Backbone no soportado"):
        build_backbone("vgg16")

    with pytest.raises(ValueError, match="Backbone no soportado"):
        obtener_version_pesos("vgg16")
