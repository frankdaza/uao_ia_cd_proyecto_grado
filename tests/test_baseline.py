"""Pruebas de línea base clásica, fábrica y orquestador (TASK-12)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config import ExperimentConfig
from src.experiments.baselines import (
    FRACCION_BASELINE,
    consolidar_metricas,
    verificar_indices_fold,
)
from src.logging.records import RunRecord
from src.models.baseline import ClassicalBaseline
from src.models.factory import build_model
from src.models.hqcnn import HQCNN


def _cfg() -> ExperimentConfig:
    return ExperimentConfig()


def _registro_sintetico(
    *,
    modelo: str = "efficientnet_b0",
    fold: int = 0,
    accuracy_val: float = 0.8,
) -> RunRecord:
    return RunRecord(
        modelo=modelo,
        data_fraction=FRACCION_BASELINE,
        fold=fold,
        semilla=42,
        dispositivo="cpu",
        n_train=100,
        n_val=20,
        epocas=15,
        n_params_entrenables=5124,
        n_capas_vqc=None,
        commit_sha="abc1234",
        timestamp="2026-08-17T01:00:00+00:00",
        accuracy_train=0.85,
        accuracy_val=accuracy_val,
        loss_train=0.5,
        loss_val=0.6,
        f1_val_weighted=0.78,
        f1_val_macro=0.77,
        sensibilidad_por_clase={
            "glioma": 0.7,
            "meningioma": 0.8,
            "pituitary": 0.75,
            "notumor": 0.85,
        },
        especificidad_por_clase={
            "glioma": 0.9,
            "meningioma": 0.88,
            "pituitary": 0.92,
            "notumor": 0.87,
        },
        train_time_s=120.0,
        inference_ms_per_batch=15.0,
    )


@pytest.fixture(scope="module")
def modelo_efficientnet() -> ClassicalBaseline:
    return ClassicalBaseline(_cfg(), backbone="efficientnet_b0")


def test_classical_baseline_forma_salida(modelo_efficientnet: ClassicalBaseline) -> None:
    x = torch.randn(2, 3, 224, 224)
    salida = modelo_efficientnet(x)
    assert salida.shape == (2, 4)


def test_classical_baseline_backbone_sin_gradiente(
    modelo_efficientnet: ClassicalBaseline,
) -> None:
    x = torch.randn(2, 3, 224, 224)
    perdida = modelo_efficientnet(x).sum()
    perdida.backward()
    for parametro in modelo_efficientnet.backbone.parameters():
        assert parametro.grad is None
    for parametro in modelo_efficientnet.cabecera.parameters():
        assert parametro.grad is not None


def test_classical_baseline_backbone_en_eval_tras_train(
    modelo_efficientnet: ClassicalBaseline,
) -> None:
    modelo_efficientnet.train()
    assert not modelo_efficientnet.backbone.training


def test_build_model_efficientnet() -> None:
    modelo = build_model("efficientnet_b0", _cfg())
    assert isinstance(modelo, ClassicalBaseline)


def test_build_model_resnet50() -> None:
    modelo = build_model("resnet50", _cfg())
    assert isinstance(modelo, ClassicalBaseline)
    assert modelo.nombre_backbone == "resnet50"


def test_build_model_hqcnn() -> None:
    modelo = build_model("hqcnn", _cfg())
    assert isinstance(modelo, HQCNN)


def test_build_model_invalido() -> None:
    with pytest.raises(ValueError, match="Modelo no soportado"):
        build_model("vgg16", _cfg())


def test_consolidar_metricas_media_y_desv() -> None:
    registros = [
        _registro_sintetico(fold=0, accuracy_val=0.80),
        _registro_sintetico(fold=1, accuracy_val=0.90),
    ]
    resumen = consolidar_metricas(registros)
    assert resumen["efficientnet_b0"]["accuracy_val"]["media"] == pytest.approx(0.85)
    assert resumen["efficientnet_b0"]["accuracy_val"]["desv_std"] == pytest.approx(
        0.070710678,
        rel=1e-3,
    )


def test_verificar_indices_fold_coherente() -> None:
    cfg = replace(ExperimentConfig(), data_fraction=FRACCION_BASELINE)
    if not (cfg.raiz_resultados / "splits.json").exists():
        pytest.skip("splits.json no generado")
    if not (cfg.raiz_resultados / "dataset_manifest.csv").exists():
        pytest.skip("dataset_manifest.csv no generado")
    train_idx, val_idx = verificar_indices_fold(cfg, fold=0)
    assert len(train_idx) > 0
    assert len(val_idx) > 0
