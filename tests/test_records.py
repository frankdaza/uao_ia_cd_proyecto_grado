"""Pruebas del contrato de registros (TASK-4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.logging.records import (
    COLUMNAS_CSV,
    EpochRecord,
    RunRecord,
    aplanar,
    especificidad_por_clase,
    sensibilidad_por_clase,
)


def _metricas_por_clase(valor: float = 0.8) -> dict[str, float]:
    return {
        "glioma": valor,
        "meningioma": valor,
        "pituitary": valor,
        "notumor": valor,
    }


def _registro_valido(**cambios) -> RunRecord:
    base = {
        "modelo": "hqcnn",
        "data_fraction": 1.0,
        "fold": 0,
        "semilla": 42,
        "dispositivo": "cpu",
        "n_train": 100,
        "n_val": 25,
        "epocas": 15,
        "n_params_entrenables": 1234,
        "n_capas_vqc": 4,
        "commit_sha": "abc1234",
        "timestamp": "2026-08-17T01:00:00+00:00",
        "accuracy_train": 0.9,
        "accuracy_val": 0.8,
        "loss_train": 0.4,
        "loss_val": 0.5,
        "f1_val_weighted": 0.79,
        "f1_val_macro": 0.78,
        "sensibilidad_por_clase": _metricas_por_clase(),
        "especificidad_por_clase": _metricas_por_clase(0.85),
        "train_time_s": 120.5,
        "inference_ms_per_batch": 12.3,
    }
    base.update(cambios)
    return RunRecord(**base)


def test_run_record_valido_pasa_validacion() -> None:
    registro = _registro_valido()
    registro.validar()
    assert registro.brecha_g == pytest.approx(0.1)


def test_run_record_rechaza_accuracy_fuera_de_rango() -> None:
    registro = _registro_valido(accuracy_val=1.5)
    with pytest.raises(ValueError, match="accuracy_val fuera de"):
        registro.validar()


def test_run_record_rechaza_clases_incompletas() -> None:
    registro = _registro_valido(
        sensibilidad_por_clase={"glioma": 0.8, "meningioma": 0.7, "pituitary": 0.6}
    )
    with pytest.raises(ValueError, match="sensibilidad"):
        registro.validar()


def test_aplanar_produce_todas_las_columnas_csv() -> None:
    fila = aplanar(_registro_valido())
    assert tuple(fila.keys()) == COLUMNAS_CSV
    assert fila["sens_glioma"] == 0.8
    assert fila["spec_notumor"] == 0.85
    assert fila["brecha_g"] == pytest.approx(0.1)


def test_aplanar_modelo_clasico_sin_capas_vqc() -> None:
    fila = aplanar(_registro_valido(modelo="efficientnet_b0", n_capas_vqc=None))
    assert fila["n_capas_vqc"] == ""


def test_epoch_record_valida_exactitudes() -> None:
    epoca = EpochRecord(
        epoca=0,
        loss_train=0.5,
        loss_val=0.6,
        accuracy_train=0.7,
        accuracy_val=0.65,
    )
    epoca.validar()


def test_epoch_record_rechaza_exactitud_invalida() -> None:
    epoca = EpochRecord(
        epoca=0,
        loss_train=0.5,
        loss_val=0.6,
        accuracy_train=1.2,
        accuracy_val=0.65,
    )
    with pytest.raises(ValueError, match="accuracy_train"):
        epoca.validar()


def test_sensibilidad_por_clase_orden_fijo() -> None:
    y_verdadero = np.array([0, 1, 2, 3, 0, 1])
    y_predicho = np.array([0, 1, 2, 2, 0, 3])
    resultado = sensibilidad_por_clase(y_verdadero, y_predicho)
    assert list(resultado.keys()) == ["glioma", "meningioma", "pituitary", "notumor"]
    assert resultado["pituitary"] == pytest.approx(1.0)
    assert resultado["notumor"] == pytest.approx(0.0)


def test_especificidad_por_clase_desde_matriz() -> None:
    matriz = np.array(
        [
            [10, 1, 0, 0],
            [2, 8, 0, 0],
            [0, 0, 7, 1],
            [0, 0, 1, 9],
        ]
    )
    resultado = especificidad_por_clase(matriz)
    assert set(resultado) == {"glioma", "meningioma", "pituitary", "notumor"}
    assert all(0.0 <= valor <= 1.0 for valor in resultado.values())
