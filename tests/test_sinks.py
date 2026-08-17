"""Pruebas de sinks CSV, JSON y utilidades (TASK-4)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.config import ExperimentConfig
from src.logging.records import COLUMNAS_CSV, EpochRecord, RunRecord
from src.logging.sinks import (
    corrida_existe,
    escribir_corrida_csv,
    escribir_historial_json,
    nombre_historial,
    obtener_commit_sha,
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
        "data_fraction": 0.5,
        "fold": 2,
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


def test_obtener_commit_sha_no_vacio() -> None:
    sha = obtener_commit_sha()
    assert sha
    assert sha != ""


def test_escribir_corrida_csv_crea_cabecera_y_append(tmp_path: Path) -> None:
    ruta = tmp_path / "experiments.csv"
    registro_1 = _registro_valido(fold=0)
    registro_2 = _registro_valido(fold=1)

    escribir_corrida_csv(registro_1, ruta)
    escribir_corrida_csv(registro_2, ruta)

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        filas = list(lector)

    assert lector.fieldnames == list(COLUMNAS_CSV)
    assert len(filas) == 2
    assert filas[0]["fold"] == "0"
    assert filas[1]["fold"] == "1"


def test_escribir_corrida_csv_rechaza_cabecera_incompatible(tmp_path: Path) -> None:
    ruta = tmp_path / "experiments.csv"
    ruta.write_text("modelo,fold\nhqcnn,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cabecera"):
        escribir_corrida_csv(_registro_valido(), ruta)


def test_nombre_historial_incluye_semilla() -> None:
    registro = _registro_valido()
    assert nombre_historial(registro) == "hqcnn_0p5_f2_s42.json"


def test_escribir_historial_json(tmp_path: Path) -> None:
    cfg = ExperimentConfig(raiz_resultados=tmp_path)
    registro = _registro_valido()
    historial = [
        EpochRecord(
            epoca=0,
            loss_train=0.9,
            loss_val=0.8,
            accuracy_train=0.6,
            accuracy_val=0.55,
        ),
        EpochRecord(
            epoca=1,
            loss_train=0.7,
            loss_val=0.65,
            accuracy_train=0.7,
            accuracy_val=0.6,
        ),
    ]

    ruta = escribir_historial_json(registro, historial, cfg)
    assert ruta.name == "hqcnn_0p5_f2_s42.json"
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert set(payload[0]) == {
        "epoca",
        "loss_train",
        "loss_val",
        "accuracy_train",
        "accuracy_val",
    }


def test_corrida_existe_detecta_duplicado(tmp_path: Path) -> None:
    ruta = tmp_path / "experiments.csv"
    registro = _registro_valido()
    escribir_corrida_csv(registro, ruta)
    assert corrida_existe(registro, ruta) is True
    assert corrida_existe(_registro_valido(fold=9), ruta) is False
