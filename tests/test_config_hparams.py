"""Pruebas de hiperparámetros congelados (TASK-11 / config)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import (
    ExperimentConfig,
    cargar_hparams_congelados,
    n_capas_congelada,
)


def test_experiment_config_default_n_capas() -> None:
    assert ExperimentConfig().n_capas == 6


def test_cargar_hparams_congelados(tmp_path: Path) -> None:
    ruta = tmp_path / "selected_hparams.json"
    ruta.write_text(
        json.dumps({"n_capas": 6, "presupuesto": {"decision": "no-go"}}),
        encoding="utf-8",
    )
    payload = cargar_hparams_congelados(ruta)
    assert payload["n_capas"] == 6
    assert n_capas_congelada(ruta) == 6


def test_cargar_hparams_congelados_falta_archivo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No existe"):
        cargar_hparams_congelados(tmp_path / "missing.json")


def test_cargar_hparams_congelados_sin_n_capas(tmp_path: Path) -> None:
    ruta = tmp_path / "selected_hparams.json"
    ruta.write_text(json.dumps({"epocas": 15}), encoding="utf-8")
    with pytest.raises(ValueError, match="n_capas"):
        cargar_hparams_congelados(ruta)
