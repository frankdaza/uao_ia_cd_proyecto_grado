"""Pruebas de la ablación de profundidad L (TASK-11)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.data.splits import FRACCIONES
from src.experiments.ablacion_L import (
    COLUMNAS_ABLACION,
    CRITERIO_SELECCION,
    FilaAblacion,
    UMBRAL_HORAS_CAMPANA,
    decidir_viabilidad,
    estimar_horas_campana,
    guardar_ablacion_csv,
    guardar_selected_hparams,
    params_cuanticos,
    seleccionar_profundidad,
)


def _fila(n_capas: int, f1: float) -> FilaAblacion:
    return {
        "n_capas": n_capas,
        "f1_val_macro": f1,
        "accuracy_val": f1,
        "loss_val": 1.0 - f1,
        "segundos_por_epoca": float(n_capas * 10),
        "n_params_cuanticos": params_cuanticos(n_capas),
        "norma_gradiente_inicial": 1.0,
        "dispositivo": "cpu",
        "n_train": 100,
        "epocas": 5,
    }


def test_params_cuanticos() -> None:
    assert params_cuanticos(2) == 24
    assert params_cuanticos(4) == 48
    assert params_cuanticos(6) == 72


def test_estimar_horas_campana_pondera_por_fraccion() -> None:
    seg_ref = 100.0
    epocas = 15
    fraccion_ref = 0.25

    horas = estimar_horas_campana(seg_ref, epocas, fraccion_ref, fracciones=FRACCIONES)
    esperado_seg = 0.0
    for fraccion in FRACCIONES:
        factor = fraccion / fraccion_ref
        esperado_seg += seg_ref * factor * epocas * 3 * 5
    assert horas == pytest.approx(esperado_seg / 3600.0)


def test_estimar_horas_campana_hqcnn_solo() -> None:
    horas_tres = estimar_horas_campana(100.0, 15, 0.25, fracciones=FRACCIONES)
    horas_uno = estimar_horas_campana(
        100.0, 15, 0.25, fracciones=FRACCIONES, n_modelos=1
    )
    assert horas_uno == pytest.approx(horas_tres / 3.0)


def test_estimar_horas_campana_rechaza_fraccion_cero() -> None:
    with pytest.raises(ValueError, match="fraccion_referencia"):
        estimar_horas_campana(10.0, 15, 0.0)


def test_seleccionar_profundidad_por_mayor_f1() -> None:
    filas = [_fila(2, 0.50), _fila(4, 0.62), _fila(6, 0.58)]
    assert seleccionar_profundidad(filas) == 4


def test_seleccionar_profundidad_desempate_hacia_l_menor() -> None:
    filas = [_fila(2, 0.60), _fila(4, 0.615), _fila(6, 0.61)]
    assert seleccionar_profundidad(filas, tolerancia=0.02) == 2


def test_seleccionar_profundidad_rechaza_lista_vacia() -> None:
    with pytest.raises(ValueError, match="No hay filas"):
        seleccionar_profundidad([])


def test_decidir_viabilidad_go() -> None:
    resultado = decidir_viabilidad(71.9, 24.0, umbral_horas=72.0)
    assert resultado.decision == "go"
    assert resultado.decision_d2 == "confirmada"
    assert resultado.mitigacion_adoptada is None


def test_decidir_viabilidad_no_go() -> None:
    resultado = decidir_viabilidad(72.1, 24.0, umbral_horas=72.0)
    assert resultado.decision == "no-go"
    assert resultado.decision_d2 == "ajustada"
    assert resultado.mitigacion_adoptada is None
    assert "sonda" in resultado.notas.lower()


def test_guardar_ablacion_csv_esquema(tmp_path: Path) -> None:
    filas = [_fila(2, 0.5), _fila(4, 0.6)]
    ruta = tmp_path / "ablacion_L.csv"
    guardar_ablacion_csv(filas, ruta)

    with ruta.open(encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == list(COLUMNAS_ABLACION)
        filas_leidas = list(lector)
    assert len(filas_leidas) == 2
    assert int(filas_leidas[0]["n_capas"]) == 2


def test_guardar_selected_hparams_esquema(tmp_path: Path) -> None:
    ruta = tmp_path / "selected_hparams.json"
    guardar_selected_hparams(
        n_capas=4,
        criterio=CRITERIO_SELECCION,
        protocolo={"fold": 0, "data_fraction": 0.25},
        presupuesto={"decision": "go", "horas_campana_estimadas": 50.0},
        ruta=ruta,
    )
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    assert payload["n_capas"] == 4
    assert payload["criterio_seleccion"] == CRITERIO_SELECCION
    assert "commit_sha" in payload
    assert payload["presupuesto"]["decision"] == "go"


def test_umbral_constante() -> None:
    assert UMBRAL_HORAS_CAMPANA == 72.0
