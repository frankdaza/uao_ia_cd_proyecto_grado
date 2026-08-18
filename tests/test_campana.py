"""Pruebas del orquestador de campaña factorial (TASK-13)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import ExperimentConfig
from src.data.splits import FRACCIONES
from src.experiments.campana import (
    MODELOS,
    N_CELDAS_BASELINE_TASK12,
    N_CELDAS_TOTAL,
    ClaveCelda,
    _actualizar_celda_estado,
    _inicializar_estado_celdas,
    archivar_corridas_no_cuda,
    cargar_estado_campana,
    comparar_costo,
    generar_celdas_design,
    guardar_estado_campana,
    orden_modelos,
    verificar_integridad,
    verificar_precondiciones,
)
from src.logging.records import RunRecord
from src.logging.sinks import escribir_corrida_csv, escribir_historial_json


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        raiz_resultados=tmp_path / "results",
        raiz_modelos=tmp_path / "models",
        raiz_figuras=tmp_path / "results" / "figures",
    )


def _metricas_por_clase(valor: float = 0.8) -> dict[str, float]:
    return {
        "glioma": valor,
        "meningioma": valor,
        "pituitary": valor,
        "notumor": valor,
    }


def _registro_campana(
    *,
    modelo: str = "efficientnet_b0",
    data_fraction: float = 0.10,
    fold: int = 0,
    train_time_s: float = 100.0,
    n_capas_vqc: int | None = None,
    dispositivo: str = "cpu",
    epocas: int = 15,
) -> RunRecord:
    return RunRecord(
        modelo=modelo,
        data_fraction=data_fraction,
        fold=fold,
        semilla=42,
        dispositivo=dispositivo,
        n_train=100,
        n_val=25,
        epocas=epocas,
        n_params_entrenables=5124,
        n_capas_vqc=n_capas_vqc,
        commit_sha="abc1234",
        timestamp="2026-08-17T01:00:00+00:00",
        accuracy_train=0.85,
        accuracy_val=0.80,
        loss_train=0.5,
        loss_val=0.6,
        f1_val_weighted=0.79,
        f1_val_macro=0.78,
        sensibilidad_por_clase=_metricas_por_clase(),
        especificidad_por_clase=_metricas_por_clase(),
        train_time_s=train_time_s,
        inference_ms_per_batch=15.0,
    )


def _escribir_baselines_al_100(ruta_csv: Path, n_folds: int = 5) -> None:
    """Escribe las 10 celdas baseline al 100 % requeridas por TASK-12."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    for modelo in ("efficientnet_b0", "resnet50"):
        for fold in range(n_folds):
            registro = _registro_campana(
                modelo=modelo,
                data_fraction=1.0,
                fold=fold,
                train_time_s=600.0,
            )
            escribir_corrida_csv(registro, ruta_csv)


def _escribir_campana_completa(cfg: ExperimentConfig) -> None:
    """Escribe las 60 celdas del diseño factorial con historial."""
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    cfg.ensure_layout()
    for modelo, fraccion, fold in generar_celdas_design(cfg):
        n_capas = 6 if modelo == "hqcnn" else None
        registro = _registro_campana(
            modelo=modelo,
            data_fraction=fraccion,
            fold=fold,
            n_capas_vqc=n_capas,
        )
        escribir_corrida_csv(registro, ruta_csv)
        escribir_historial_json(registro, [], cfg)


def test_orden_modelos_hqcnn_ultimo_en_100() -> None:
    assert orden_modelos(1.0)[-1] == "hqcnn"
    assert orden_modelos(0.10) == MODELOS


def test_generar_celdas_design_total() -> None:
    cfg = ExperimentConfig()
    assert len(generar_celdas_design(cfg)) == N_CELDAS_TOTAL


def test_verificar_precondiciones_rechaza_sin_baselines(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    hparams = {"n_capas": 6, "presupuesto": {"decision": "go", "horas_campana_estimadas": 100.0}}

    with (
        patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams),
        patch("src.experiments.campana.cargar_splits", return_value={}),
    ):
        with pytest.raises(ValueError, match="Faltan celdas baseline"):
            verificar_precondiciones(cfg)


def test_verificar_precondiciones_ok_con_baselines(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    _escribir_baselines_al_100(cfg.raiz_resultados / "experiments.csv")
    hparams = {"n_capas": 6, "presupuesto": {"decision": "go", "horas_campana_estimadas": 100.0}}

    with (
        patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams),
        patch("src.experiments.campana.cargar_splits", return_value={}),
    ):
        resultado = verificar_precondiciones(cfg)

    assert resultado["n_capas"] == 6
    assert resultado["n_baselines_100"] == N_CELDAS_BASELINE_TASK12


def test_comparar_costo_calcula_desviacion(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    cfg.ensure_layout()
    registro = _registro_campana(train_time_s=3600.0)
    escribir_corrida_csv(registro, ruta_csv)

    hparams = {"presupuesto": {"horas_campana_estimadas": 2.0}}
    costo = comparar_costo(cfg, hparams=hparams)

    assert costo["horas_reales"] == 1.0
    assert costo["horas_estimadas"] == 2.0
    assert costo["desviacion_pct"] == -50.0


def test_guardar_estado_celda_fallida(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    celdas = _inicializar_estado_celdas(cfg)
    clave: ClaveCelda = ("hqcnn", 0.10, 0)

    _actualizar_celda_estado(
        celdas,
        clave,
        estado="fallida",
        motivo="RuntimeError: memoria agotada",
    )

    hparams = {"presupuesto": {"horas_campana_estimadas": 100.0}}
    with patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams):
        ruta = guardar_estado_campana(cfg, celdas, hparams=hparams)

    payload = json.loads(ruta.read_text(encoding="utf-8"))
    celda_fallida = next(c for c in payload["celdas"] if c["estado"] == "fallida")
    assert celda_fallida["motivo"] == "RuntimeError: memoria agotada"
    assert payload["resumen"]["fallidas"] == 1
    assert payload["resumen"]["pendientes"] == N_CELDAS_TOTAL - 1


def test_verificar_integridad_60_celdas(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _escribir_campana_completa(cfg)
    resultado = verificar_integridad(cfg)
    assert resultado["ok"] is True
    assert resultado["n_celdas_presentes"] == N_CELDAS_TOTAL


def test_verificar_integridad_detecta_duplicados(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _escribir_campana_completa(cfg)
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    registro = _registro_campana(modelo="efficientnet_b0", data_fraction=0.10, fold=0)
    escribir_corrida_csv(registro, ruta_csv)

    with pytest.raises(AssertionError, match="duplicadas"):
        verificar_integridad(cfg)


def test_verificar_integridad_falla_con_celdas_faltantes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    ruta_csv = cfg.raiz_resultados / "experiments.csv"
    registro = _registro_campana()
    escribir_corrida_csv(registro, ruta_csv)
    (cfg.raiz_resultados / "history").mkdir(parents=True, exist_ok=True)
    escribir_historial_json(registro, [], cfg)

    with pytest.raises(AssertionError, match="faltantes"):
        verificar_integridad(cfg)


def test_verificar_precondiciones_admite_baselines_archivadas(tmp_path: Path) -> None:
    """Con las corridas MPS archivadas (D3), la campaña arranca sin ellas en el CSV."""
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    hparams = {"n_capas": 6, "presupuesto": {"decision": "go", "horas_campana_estimadas": 100.0}}

    with (
        patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams),
        patch("src.experiments.campana.cargar_splits", return_value={}) as splits,
    ):
        resultado = verificar_precondiciones(cfg, exigir_baselines=False)

    assert resultado["n_capas"] == 6
    assert resultado["n_baselines_100"] == 0
    splits.assert_called_once()


def test_archivar_corridas_no_cuda_separa_y_conserva(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    ruta_csv = cfg.raiz_resultados / "experiments.csv"

    for fold in range(5):
        for modelo in ("efficientnet_b0", "resnet50"):
            registro = _registro_campana(
                modelo=modelo,
                data_fraction=1.0,
                fold=fold,
                dispositivo="mps",
            )
            escribir_corrida_csv(registro, ruta_csv)
            escribir_historial_json(registro, [], cfg)

    sonda = _registro_campana(data_fraction=0.10, fold=0, dispositivo="mps", epocas=1)
    escribir_corrida_csv(sonda, ruta_csv)
    escribir_historial_json(sonda, [], cfg)

    en_cuda = _registro_campana(data_fraction=0.25, fold=1, dispositivo="cuda")
    escribir_corrida_csv(en_cuda, ruta_csv)
    escribir_historial_json(en_cuda, [], cfg)

    hparams = {"presupuesto": {"horas_campana_estimadas": 100.0}}
    with patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams):
        resumen = archivar_corridas_no_cuda(cfg)

    assert resumen["historicas"] == 10
    assert resumen["informales"] == 1
    assert resumen["conservadas"] == 1
    assert resumen["historiales_movidos"] == 11

    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    assert [fila["dispositivo"] for fila in filas] == ["cuda"]

    ruta_historico = cfg.raiz_resultados / "historico_mps.csv"
    ruta_informales = cfg.raiz_resultados / "pruebas_informales.csv"
    with ruta_historico.open(newline="", encoding="utf-8") as arch:
        assert len(list(csv.DictReader(arch))) == 10
    with ruta_informales.open(newline="", encoding="utf-8") as arch:
        assert len(list(csv.DictReader(arch))) == 1

    archivados = list((cfg.raiz_resultados / "history_mps").glob("*.json"))
    assert len(archivados) == 11
    assert list((cfg.raiz_resultados / "history").glob("*.json")) != []


def test_archivar_corridas_no_cuda_repone_celdas_pendientes(tmp_path: Path) -> None:
    """Las celdas archivadas vuelven a pendiente para que no se omitan por reanudabilidad."""
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    ruta_csv = cfg.raiz_resultados / "experiments.csv"

    registro = _registro_campana(data_fraction=1.0, fold=0, dispositivo="mps")
    escribir_corrida_csv(registro, ruta_csv)

    celdas = _inicializar_estado_celdas(cfg)
    _actualizar_celda_estado(celdas, ("efficientnet_b0", 1.0, 0), estado="completada")

    hparams = {"presupuesto": {"horas_campana_estimadas": 100.0}}
    with patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams):
        guardar_estado_campana(cfg, celdas, hparams=hparams)
        archivar_corridas_no_cuda(cfg)

        estado = cargar_estado_campana(cfg)

    assert estado is not None
    celda = next(
        c
        for c in estado["celdas"]
        if (c["modelo"], c["data_fraction"], c["fold"]) == ("efficientnet_b0", 1.0, 0)
    )
    assert celda["estado"] == "pendiente"
    assert estado["resumen"]["completadas"] == 0


def test_archivar_corridas_no_cuda_es_idempotente(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_layout()
    escribir_corrida_csv(
        _registro_campana(dispositivo="cuda"),
        cfg.raiz_resultados / "experiments.csv",
    )

    hparams = {"presupuesto": {"horas_campana_estimadas": 100.0}}
    with patch("src.experiments.campana.cargar_hparams_congelados", return_value=hparams):
        resumen = archivar_corridas_no_cuda(cfg)

    assert resumen == {
        "conservadas": 1,
        "historicas": 0,
        "informales": 0,
        "historiales_movidos": 0,
    }
    assert not (cfg.raiz_resultados / "historico_mps.csv").exists()


def test_fracciones_campana_coinciden_con_splits() -> None:
    assert FRACCIONES == (0.10, 0.25, 0.50, 1.00)
