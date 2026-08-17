"""Pruebas del bucle unificado de entrenamiento (TASK-8)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.logging.records import RunRecord
from src.logging.sinks import escribir_corrida_csv
from src.train.trainer import Trainer, nombre_pesos

_DIM_ENTRADA = 16
_N_CLASES = 4
_N_MUESTRAS = 128
_BATCH = 16
_EPOCAS = 3


class ModeloClasico(nn.Module):
    """Línea base trivial con logits no acotados."""

    def __init__(self) -> None:
        super().__init__()
        self.lineal = nn.Linear(_DIM_ENTRADA, _N_CLASES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lineal(x)


class ModeloCuanticoSimulado(nn.Module):
    """Simula salida de VQC en [-1, 1] sin importar PennyLane."""

    n_capas_vqc = 2

    def __init__(self) -> None:
        super().__init__()
        self.lineal = nn.Linear(_DIM_ENTRADA, _N_CLASES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lineal(x))


def _semilla_fija() -> None:
    torch.manual_seed(42)


def _cargadores_sinteticos() -> tuple[DataLoader, DataLoader]:
    _semilla_fija()
    generador = torch.Generator().manual_seed(42)
    entradas = torch.randn(_N_MUESTRAS, _DIM_ENTRADA, generator=generador)
    etiquetas = torch.randint(0, _N_CLASES, (_N_MUESTRAS,), generator=generador)
    train = TensorDataset(entradas[:96], etiquetas[:96])
    val = TensorDataset(entradas[96:], etiquetas[96:])
    cargador_train = DataLoader(train, batch_size=_BATCH, shuffle=True)
    cargador_val = DataLoader(val, batch_size=_BATCH, shuffle=False)
    return cargador_train, cargador_val


def _cfg(modelo: str, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        modelo=modelo,
        epocas=_EPOCAS,
        batch_size=_BATCH,
        lr=1e-3,
        semilla=42,
        raiz_resultados=tmp_path / "results",
        raiz_modelos=tmp_path / "models",
    )


def _entrenar(modelo: nn.Module, cfg: ExperimentConfig) -> tuple[RunRecord, Trainer]:
    cargador_train, cargador_val = _cargadores_sinteticos()
    entrenador = Trainer(modelo, cfg, torch.device("cpu"), fold=0)
    registro, historial = entrenador.ajustar(cargador_train, cargador_val)
    return registro, entrenador


def test_trainer_no_importa_pennylane_ni_torchvision() -> None:
    ruta = Path(__file__).resolve().parents[1] / "src" / "train" / "trainer.py"
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    imports = {
        n.name
        for n in ast.walk(arbol)
        if isinstance(n, ast.Import)
        for n in n.names
    } | {
        n.module.split(".")[0]
        for n in ast.walk(arbol)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "pennylane" not in imports
    assert "torchvision" not in imports


def test_mismo_trainer_entrena_modelo_clasico_y_cuantico_simulado(
    tmp_path: Path,
) -> None:
    registro_clasico, _ = _entrenar(ModeloClasico(), _cfg("efficientnet_b0", tmp_path))
    registro_cuantico, _ = _entrenar(
        ModeloCuanticoSimulado(),
        _cfg("hqcnn", tmp_path),
    )

    registro_clasico.validar()
    registro_cuantico.validar()
    assert registro_clasico.n_capas_vqc is None
    assert registro_cuantico.n_capas_vqc == 2
    assert registro_clasico.epocas == _EPOCAS
    assert registro_cuantico.epocas == _EPOCAS


def test_presupuesto_fijo_de_epocas_sin_seleccion_por_validacion(
    tmp_path: Path,
) -> None:
    cargador_train, cargador_val = _cargadores_sinteticos()
    cfg = _cfg("efficientnet_b0", tmp_path)
    entrenador = Trainer(ModeloClasico(), cfg, torch.device("cpu"), fold=0)
    _, historial = entrenador.ajustar(cargador_train, cargador_val)

    assert len(historial) == cfg.epocas
    assert [epoca.epoca for epoca in historial] == list(range(cfg.epocas))


def test_run_record_emite_contrato_completo(tmp_path: Path) -> None:
    registro, _ = _entrenar(ModeloClasico(), _cfg("efficientnet_b0", tmp_path))
    registro.validar()
    assert registro.modelo == "efficientnet_b0"
    assert registro.fold == 0
    assert registro.semilla == 42
    assert 0.0 <= registro.f1_val_weighted <= 1.0
    assert set(registro.sensibilidad_por_clase) == {
        "glioma",
        "meningioma",
        "pituitary",
        "notumor",
    }


def test_guardar_y_recargar_state_dict_reproduce_metricas(tmp_path: Path) -> None:
    cargador_train, cargador_val = _cargadores_sinteticos()
    cfg = _cfg("efficientnet_b0", tmp_path)
    modelo_original = ModeloClasico()
    entrenador = Trainer(modelo_original, cfg, torch.device("cpu"), fold=0)
    registro_original, _ = entrenador.ajustar(cargador_train, cargador_val)
    ruta = entrenador.guardar_pesos()

    estado = torch.load(ruta, weights_only=True)
    assert isinstance(estado, dict)
    assert all(isinstance(k, str) for k in estado)

    modelo_recargado = ModeloClasico()
    entrenador_recargado = Trainer(modelo_recargado, cfg, torch.device("cpu"), fold=0)
    entrenador_recargado.cargar_pesos(ruta)

    _semilla_fija()
    entrada = torch.randn(8, _DIM_ENTRADA)
    with torch.inference_mode():
        salida_original = modelo_original(entrada)
        salida_recargada = modelo_recargado(entrada)
    assert torch.allclose(salida_original, salida_recargada)

    metricas_val = entrenador_recargado._evaluar_completo(cargador_val)
    assert float(metricas_val["accuracy"]) == pytest.approx(registro_original.accuracy_val)
    assert float(metricas_val["loss"]) == pytest.approx(registro_original.loss_val)


def test_corrida_completada_detecta_tupla_completa(tmp_path: Path) -> None:
    cfg = _cfg("hqcnn", tmp_path)
    cfg.ensure_layout()
    entrenador = Trainer(ModeloCuanticoSimulado(), cfg, torch.device("cpu"), fold=1)
    assert entrenador.corrida_completada() is False

    registro = RunRecord(
        modelo=cfg.modelo,
        data_fraction=cfg.data_fraction,
        fold=1,
        semilla=cfg.semilla,
        dispositivo="cpu",
        n_train=10,
        n_val=5,
        epocas=cfg.epocas,
        n_params_entrenables=100,
        n_capas_vqc=2,
        commit_sha="abc1234",
        timestamp="2026-08-17T01:00:00+00:00",
        accuracy_train=0.5,
        accuracy_val=0.5,
        loss_train=0.5,
        loss_val=0.5,
        f1_val_weighted=0.5,
        f1_val_macro=0.5,
        sensibilidad_por_clase={
            "glioma": 0.5,
            "meningioma": 0.5,
            "pituitary": 0.5,
            "notumor": 0.5,
        },
        especificidad_por_clase={
            "glioma": 0.5,
            "meningioma": 0.5,
            "pituitary": 0.5,
            "notumor": 0.5,
        },
        train_time_s=1.0,
        inference_ms_per_batch=1.0,
    )
    escribir_corrida_csv(registro, cfg.raiz_resultados / "experiments.csv")
    assert entrenador.corrida_completada() is True

    otro_fold = Trainer(ModeloCuanticoSimulado(), cfg, torch.device("cpu"), fold=9)
    assert otro_fold.corrida_completada() is False


def test_nombre_pesos_incluye_semilla() -> None:
    assert nombre_pesos("hqcnn", 0.25, 2, 42) == "hqcnn_0p25_f2_s42.pt"


def test_modulo_train_exporta_api_publica() -> None:
    spec = importlib.util.find_spec("src.train")
    assert spec is not None
    modulo = importlib.import_module("src.train")
    assert hasattr(modulo, "Trainer")
    assert hasattr(modulo, "construir_loaders_para_fold")
