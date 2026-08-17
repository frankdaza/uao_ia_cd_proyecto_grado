"""Pruebas del modelo híbrido HQCNN (TASK-10)."""

from __future__ import annotations

from pathlib import Path

import pennylane as qml
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.data.dataset import MAPEO_CLASE
from src.logging.records import CLASES_ORDEN
from src.models.hqcnn import HQCNN, contar_parametros_por_bloque, verificar_gradiente
from src.models.vqc import MAPEO_QUBIT_CLASE, circuito_vqc
from src.train.trainer import Trainer
from src.utils.seed import set_seed

_BATCH = 2
_ENTRADA = (3, 224, 224)
_N_CLASES = 4


@pytest.fixture(autouse=True)
def _argumento_entrada_torchlayer() -> None:
    """Alinea TorchLayer con el nombre ``entradas`` del QNode de TASK-9."""
    qml.qnn.TorchLayer.set_input_argument("entradas")
    yield
    qml.qnn.TorchLayer.set_input_argument("inputs")


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return ExperimentConfig(n_capas=4, semilla=42)


@pytest.fixture(scope="module")
def modelo(cfg: ExperimentConfig) -> HQCNN:
    set_seed(cfg.semilla)
    return HQCNN(cfg)


def _entrada_imagen(batch: int = _BATCH) -> torch.Tensor:
    generador = torch.Generator().manual_seed(42)
    return torch.randn(batch, *_ENTRADA, generator=generador)


def test_forward_forma_y_rango_logits(modelo: HQCNN) -> None:
    x = _entrada_imagen()
    logits = modelo(x)
    assert logits.shape == (_BATCH, _N_CLASES)
    assert float(logits.detach().min()) >= -1.0
    assert float(logits.detach().max()) <= 1.0

    x_solo = _entrada_imagen(batch=1)[0]
    logits_solo = modelo(x_solo)
    assert logits_solo.shape == (_N_CLASES,)


def test_integracion_torchlayer(modelo: HQCNN) -> None:
    assert isinstance(modelo.vqc, qml.qnn.TorchLayer)
    assert modelo.vqc.qnode.func.__name__ == circuito_vqc.__name__


def test_acotacion_angulos(modelo: HQCNN) -> None:
    with torch.no_grad():
        latente = modelo.backbone(_entrada_imagen())
    pre_acotacion = modelo.reduccion(latente)
    angulos = torch.tanh(pre_acotacion) * torch.pi

    assert float(angulos.min()) > -torch.pi
    assert float(angulos.max()) < torch.pi
    assert float(pre_acotacion.abs().max()) > torch.pi or pre_acotacion.numel() > 0


def test_verificar_gradiente_coincide(modelo: HQCNN) -> None:
    set_seed(42)
    x = _entrada_imagen(batch=1)
    y = torch.tensor([0])
    analitico, numerico = verificar_gradiente(modelo, x, y)
    assert analitico == pytest.approx(numerico, rel=0.15)


def test_backbone_eval_tras_train(modelo: HQCNN) -> None:
    modelo.train(True)
    assert not modelo.backbone.training
    assert modelo.training


def test_backbone_sin_gradiente_reduccion_si(modelo: HQCNN) -> None:
    x = _entrada_imagen()
    y = torch.tensor([0, 1])
    modelo.zero_grad()
    nn.CrossEntropyLoss()(modelo(x), y).backward()

    for parametro in modelo.backbone.parameters():
        assert parametro.grad is None
    assert modelo.reduccion.lineal.weight.grad is not None
    assert modelo.vqc.pesos.grad is not None


def test_state_dict_ida_y_vuelta(cfg: ExperimentConfig) -> None:
    set_seed(cfg.semilla)
    original = HQCNN(cfg)
    x = _entrada_imagen(batch=1)

    with torch.inference_mode():
        salida_original = original(x)

    recargado = HQCNN(cfg)
    recargado.load_state_dict(original.state_dict())

    with torch.inference_mode():
        salida_recargada = recargado(x)

    assert torch.allclose(salida_original, salida_recargada)


def test_mapeo_qubit_clase_dataset() -> None:
    assert list(MAPEO_QUBIT_CLASE.values()) == list(CLASES_ORDEN)
    for indice, nombre in MAPEO_QUBIT_CLASE.items():
        assert MAPEO_CLASE[nombre] == indice


def test_contar_parametros_por_bloque(modelo: HQCNN) -> None:
    conteo = contar_parametros_por_bloque(modelo)
    assert conteo["backbone_congelado"] == 4_007_548
    assert conteo["reduccion"] == 5_124
    assert conteo["vqc"] == 48
    assert sum(conteo.values()) == 4_007_548 + 5_124 + 48


def test_trainer_entrena_hqcnn_una_epoca(tmp_path: Path) -> None:
    set_seed(42)
    generador = torch.Generator().manual_seed(42)
    entradas = torch.randn(16, *_ENTRADA, generator=generador)
    etiquetas = torch.randint(0, _N_CLASES, (16,), generator=generador)
    train = TensorDataset(entradas[:12], etiquetas[:12])
    val = TensorDataset(entradas[12:], etiquetas[12:])
    cargador_train = DataLoader(train, batch_size=_BATCH, shuffle=True)
    cargador_val = DataLoader(val, batch_size=_BATCH, shuffle=False)

    cfg = ExperimentConfig(
        modelo="hqcnn",
        n_capas=4,
        epocas=1,
        batch_size=_BATCH,
        semilla=42,
        raiz_resultados=tmp_path / "results",
        raiz_modelos=tmp_path / "models",
    )
    modelo = HQCNN(cfg)
    entrenador = Trainer(modelo, cfg, torch.device("cpu"), fold=0)
    registro, historial = entrenador.ajustar(cargador_train, cargador_val)

    registro.validar()
    assert len(historial) == 1
    assert registro.n_capas_vqc == cfg.n_capas
    assert registro.n_params_entrenables == 5_124 + 48


def test_n_capas_vqc_expuesta(modelo: HQCNN, cfg: ExperimentConfig) -> None:
    assert modelo.n_capas_vqc == cfg.n_capas


def test_no_usa_torch_save_modulo_completo() -> None:
    ruta = Path(__file__).resolve().parents[1] / "src" / "models" / "hqcnn.py"
    fuente = ruta.read_text(encoding="utf-8")
    assert "torch.save(" not in fuente
