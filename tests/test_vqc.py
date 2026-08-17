"""Pruebas del clasificador cuántico variacional (TASK-9)."""

from __future__ import annotations

from pathlib import Path

import pennylane as qml
import pytest
import torch

from src.data.dataset import MAPEO_CLASE
from src.logging.records import CLASES_ORDEN
from src.models.vqc import (
    ESCALA_INICIALIZACION,
    MAPEO_QUBIT_CLASE,
    N_QUBITS,
    circuito_vqc,
    forma_pesos_vqc,
    inicializar_pesos,
    norma_gradiente_inicial,
)


def test_mapeo_qubit_clase_coincide_con_dataset() -> None:
    assert list(MAPEO_QUBIT_CLASE.values()) == list(CLASES_ORDEN)
    for indice, nombre in MAPEO_QUBIT_CLASE.items():
        assert MAPEO_CLASE[nombre] == indice


def test_forma_pesos_vqc_usa_shape_del_ansatz() -> None:
    for n_capas in (2, 4, 6):
        esperada = qml.StronglyEntanglingLayers.shape(n_layers=n_capas, n_wires=N_QUBITS)
        assert forma_pesos_vqc(n_capas) == esperada


def test_inicializar_pesos_forma_y_escala() -> None:
    n_capas = 4
    pesos = inicializar_pesos(n_capas)
    assert pesos.shape == forma_pesos_vqc(n_capas)
    assert float(pesos.abs().max()) <= ESCALA_INICIALIZACION * 5


def test_circuito_vqc_salida_en_rango() -> None:
    entradas = torch.rand(N_QUBITS) * torch.pi
    pesos = inicializar_pesos(2)
    salidas = circuito_vqc(entradas, pesos)
    assert len(salidas) == N_QUBITS
    for valor in salidas:
        assert -1.0 <= float(valor) <= 1.0


def test_circuito_vqc_propaga_gradiente() -> None:
    entradas = torch.rand(N_QUBITS) * torch.pi
    pesos = inicializar_pesos(2).requires_grad_(True)
    salida = torch.stack(circuito_vqc(entradas, pesos)).sum()
    salida.backward()
    assert pesos.grad is not None
    assert float(pesos.grad.norm()) > 0.0


def test_norma_gradiente_inicial_positiva() -> None:
    norma = norma_gradiente_inicial(2, n_muestras=3, semilla=42)
    assert norma > 0.0


def test_no_usa_api_deprecada_pauliz() -> None:
    ruta = Path(__file__).resolve().parents[1] / "src" / "models" / "vqc.py"
    fuente = ruta.read_text(encoding="utf-8")
    assert "PauliZ" not in fuente
